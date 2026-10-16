"""Read-only frozen-token comparison inside the pinned MarinSkyRL runtime.

Installed only in the diagnostic jobs. Native weight loading, broadcast, model
forward, log-probability extraction, and router replay remain upstream code.
"""

import gzip
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import ray
from finestore.rl import mismatch_probe as schema
from loguru import logger
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.callbacks.base import TrainerCallback
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.mismatch_probe.archive import COMPLETE_STATUS, MismatchArchive
from skyrl_train.mismatch_probe.collect import BatchLayout, ProbeCollector, _candidate_logprobs, _prompt_group_contract
from skyrl_train.mismatch_probe.protocol import ProbeSample, probe_hash
from skyrl_train.mismatch_probe.provenance import manifest
from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync


def write_json(root, name, data):
    with (StoragePath(root) / name).open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def summarize(reference, candidate):
    delta = np.concatenate(
        [
            np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
            for a, b in zip(reference, candidate, strict=True)
        ]
    )
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite probability comparison")
    weights = np.exp(delta - delta.max())
    return {
        "tokens": len(delta),
        "mean_log_ratio": float(delta.mean()),
        "mean_abs_log_ratio": float(np.abs(delta).mean()),
        "abs_log_ratio_p50_p90_p99_p999_max": np.quantile(np.abs(delta), [0.5, 0.9, 0.99, 0.999, 1]).tolist(),
        "k3": float(np.mean(np.expm1(delta) - delta)),
        "share_beyond_2x": float(np.mean(np.abs(delta) > np.log(2))),
        "ess_fraction": float(weights.sum() ** 2 / (len(weights) * (weights**2).sum())),
        "exact_fraction": float(np.mean(delta == 0)),
    }


class WeightVerifier(ExpertBlockSync):
    """Use native verification collectives, retaining every participant report."""

    async def _policy(self, method, *args):
        rows = await super()._policy(method, *args)
        if method == "verify":
            self.policy_reports = rows
        return rows

    async def _receivers(self, method, *args):
        rows = await super()._receivers(method, *args)
        if method == "verify":
            self.receiver_reports = rows
        return rows


async def verify_weights(trainer, root):
    verifier = WeightVerifier(
        policy_model=trainer.policy_model, inference_engine_client=trainer.inference_engine_client, timeout_seconds=1200
    )
    await trainer.inference_engine_client.pause_generation()
    try:
        preparation = await verifier.prepare()
        try:
            timing = await verifier.verify(trainer.global_step)
        finally:
            write_json(
                root,
                "weight-verification.json",
                {
                    "global_step": trainer.global_step,
                    "preparation": preparation,
                    "policy": getattr(verifier, "policy_reports", []),
                    "receivers": getattr(verifier, "receiver_reports", []),
                    "method": "Original broadcast installation; independent native expert-block scratch replay, no install",
                },
            )
        logger.info("PARITY all-parameter weight verification passed: {}", timing)
    finally:
        await verifier.close()
        await trainer.inference_engine_client.resume_generation()
    # The native scratch comparison normalizes widened router dtype. Check
    # router weights and bias in full FP32 separately, including every replica.
    names = [f"model.layers.{layer}.mlp.router.{suffix}" for layer in range(26) for suffix in ("weight", "bias")]
    snapshots = ray.get(trainer.policy_model.async_run_ray_method("pass_through", "grug_validation_snapshot", names))
    reference = next(row.weights for row in snapshots if row.rank == 0)
    reports = []
    for index, engine in enumerate(trainer.inference_engine_client.engines):
        workers = await engine.inference_engine_actor.read_engine_weights.remote(names, False)
        for worker in workers:
            for name in names:
                value = worker[name]
                if not value["found"]:
                    raise ValueError(f"Router tensor not found: {name}")
                expected, actual = reference[name], value["tensor"]
                reports.append(
                    {
                        "engine": index,
                        "ranks": worker["__ranks__"],
                        "name": name,
                        "dtype": value["dtype"],
                        "elements": expected.numel(),
                        "unequal": int((expected != actual).sum()),
                        "max_abs": float((expected - actual).abs().max()),
                    }
                )
    write_json(root, "router-weight-verification.json", reports)
    if any(row["unequal"] or row["dtype"] != "float32" for row in reports):
        raise ValueError("FP32 router state differs after broadcast")


async def score_vllm(probe, trainer):
    """Score every fixed response token and capture routes for that same replay."""
    await trainer.inference_engine_client.reset_prefix_cache()
    rows = probe.probes
    overrides = [{"logprob_token_ids": [row.vllm_output_ids[-1]]} for row in rows]
    output = await trainer.inference_engine_client.generate(
        {
            "prompts": None,
            "prompt_token_ids": [row.prompt_token_ids + row.vllm_output_ids[:-1] for row in rows],
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_tokens": 0,
                "prompt_logprobs": 1,
                "logprobs": 1,
                "skip_reading_prefix_cache": True,
                "seed": int(probe.spec.seed),
            },
            "sampling_params_per_prompt": overrides,
            "session_ids": None,
        }
    )
    final_scores = _candidate_logprobs(output, overrides)
    scores, routes = [], []
    for row, values, final, raw_routes in zip(
        rows, output["prompt_logprobs"], final_scores, output["routed_experts"], strict=True
    ):
        n = len(row.prompt_token_ids)
        chosen = [float(values[n + i][token]) for i, token in enumerate(row.vllm_output_ids[:-1])] + [final]
        if not np.isfinite(chosen).all():
            raise ValueError("vLLM omitted a frozen token probability")
        if raw_routes is None or raw_routes.shape != (n + len(chosen) - 1, 26, 4):
            raise ValueError(f"Unexpected vLLM captured-route shape for {row.sample_id}")
        route = np.zeros((len(chosen), 26, 4), dtype=np.uint8)
        if raw_routes.min() < 0 or raw_routes.max() >= 256:
            raise ValueError("Captured expert IDs outside Snowball's expert range")
        route[:-1] = raw_routes[n:]
        routes.append(route)
        scores.append(
            schema.ScoreRow(
                probe_hash=probe.probe_hash,
                sample_id=row.sample_id,
                scorer="vllm.rescore",
                update=0,
                global_step=trainer.global_step,
                cache_mode="off",
                logprobs=chosen,
            )
        )
    return scores, routes


def collate(probe, trainer, scores, routes, samples_per_prompt=2):
    rows = probe.probes
    trajectory = {
        "prompt_token_ids": [row.prompt_token_ids for row in rows],
        "response_ids": [row.vllm_output_ids for row in rows],
        "loss_masks": [list(map(int, row.loss_mask)) for row in rows],
        "rewards": [[0.0] * len(row.vllm_output_ids) for row in rows],
        "rollout_logprobs": [np.asarray(row.logprobs, dtype=np.float32) for row in scores],
        "rollout_routed_experts": routes,
        "stop_reasons": ["frozen_replay"] * len(rows),
        "rollout_metrics": None,
    }
    original_contract = trainer.group_advantage_invariant
    try:
        trainer.group_advantage_invariant = _prompt_group_contract(trainer, samples_per_prompt)
        batch = trainer.convert_to_training_input(trajectory, [row.prompt_id for row in rows])
    finally:
        trainer.group_advantage_invariant = original_contract
    width = batch["response_mask"].shape[1]
    prefix_width = batch["sequences"].shape[1] - width
    frozen = []
    for index, (row, route) in enumerate(zip(rows, routes, strict=True)):
        prompt = batch["sequences"][index, :prefix_width][batch["attention_mask"][index, :prefix_width].bool()].tolist()
        response = batch["sequences"][index, prefix_width : prefix_width + len(row.vllm_output_ids)].tolist()
        assert prompt == row.prompt_token_ids and response == row.vllm_output_ids
        assert batch["loss_mask"][index, : len(response)].bool().tolist() == row.loss_mask
        frozen.append(
            row.model_copy(
                update={
                    "routed_experts": route.tobytes(),
                    "routed_experts_shape": list(route.shape),
                    "routed_experts_dtype": str(route.dtype),
                    "route_valid_mask": (route != 0).any(axis=-1).tolist(),
                }
            )
        )
    probe.probes = frozen
    probe.training_input = batch
    probe.batch_layout = BatchLayout(
        sample_ids=[row.sample_id for row in rows],
        native_order=list(range(len(rows))),
        repeat_order=list(reversed(range(len(rows)))),
        padded_rows=len(batch["sequences"]) - len(rows),
        native_micro_batch_size=1,
        repeat_micro_batch_size=2,
        collated_bytes=sum(value.numel() * value.element_size() for value in batch.values() if value is not None),
        collated_route_bytes=batch.routed_experts.nbytes,
        nonzero_advantage_samples=0,
    )


def decode_control_inputs(panel, dp_size):
    """Give every data-parallel rank at least two rows, including the repeat pass."""
    chosen = {}
    for row in panel["rows"]:
        if row["family"] in {"G01", "G03", "P03", "S09"}:
            chosen.setdefault(row["family"], row)
    if len(chosen) != 4:
        raise ValueError("The decode control requires all four selected question categories")
    samples_per_prompt = (2 * dp_size + len(chosen) - 1) // len(chosen)
    return [row for row in chosen.values() for _ in range(samples_per_prompt)], samples_per_prompt


async def decode_control(trainer, panel, root):
    """Separate fresh decode/pre-fill differences from the common-panel comparison."""
    inputs, samples_per_prompt = decode_control_inputs(panel, trainer.policy_model.actor_infos[0].rank.dp_size)
    generated = await trainer.inference_engine_client.generate(
        {
            "prompts": None,
            "prompt_token_ids": [row["prompt_token_ids"] for row in inputs],
            "sampling_params": {
                "max_tokens": 256,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_tokens": 0,
                "logprobs": 0,
                "seed": panel["seed"],
            },
            "sampling_params_per_prompt": [{"seed": panel["seed"] + index} for index in range(len(inputs))],
            "session_ids": None,
        }
    )
    probe = ProbeCollector(trainer.cfg)
    probe.archive_uri = root + "/decode-control"
    probe.starting_global_step = trainer.global_step
    probe.scored_global_steps = {0: trainer.global_step}
    probe.probes = []
    for index, (row, tokens) in enumerate(zip(inputs, generated["response_ids"], strict=True)):
        probe.probes.append(
            schema.ProbeRow(
                probe_hash="pending",
                sample_id=row["prompt_id"] + f":decode-{index}",
                prompt_id=row["prompt_id"],
                prompt_token_ids=row["prompt_token_ids"],
                trainer_prompt_ids=row["prompt_token_ids"],
                vllm_output_ids=tokens,
                trainer_input_ids=tokens,
                response_mask=[True] * len(tokens),
                loss_mask=[True] * len(tokens),
                request_seed=panel["seed"],
                batch_position=index,
            )
        )
    probe.probe_hash = probe_hash(
        ProbeSample(
            row.sample_id,
            tuple(row.prompt_token_ids),
            tuple(row.vllm_output_ids),
            tuple(row.response_mask),
            tuple(row.loss_mask),
        )
        for row in probe.probes
    )
    probe.probes = [row.model_copy(update={"probe_hash": probe.probe_hash}) for row in probe.probes]
    rescored, _ = await score_vllm(probe, trainer)
    routes = []
    for row, captured in zip(probe.probes, generated["routed_experts"], strict=True):
        n = len(row.prompt_token_ids)
        if captured.shape != (n + len(row.vllm_output_ids) - 1, 26, 4):
            raise ValueError("Decode-control capture does not align with its tokens")
        route = np.zeros((len(row.vllm_output_ids), 26, 4), dtype=np.uint8)
        route[:-1] = captured[n:]
        routes.append(route)
    collate(probe, trainer, rescored, routes, samples_per_prompt=samples_per_prompt)
    trained = probe._trainer_scores(trainer, update=0)
    generation_scores = [
        schema.ScoreRow(
            probe_hash=probe.probe_hash,
            sample_id=row.sample_id,
            scorer="vllm.generate",
            update=0,
            global_step=trainer.global_step,
            logprobs=values,
        )
        for row, values in zip(probe.probes, generated["response_logprobs"], strict=True)
    ]
    reference = [row.logprobs for row in generation_scores]
    metrics = {"prefill_vs_decode": summarize(reference, [row.logprobs for row in rescored])}
    for mode in ("native", "repeat", *probe.spec.extra_trainer_modes):
        indexed = {row.sample_id: row for row in trained if row.mode == mode}
        metrics[mode] = summarize(reference, [indexed[row.sample_id].logprobs for row in probe.probes])
    probe.metrics = metrics
    archive = MismatchArchive(probe.archive_uri, writer_id=f"decode-{trainer.global_step}")
    try:
        archive.write(
            probes=probe.probes,
            scores=rescored + trained + generation_scores,
            manifest=manifest(probe, trainer, status=COMPLETE_STATUS),
        )
    finally:
        archive.close()
    write_json(
        probe.archive_uri,
        "complete.json",
        {
            "global_step": trainer.global_step,
            "metrics": metrics,
            "note": "Checkpoint-specific generated answers; auxiliary control",
        },
    )
    return metrics


class ParityCallback(TrainerCallback):
    error_behavior = "raise"

    def __init__(self, cfg):
        self.cfg = cfg

    async def on_train_begin_async(self, state, control, **kwargs):
        trainer = kwargs["trainer"]
        control.step_limit = state.global_step
        control.should_training_stop = True
        if self.cfg.trainer.mismatch_probe.updates != 0:
            raise ValueError("The parity diagnostic forbids optimizer updates")
        root = self.cfg.trainer.mismatch_probe.archive_uri
        write_json(root, "started.json", {"global_step": trainer.global_step, "at": datetime.now(UTC).isoformat()})
        with StoragePath(os.environ["SNOWBALL_PARITY_PANEL_URI"]).open("rb") as stream:
            compressed = stream.read()
        if hashlib.sha256(compressed).hexdigest() != os.environ["SNOWBALL_PARITY_PANEL_SHA256"]:
            raise ValueError("Frozen panel checksum mismatch")
        panel = json.loads(gzip.decompress(compressed))
        await verify_weights(trainer, root)
        probe = ProbeCollector(self.cfg)
        probe.starting_global_step = trainer.global_step
        probe.scored_global_steps = {0: trainer.global_step}
        probe.tokenizer_fingerprint = tokenizer_vocabulary_fingerprint(trainer.inference_engine_client.tokenizer)
        probe.probes = [
            schema.ProbeRow(
                probe_hash="pending",
                sample_id=row["sample_id"],
                prompt_id=row["prompt_id"],
                prompt_token_ids=row["prompt_token_ids"],
                trainer_prompt_ids=row["prompt_token_ids"],
                vllm_output_ids=row["response_token_ids"],
                trainer_input_ids=row["response_token_ids"],
                response_mask=[True] * len(row["response_token_ids"]),
                loss_mask=list(map(bool, row["loss_mask"])),
                request_seed=panel["seed"],
                batch_position=index,
            )
            for index, row in enumerate(panel["rows"])
        ]
        probe.probe_hash = probe_hash(
            ProbeSample(
                row.sample_id,
                tuple(row.prompt_token_ids),
                tuple(row.vllm_output_ids),
                tuple(row.response_mask),
                tuple(row.loss_mask),
            )
            for row in probe.probes
        )
        probe.probes = [row.model_copy(update={"probe_hash": probe.probe_hash}) for row in probe.probes]
        vllm_scores, routes = await score_vllm(probe, trainer)
        collate(probe, trainer, vllm_scores, routes)
        archive = MismatchArchive(root, writer_id=f"parity-{trainer.global_step}")
        try:
            archive.write(probes=probe.probes, scores=vllm_scores)
            logger.info(
                "PARITY vLLM frozen replay complete: {} response tokens", sum(len(r.logprobs) for r in vllm_scores)
            )
            trainer_scores = probe._trainer_scores(trainer, update=0)
            archive.write(scores=trainer_scores)
            # Repeat vLLM after the learner forwards, with unchanged parameters.
            repeated, _ = await score_vllm(probe, trainer)
            repeated = [row.model_copy(update={"mode": "repeat"}) for row in repeated]
            archive.write(scores=repeated)
            reference = [row.logprobs for row in vllm_scores]
            metrics = {"vllm_repeat": summarize(reference, [row.logprobs for row in repeated])}
            for mode in ("native", "repeat", *probe.spec.extra_trainer_modes):
                indexed = {row.sample_id: row for row in trainer_scores if row.mode == mode}
                metrics[mode] = summarize(reference, [indexed[row.sample_id].logprobs for row in probe.probes])
            probe.metrics = metrics
            archive.write(manifest=manifest(probe, trainer, status=COMPLETE_STATUS))
            fixed_result = {
                "global_step": trainer.global_step,
                "checkpoint": str(trainer.loaded_checkpoint_path),
                "optimizer_updates": 0,
                "probe_hash": probe.probe_hash,
                "panel_sha256": hashlib.sha256(compressed).hexdigest(),
                "sequences": len(probe.probes),
                "metrics": metrics,
                "batch_layout": asdict(probe.batch_layout),
                "at": datetime.now(UTC).isoformat(),
            }
            write_json(root, "fixed-panel-complete.json", fixed_result)
            auxiliary = await decode_control(trainer, panel, root)
            write_json(root, "complete.json", {**fixed_result, "decode_control": auxiliary})
            logger.info("PARITY COMPLETE step={} metrics={}", trainer.global_step, json.dumps(metrics))
        finally:
            archive.close()
        return control
