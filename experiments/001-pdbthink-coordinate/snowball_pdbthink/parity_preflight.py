"""Exercise the native CPU batch path before allocating diagnostic GPUs."""

import gzip
import json
import tempfile
from types import SimpleNamespace

import numpy as np
from finestore.reader import ReadView
from finestore.rl import mismatch_probe as schema
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.mismatch_probe.archive import COMPLETE_STATUS, MismatchArchive
from skyrl_train.mismatch_probe.collect import ProbeCollector
from skyrl_train.mismatch_probe.provenance import manifest
from skyrl_train.trainer import RayPPOTrainer

from .parity_runtime import collate, decode_control_inputs, write_archive


def check_archive(probe, trainer):
    """Round-trip full-sized synthetic routing observations through native FineStore."""
    scores = []
    for row in probe.probes:
        routes = np.frombuffer(row.routed_experts, dtype=np.uint8).reshape(row.routed_experts_shape).astype(np.int32)
        for mode in ("native", "repeat", "router_replay"):
            scores.append(
                schema.ScoreRow(
                    probe_hash=probe.probe_hash,
                    sample_id=row.sample_id,
                    scorer="trainer",
                    mode=mode,
                    update=0,
                    global_step=0,
                    logprobs=[-1.0] * len(row.vllm_output_ids),
                    expert_choices=routes.tobytes(),
                    expert_choices_shape=list(routes.shape),
                    expert_choices_dtype=str(routes.dtype),
                    replacement_mask=np.zeros_like(routes, dtype=np.bool_).tobytes(),
                )
            )
    with tempfile.TemporaryDirectory(prefix="snowball-parity-archive-preflight-") as directory:
        probe.archive_uri = directory
        probe.tokenizer_fingerprint = "synthetic-cpu-preflight"
        probe.starting_global_step = 0
        probe.scored_global_steps = {0: 0}
        archive = MismatchArchive(directory, writer_id="cpu-preflight")
        try:
            write_archive(
                archive,
                probes=probe.probes,
                scores=scores,
                manifest=manifest(probe, trainer, status=COMPLETE_STATUS),
            )
        finally:
            archive.close()
        view = ReadView(directory)
        stored_probes = view.scan("probe")
        stored_scores = view.scan("scores")
        if stored_probes.num_rows != len(probe.probes) or stored_scores.num_rows != len(scores):
            raise ValueError("Preflight archive lost rows")
        indexed = {(row["sample_id"], row["mode"]): row for row in stored_scores.to_pylist()}
        for row in scores:
            stored = indexed[row.sample_id, row.mode]
            for field in ("logprobs", "expert_choices", "expert_choices_shape", "expert_choices_dtype", "replacement_mask"):
                if stored[field] != getattr(row, field):
                    raise ValueError(f"Preflight archive changed {field}")
        if view.scan("manifest").num_rows != 1:
            raise ValueError("Preflight archive has no unique completion manifest")
        return {"score_rows": stored_scores.num_rows, "uncompressed_score_bytes": stored_scores.nbytes}


def check_collation(config, panel_path):
    panel = json.loads(gzip.decompress(panel_path.read_bytes()))
    defaults = get_default_config()
    OmegaConf.set_struct(defaults, False)
    cfg = OmegaConf.merge(defaults, config)
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.group_advantage_invariant = GroupAdvantageInvariant.exact_physical(physical_group_size=4)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=16))])
    trainer.critic_model = trainer.ref_model = None
    trainer._num_experts_cache = 256
    trainer.all_metrics = {}
    trainer.global_step = 0
    trainer.loaded_checkpoint_path = "synthetic-cpu-preflight"
    auxiliary, samples = decode_control_inputs(panel, 16)
    checked = []
    for name, inputs, group_size, cap in (
        ("fixed-panel", panel["rows"], 2, 2048),
        ("decode-control", auxiliary, samples, 256),
    ):
        probe = ProbeCollector(cfg)
        probe.probe_hash = "cpu-preflight-only"
        probe.probes = []
        scores, routes = [], []
        for index, row in enumerate(inputs):
            tokens = row["response_token_ids"][:cap]
            probe.probes.append(
                schema.ProbeRow(
                    probe_hash="cpu-preflight-only",
                    sample_id=f"preflight-{index}",
                    prompt_id=row["prompt_id"],
                    prompt_token_ids=row["prompt_token_ids"],
                    trainer_prompt_ids=row["prompt_token_ids"],
                    vllm_output_ids=tokens,
                    trainer_input_ids=tokens,
                    response_mask=[True] * len(tokens),
                    loss_mask=row["loss_mask"][:cap],
                    request_seed=0,
                    batch_position=index,
                )
            )
            scores.append(SimpleNamespace(logprobs=[-1.0] * len(tokens)))
            route = np.broadcast_to(np.arange(1, 5, dtype=np.uint8), (len(tokens), 26, 4)).copy()
            route[-1] = 0
            routes.append(route)
        collate(probe, trainer, scores, routes, samples_per_prompt=group_size)
        batch = probe.training_input
        if batch.batch_size % 32 or any(
            value.shape[0] != batch.batch_size for value in batch.values() if value is not None
        ):
            raise ValueError("Preflight batch does not fill both forward micro-batch layouts")
        if trainer.group_advantage_invariant.physical_group_size != 4:
            raise ValueError("Preflight group contract was not restored")
        archived = check_archive(probe, trainer)
        if name == "fixed-panel" and archived["uncompressed_score_bytes"] <= 104857600:
            raise ValueError("Preflight did not exercise the original oversized-transaction case")
        checked.append({"batch": name, "sequences": len(inputs), "padded_rows": batch.batch_size, **archived})
    print(json.dumps({"native_collation_preflight": checked, "model_probabilities_computed": False}), flush=True)
