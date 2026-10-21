"""Checks and durable metrics for the route-replay training retry."""

import hashlib
import json
import math
import os

import ray
import torch
from loguru import logger
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.callbacks.base import TrainerCallback
from snowball_parity_runtime import verify_weights, write_json


def frozen_biases(worker):
    values = {
        f"{index}:{name}": value.detach().clone()
        for index, chunk in enumerate(worker.actor_module)
        for name, value in chunk.named_buffers()
        if name.endswith("router.expert_bias")
    }
    if not values:
        raise ValueError("No frozen Grug routing biases found")
    return values


def check_biases(worker, before):
    after = frozen_biases(worker)
    if before.keys() != after.keys() or any(not torch.equal(value, after[name]) for name, value in before.items()):
        raise ValueError("Frozen routing bias changed during an optimizer update")


class ReplayAuditCallback(TrainerCallback):
    error_behavior = "raise"

    def __init__(self, cfg):
        self.root = os.environ["SNOWBALL_REPLAY_AUDIT_URI"]
        self.bias_hash = None
        self.cfg = cfg

    async def bias_digest(self, trainer):
        names = [f"model.layers.{layer}.mlp.router.bias" for layer in range(26)]
        snapshots = ray.get(
            trainer.policy_model.async_run_ray_method("pass_through", "grug_validation_snapshot", names)
        )
        reference = next(row.weights for row in snapshots if row.rank == 0)
        digest = hashlib.sha256()
        for name in names:
            digest.update(name.encode())
            digest.update(reference[name].numpy().tobytes())
        return digest.hexdigest()

    async def on_train_begin_async(self, state, control, **kwargs):
        trainer = kwargs["trainer"]
        if self.cfg.trainer.policy.grug_query_bias_update_mode != "frozen":
            raise ValueError("This experiment requires frozen quantile/query bias")
        await verify_weights(trainer, self.root + f"/startup-{state.global_step}")
        self.bias_hash = await self.bias_digest(trainer)
        write_json(
            self.root,
            f"startup-{state.global_step}.json",
            {
                "global_step": state.global_step,
                "frozen_bias_sha256": self.bias_hash,
                "router_replay": True,
                "importance_correction_cap": 4.0,
                "overlay_sha256": os.environ["SNOWBALL_REPLAY_CODE_SHA256"],
            },
        )
        logger.info("REPLAY_AUDIT startup weight equality verified; bias={}", self.bias_hash)
        baseline_uri = os.environ.get("SNOWBALL_REPLAY_BASELINE_URI")
        if baseline_uri:
            if state.global_step != 0 or self.bias_hash != os.environ["SNOWBALL_REPLAY_BASELINE_BIAS_SHA256"]:
                raise ValueError("Baseline reuse requires the verified original step-zero policy")
            with StoragePath(baseline_uri).open("rb") as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != os.environ["SNOWBALL_REPLAY_BASELINE_SHA256"]:
                raise ValueError("Reused baseline checksum mismatch")
            baseline = json.loads(payload)
            if baseline["global_step"] != 0:
                raise ValueError("Only an unchanged step-zero baseline may be reused")
            metrics = {**baseline["metrics"], "eval/baseline_reused": 1.0}
            self.on_evaluate(state, control, metrics=metrics, baseline_origin=baseline_uri)
            trainer._log_metrics_stdout(metrics, step=0, kind="eval")
            trainer.tracker.log(metrics, step=0, commit=True)
            logger.info("REPLAY_AUDIT reused verified step-zero baseline from {}", baseline_uri)
        return control

    async def on_step_end_async(self, state, control, **kwargs):
        trainer = kwargs["trainer"]
        metrics = {key: float(value) for key, value in trainer.all_metrics.items() if isinstance(value, (int, float))}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("Nonfinite training metric; stopping the replay run")
        if metrics.get("async/staleness_max", 0) != 0:
            raise ValueError("Replay retry requires synchronized rollout weights")
        if state.global_step <= 2 or state.global_step % 32 == 0:
            current = await self.bias_digest(trainer)
            if current != self.bias_hash:
                raise ValueError("Frozen routing bias changed between steps")
        write_json(
            self.root,
            f"steps/{state.global_step:06d}.json",
            {
                "global_step": state.global_step,
                "metrics": metrics,
                "frozen_bias_sha256": self.bias_hash,
            },
        )
        logger.info("REPLAY_AUDIT completed optimizer step {}", state.global_step)
        return control

    def on_evaluate(self, state, control, **kwargs):
        metrics = kwargs["metrics"]
        write_json(
            self.root,
            f"validation/{state.global_step:06d}.json",
            {
                "global_step": state.global_step,
                "metrics": metrics,
                "expected_tasks": 1003,
                "baseline_origin": kwargs.get("baseline_origin"),
            },
        )
        errors = {key: value for key, value in metrics.items() if key.endswith("/agent_loop_error") and value > 0}
        if errors:
            raise ValueError(f"Validation had execution errors; refusing to train on an invalid baseline: {errors}")
        family_counts = {
            "G01": 110,
            "G02": 120,
            "G03": 108,
            "G04": 3,
            "N01": 72,
            "P01": 32,
            "P02": 23,
            "P03": 118,
            "S01": 50,
            "S02": 9,
            "S03": 119,
            "S04": 111,
            "S05": 12,
            "S09": 116,
        }
        expected = {"all": 1003, **family_counts}
        bad_counts = {
            key: metrics.get(f"eval/{key}/sequences")
            for key, count in expected.items()
            if metrics.get(f"eval/{key}/sequences") != count
        }
        if bad_counts:
            raise ValueError(f"Incomplete validation coverage: {bad_counts}")
        return control
