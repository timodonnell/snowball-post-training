"""Checks and durable metrics for the route-replay training retry."""

import hashlib
import math
import os

import ray
import torch
from loguru import logger
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
            },
        )
        errors = {key: value for key, value in metrics.items() if key.endswith("/agent_loop_error") and value > 0}
        if errors:
            raise ValueError(f"Validation had execution errors; refusing to train on an invalid baseline: {errors}")
        coverage = sum(
            value for key, value in metrics.items() if key.startswith("eval/") and key.endswith("/sequences")
        )
        if coverage != 1003:
            raise ValueError(f"Incomplete validation coverage: {coverage}/1003")
        return control
