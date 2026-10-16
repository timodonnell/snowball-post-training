"""Exercise the native CPU batch path before allocating diagnostic GPUs."""

import gzip
import json
from types import SimpleNamespace

import numpy as np
from finestore.rl import mismatch_probe as schema
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.mismatch_probe.collect import ProbeCollector
from skyrl_train.trainer import RayPPOTrainer

from .parity_runtime import collate, decode_control_inputs


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
    auxiliary, samples = decode_control_inputs(panel, 16)
    checked = []
    for name, inputs, group_size, cap in (
        ("fixed-panel", panel["rows"], 2, 2048),
        ("decode-control", auxiliary, samples, 256),
    ):
        probe = ProbeCollector(cfg)
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
        checked.append({"batch": name, "sequences": len(inputs), "padded_rows": batch.batch_size})
    print(json.dumps({"native_collation_preflight": checked, "model_probabilities_computed": False}), flush=True)
