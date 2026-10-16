"""Install tightly scoped diagnostic instrumentation in a job-local source tree."""

import hashlib
import json
import shutil
import sys
from pathlib import Path


def patch_role_plan(source):
    anchor = "    tensor_pipeline_size = tensor_parallel_size * pipeline_parallel_size\n"
    if source.count(anchor) != 1:
        raise ValueError("Diagnostic role-planner anchor changed")
    return source.replace(
        anchor,
        "    if (replicas, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, "
        "gpus_per_node, multiprocessing_backend) == (2, 1, 1, 8, 4, False):\n"
        "        return 4  # Verified cross-node EP placement supported by the pinned runtime.\n" + anchor,
    )


def install(root, bundle, launcher_root=None):
    expected = json.loads((bundle / "source-hashes.json").read_text())
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Diagnostic source does not match pinned implementation: " + name)
    role_plan = "cloud/iris/role_plan.py"

    def change(name, old, new, count=1):
        path = root / name
        source = path.read_text()
        if source.count(old) != count:
            raise ValueError(f"Diagnostic instrumentation anchor changed: {name}")
        path.write_text(source.replace(old, new))

    prefix = "skyrl-train/skyrl_train/"
    (root / role_plan).write_text(patch_role_plan((root / role_plan).read_text()))
    # The launcher applies this patch before computing the bundle's file
    # identities. Never mutate that bundle after its identity is recorded.
    if launcher_root is not None and (root / role_plan).read_bytes() != (launcher_root / role_plan).read_bytes():
        raise ValueError("Staged launcher role planner differs from the declared diagnostic patch")
    shutil.copyfile(bundle / "parity_runtime.py", root / "skyrl-train/snowball_parity_runtime.py")
    change(
        prefix + "callbacks/builtin.py",
        "from skyrl_train.mismatch_probe.callback import MismatchProbeCallback",
        "from snowball_parity_runtime import ParityCallback as MismatchProbeCallback",
    )
    worker = prefix + "workers/megatron/megatron_worker.py"
    change(
        worker,
        "        self._expert_block_sender = (\n            ExpertBlockSender(self, mpu)\n"
        "            if self.cfg.generator.weight_sync_transport == WeightSyncTransport.EXPERT_BLOCK\n"
        "            else None\n        )",
        "        self._expert_block_sender = ExpertBlockSender(self, mpu)",
    )
    change(
        worker,
        '    def _ppo_train_impl(self, train_data, timing: PhaseBreakdown) -> "TrainingOutputBatch":\n',
        '    def _ppo_train_impl(self, train_data, timing: PhaseBreakdown) -> "TrainingOutputBatch":\n'
        '        raise RuntimeError("Optimizer updates are forbidden in the frozen parity diagnostic")\n',
    )
    # Preserve captured arrays through the existing batched token-scoring path.
    engine = prefix + "inference_engines/vllm/vllm_engine.py"
    change(
        engine,
        "        if response_top_k is not None and response_top_k > 0:\n"
        '            result["student_topk_indices"] = student_topk_indices',
        '        result["routed_experts"] = [item.outputs[0].routed_experts for item in outputs]\n'
        "        if response_top_k is not None and response_top_k > 0:\n"
        '            result["student_topk_indices"] = student_topk_indices',
    )
    client = prefix + "inference_engines/inference_engine_client.py"
    change(
        client,
        '        n = len(prompt_token_ids)\n        responses: list[str] = [""] * n',
        "        n = len(prompt_token_ids)\n        routed_experts = [None] * n\n"
        '        responses: list[str] = [""] * n',
    )
    change(
        client,
        '                responses[original_idx] = result["responses"][local_idx]',
        '                routed_experts[original_idx] = result["routed_experts"][local_idx]\n'
        '                responses[original_idx] = result["responses"][local_idx]',
    )
    change(
        client,
        "        if add_student_topk:\n",
        '        output["routed_experts"] = routed_experts\n        if add_student_topk:\n',
    )
    # Verification uses scratch buffers only. Support the documented CUTLASS
    # [up;gate] storage by making a canonical read-only snapshot of those stacks.
    receiver = prefix + "weight_sync/expert_block/receiver.py"
    change(
        receiver,
        "            if backend != SUPPORTED_MOE_BACKEND:",
        '            if backend not in {SUPPORTED_MOE_BACKEND, "FLASHINFER_CUTLASS"}:',
    )
    change(
        receiver,
        "        self.parameters = installable_parameters(self.model)",
        "        self.parameters = installable_parameters(self.model)\n"
        "        for name, module in self.model.named_modules():\n"
        "            if not name.endswith(ROUTED_EXPERTS):\n"
        "                continue\n"
        '            if module.quant_method.unquantized_backend.name == "FLASHINFER_CUTLASS":\n'
        '                key = name + ".w13_weight"\n'
        "                value = self.parameters[key]\n"
        "                half = value.shape[1] // 2\n"
        "                self.parameters[key] = torch.cat((value[:, half:], value[:, :half]), dim=1)",
    )
    record = {
        "source_sha256": expected,
        "installed_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected},
    }
    if launcher_root is not None:
        record["staged_role_plan_sha256"] = hashlib.sha256((launcher_root / role_plan).read_bytes()).hexdigest()
    (root / "snowball-parity-instrumentation.json").write_text(json.dumps(record, indent=2))
    print("Installed read-only Snowball parity instrumentation", flush=True)


if __name__ == "__main__":
    install(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
