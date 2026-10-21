"""Guard Snowball's termination semantics through the real chat request path."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from skyrl_train.config.behavior_logprobs import configure_behavior_logprob_sampling
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.inference_engines.vllm.utils import apply_openai_sampling
from skyrl_train.trajectory_runners.model_clients import DirectModelClient


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["sampling_params", "eval_sampling_params"])
async def test_original_eos_survives_neutral_generation_config(phase):
    cfg = get_default_config()
    OmegaConf.set_struct(cfg, False)
    overrides = OmegaConf.load(Path(__file__).with_name("launch.yaml")).skyrl
    cfg = OmegaConf.merge(cfg, overrides)
    configure_behavior_logprob_sampling(cfg.generator)
    assert cfg.generator.engine_init_kwargs.generation_config == "vllm"
    sampling = get_vllm_sampling_params(cfg.generator[phase])
    engine = AsyncMock()
    engine.model_name = "snowball"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "answer"
    engine.tokenize.return_value = {"tokens": [11, 12]}
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
                "token_ids": [21, 128009],
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
            }
        ]
    }
    output = await DirectModelClient(engine).generate(
        {
            "prompts": [[{"role": "user", "content": "question"}]],
            "chat_completion_params": [{}],
            "sampling_params": sampling,
        }
    )
    body = engine.chat_completion.await_args.args[0]["json"]
    # The shared serving actor always has training validation enabled. It must
    # use the per-request settings, including greedy evaluation without logs.
    apply_openai_sampling(body, {}, validate_rollout_logprob_sampling=True)
    assert body["stop_token_ids"] == [128001, 128009]
    assert body["temperature"] == (1.0 if phase == "sampling_params" else 0.0)
    assert output["response_ids"] == [[21, 128009]]


def test_training_probability_guard_still_rejects_sampling_processors():
    body = {"temperature": 1.0, "logprobs": True, "min_tokens": 1}
    with pytest.raises(ValueError, match="min_tokens=1"):
        apply_openai_sampling(body, {}, validate_rollout_logprob_sampling=True)


@pytest.mark.parametrize("failure", ["execution", "coverage"])
def test_validation_audit_rejects_invalid_baseline(monkeypatch, failure):
    from types import SimpleNamespace

    import snowball_replay_runtime as audit

    monkeypatch.setenv("SNOWBALL_REPLAY_AUDIT_URI", "unused")
    records = []
    monkeypatch.setattr(audit, "write_json", lambda *args: records.append(args))
    metrics = json.loads((Path(__file__).parent / "results/baseline-step0.json").read_text())["metrics"]
    if failure == "execution":
        metrics["eval/G01/environment/pdbthink/agent_loop_error"] = 1.0
    else:
        metrics["eval/G01/sequences"] -= 1
    with pytest.raises(ValueError, match="execution errors|Incomplete validation"):
        audit.ReplayAuditCallback(None).on_evaluate(SimpleNamespace(global_step=0), None, metrics=metrics)
    assert records  # Preserve evidence before stopping the run.


def test_validation_accepts_both_aggregate_and_family_counts(monkeypatch):
    from types import SimpleNamespace

    import snowball_replay_runtime as audit

    monkeypatch.setenv("SNOWBALL_REPLAY_AUDIT_URI", "unused")
    monkeypatch.setattr(audit, "write_json", lambda *args: None)
    metrics = json.loads((Path(__file__).parent / "results/baseline-step0.json").read_text())["metrics"]
    control = object()
    assert (
        audit.ReplayAuditCallback(None).on_evaluate(SimpleNamespace(global_step=0), control, metrics=metrics) is control
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "checksum", "bias"])
async def test_baseline_reuse_checks_identity_and_logs_provenance(monkeypatch, failure):
    import hashlib
    import io
    from types import SimpleNamespace

    import snowball_replay_runtime as audit

    payload = (Path(__file__).parent / "results/baseline-step0.json").read_bytes()
    monkeypatch.setenv("SNOWBALL_REPLAY_AUDIT_URI", "unused")
    monkeypatch.setenv("SNOWBALL_REPLAY_CODE_SHA256", "code")
    monkeypatch.setenv("SNOWBALL_REPLAY_BASELINE_URI", "baseline")
    monkeypatch.setenv(
        "SNOWBALL_REPLAY_BASELINE_SHA256", "bad" if failure == "checksum" else hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setenv("SNOWBALL_REPLAY_BASELINE_BIAS_SHA256", "bad" if failure == "bias" else "bias")
    monkeypatch.setattr(audit, "verify_weights", AsyncMock())
    monkeypatch.setattr(audit, "StoragePath", lambda uri: SimpleNamespace(open=lambda mode: io.BytesIO(payload)))
    records = []
    monkeypatch.setattr(audit, "write_json", lambda *args: records.append(args))
    cfg = OmegaConf.create({"trainer": {"policy": {"grug_query_bias_update_mode": "frozen"}}})
    callback = audit.ReplayAuditCallback(cfg)
    callback.bias_digest = AsyncMock(return_value="bias")
    trainer = MagicMock()
    if failure:
        with pytest.raises(ValueError, match="checksum|step-zero policy"):
            await callback.on_train_begin_async(SimpleNamespace(global_step=0), None, trainer=trainer)
        trainer.tracker.log.assert_not_called()
    else:
        await callback.on_train_begin_async(SimpleNamespace(global_step=0), None, trainer=trainer)
        metrics = trainer.tracker.log.call_args.args[0]
        assert metrics["eval/baseline_reused"] == 1.0
        assert metrics["eval/all/sequences"] == 1003
        assert records[-1][2]["baseline_origin"] == "baseline"
