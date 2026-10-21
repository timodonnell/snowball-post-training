"""Guard Snowball's termination semantics through the real chat request path."""

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
    apply_openai_sampling(body, sampling, validate_rollout_logprob_sampling=phase == "sampling_params")
    assert body["stop_token_ids"] == [128001, 128009]
    assert body["temperature"] == (1.0 if phase == "sampling_params" else 0.0)
    assert output["response_ids"] == [[21, 128009]]
