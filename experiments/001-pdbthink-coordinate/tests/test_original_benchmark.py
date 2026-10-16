"""Reject silent context clamps, tokenization drift, and tool-enabled responses."""

import copy

import pytest
from snowball_pdbthink.run_original_benchmark import validate_response


@pytest.fixture
def response_and_task():
    return (
        {
            "choices": [{"finish_reason": "stop", "message": {"content": "FINAL: 3.42"}}],
            "usage": {"prompt_tokens": 30000, "completion_tokens": 500},
        },
        {"render_id": "frozen-example", "prompt_tokens": 30000, "body": {"max_tokens": 2768}},
    )


def test_complete_and_full_native_truncation_are_retained(response_and_task):
    response, task = response_and_task
    validate_response(response, task)
    response["choices"][0]["finish_reason"] = "length"
    response["usage"]["completion_tokens"] = 2768
    validate_response(response, task)  # The verifier will score this as truncated, not drop it.


@pytest.mark.parametrize("mutation", ["prompt", "clamp", "tool", "multiple"])
def test_protocol_violations_abort_the_measurement(response_and_task, mutation):
    response, task = copy.deepcopy(response_and_task)
    if mutation == "prompt":
        response["usage"]["prompt_tokens"] -= 1
    elif mutation == "clamp":
        response["choices"][0]["finish_reason"] = "length"  # Stops well below the available 2,768 tokens.
    elif mutation == "tool":
        response["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "python"}}]
    else:
        response["choices"] *= 2
    with pytest.raises(ValueError):
        validate_response(response, task)
