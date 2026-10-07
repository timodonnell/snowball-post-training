"""Load the release verifier and report its exact, binary outcome."""

import importlib.util
import json
import sys
from pathlib import Path


def load_scorer(directory):
    package = Path(directory) / "coordinate_scoring"
    spec = importlib.util.spec_from_file_location(
        "pdbthink_release_scoring", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.score_response


def tool_events(value):
    if isinstance(value, dict):
        return any(
            (key in ("tool_calls", "function_call", "annotations") and bool(item))
            or (key == "type" and item in ("tool_use", "tool_result", "function_call", "web_search_call"))
            or tool_events(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(tool_events(item) for item in value)
    return False


def score(scorer, text, ground_truth, *, truncated=False, refusal=False, tool_violation=False):
    gold = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    outcome = scorer(
        text if len(text.encode()) <= 8_000_000 else "",
        gold["answer_schema"],
        gold["gold_answer"],
        parameters=gold["parameters"],
        truncated=truncated,
        provider_refusal=refusal,
    )
    return {
        "reward": float(outcome["score"].get("correct", False) and not tool_violation),
        "format_error": outcome["format_error"],
        "truncated": bool(truncated),
        "refusal": outcome["refusal"],
        "tool_violation": bool(tool_violation),
        "outcome": outcome,
    }
