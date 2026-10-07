import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from snowball_pdbthink.adapter import build_adapter
from snowball_pdbthink.evaluate import request_for, summarize
from snowball_pdbthink.scoring import load_scorer, score, tool_events


@pytest.fixture
def scorer(tmp_path):
    with zipfile.ZipFile(Path(__file__).parent / "fixtures/verifier-v1.3.0.zip") as archive:
        archive.extractall(tmp_path)
    return load_scorer(tmp_path)


def test_inclusive_tolerance_and_unterminated_answer(scorer):
    gold = {"answer_schema": "distance", "gold_answer": {"value": 1.0}, "parameters": {"tolerance": 0.02}}
    assert score(scorer, "FINAL: 1.02", gold)["reward"] == 1.0
    assert score(scorer, "FINAL: 1.021", gold)["reward"] == 0.0
    assert score(scorer, "FINAL: 1.0", gold, truncated=True)["reward"] == 0.0
    assert score(scorer, "FINAL: 1.0", gold, refusal=True)["reward"] == 0.0
    assert score(scorer, "FINAL: 1.0", gold, tool_violation=True)["reward"] == 0.0


def test_partial_sets_never_become_rl_reward(scorer):
    gold = {"answer_schema": "residue_set", "gold_answer": {"value": ["A:A1", "A:G2"]}, "parameters": {}}
    result = score(scorer, "FINAL: A:A1", gold)
    assert 0 < result["outcome"]["score"]["score"] < 1
    assert result["reward"] == 0.0
    assert score(scorer, "FINAL: A:A1, A:G2", gold)["reward"] == 1.0


def test_native_verify_entrypoint_agrees_with_adapter(scorer, tmp_path):
    import importlib.util
    import sys

    sys.path.insert(0, str(tmp_path))
    spec = importlib.util.spec_from_file_location("pinned_verify", tmp_path / "verify.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    gold = {"answer_schema": "integer", "gold_answer": {"value": 7}, "parameters": {}}
    (tmp_path / "gold.json").write_text(json.dumps(gold))
    answer = tmp_path / "answer.txt"
    answer.write_text("FINAL: 7")
    for truncated, tool_violation in [(False, False), (True, False), (False, True)]:
        (tmp_path / "answer_status.json").write_text(
            json.dumps({"truncated": truncated, "tool_events": ["call"] if tool_violation else []})
        )
        native = verifier.verify(tmp_path, answer, tmp_path / "logs")
        adapted = score(scorer, answer.read_text(), gold, truncated=truncated, tool_violation=tool_violation)
        assert native["reward"] == adapted["reward"]
        assert float((tmp_path / "logs/reward.txt").read_text()) == adapted["reward"]
    sys.path.remove(str(tmp_path))


def test_request_uses_full_budget_and_no_gold():
    row = {
        "path": "task-a",
        "input_tokens": 3000,
        "prompt": [{"role": "system", "content": "system"}, {"role": "user", "content": "coordinates"}],
        "reward_model": {"ground_truth": "SECRET GOLD"},
    }
    request = request_for(row, "snowball")
    assert request["max_tokens"] == 29768
    assert request["tools"] == [] and request["tool_choice"] == "none"
    assert "SECRET GOLD" not in json.dumps(request)
    assert request["messages"] == row["prompt"]
    assert tool_events({"choices": [{"message": {"tool_calls": [{"name": "python"}]}}]})
    assert not tool_events({"choices": [{"message": {"content": "I considered a tool"}}]})


def test_macro_does_not_overweight_larger_families():
    def row(family, reward):
        return {
            "family": family,
            "reward": reward,
            "format_error": False,
            "truncated": False,
            "tool_violation": False,
            "output_tokens": 10,
            "finish_reason": "stop",
        }

    report = summarize([row("G01", 1)] * 9 + [row("G04", 0)], expected=12)
    assert report["task_weighted_accuracy"] == 0.9
    assert report["family_macro_accuracy"] == 0.5
    assert report["coverage"] == 10 / 12
    assert report["families"]["G04"]["n"] == 1


def test_overlay_carries_the_verified_native_template(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    template = (fixtures / "chat_template.jinja").read_bytes()
    (tmp_path / "native_chat_template.jinja").write_bytes(template)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"model_metadata_sha256": {"chat_template.jinja": hashlib.sha256(template).hexdigest()}})
    )
    with zipfile.ZipFile(fixtures / "verifier-v1.3.0.zip") as archive:
        archive.extractall(tmp_path / "native_verifier")
    identity = build_adapter(tmp_path)
    assert build_adapter(tmp_path) == identity
    with zipfile.ZipFile(tmp_path / "adapter.zip") as archive:
        assert archive.read("native_chat_template.jinja") == template
        assert "native_verifier/coordinate_scoring/scorers.py" in archive.namelist()
    (tmp_path / "native_chat_template.jinja").write_bytes(template + b"changed")
    with pytest.raises(ValueError, match="Native template differs"):
        build_adapter(tmp_path)
