"""Analytic checks of the diagnostic, independent of GPU inference."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest


def runtime_function(name):
    # The runtime module imports GPU orchestration packages. Exercise its exact
    # pure numerical function without importing those optional dependencies.
    source = Path(__file__).parents[1] / "snowball_pdbthink/parity_runtime.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace[name]


def metric_function():
    return runtime_function("summarize")


def test_identical_distributions_have_zero_mismatch():
    result = metric_function()([[-0.001, -4.0], [-18.0]], [[-0.001, -4.0], [-18.0]])
    assert result["k3"] == 0
    assert result["ess_fraction"] == 1
    assert result["exact_fraction"] == 1


def test_uniform_ratio_can_have_full_ess_and_nonzero_mismatch():
    # ESS measures concentration, not agreement: every ratio is three here.
    result = metric_function()([[-3.0, -4.0]], [[-3.0 + math.log(3), -4.0 + math.log(3)]])
    assert result["k3"] == pytest.approx(2 - math.log(3))
    assert result["ess_fraction"] == pytest.approx(1)
    assert result["share_beyond_2x"] == 1


def test_ratio_tail_reduces_ess():
    # One ratio of nine and nine ratios of one gives ESS = 18²/(10*90).
    reference = [[-5.0] * 10]
    candidate = [[-5.0 + math.log(9)] + [-5.0] * 9]
    result = metric_function()(reference, candidate)
    assert result["ess_fraction"] == pytest.approx(0.36)
    assert result["share_beyond_2x"] == pytest.approx(0.1)
    assert result["k3"] == pytest.approx((8 - math.log(9)) / 10)


def test_rejects_incomplete_and_nonfinite_scores():
    with pytest.raises(ValueError):
        metric_function()([[-1.0], [-2.0]], [[-1.0]])
    with pytest.raises(ValueError, match="Nonfinite"):
        metric_function()([[-1.0]], [[float("nan")]])


def test_decode_control_fills_all_data_parallel_ranks():
    rows = [{"family": family, "prompt_id": family} for family in ("G01", "G03", "P03", "S09")]
    inputs, samples = runtime_function("decode_control_inputs")({"rows": rows * 2}, 16)
    assert samples == 8
    assert len(inputs) == 32
    for row in rows:
        assert inputs.count(row) == 8
