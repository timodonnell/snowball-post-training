"""Checks for the family-composition control used in the regression audit."""

import pytest
from snowball_pdbthink.regression_audit import standardized_training_windows


def test_mixture_control_removes_composition_only_change():
    # Identical within-family success, radically different easy/hard proportions.
    steps = [
        {"step": 1, "responses": 100, "family_counts": {"easy": 80, "hard": 20},
         "family_successes": {"easy": 80, "hard": 0}, "mean_input_tokens": 10},
        {"step": 2, "responses": 100, "family_counts": {"easy": 20, "hard": 80},
         "family_successes": {"easy": 20, "hard": 0}, "mean_input_tokens": 20},
    ]
    result = standardized_training_windows(steps, [(1, 1), (2, 2)], reference_end=1)
    a, b = result["windows"]
    assert a["raw_accuracy"] == 0.8 and b["raw_accuracy"] == 0.2
    assert a["fixed_family_mix_accuracy"] == b["fixed_family_mix_accuracy"] == 0.8
    assert b["family_mix_total_variation"] == pytest.approx(0.6)


def test_absent_family_is_excluded_and_weights_renormalized():
    steps = [
        {"step": 1, "responses": 10, "family_counts": {"common": 8, "rare": 2},
         "family_successes": {"common": 4, "rare": 2}, "mean_input_tokens": 10},
        {"step": 2, "responses": 10, "family_counts": {"common": 10},
         "family_successes": {"common": 5}, "mean_input_tokens": 10},
    ]
    result = standardized_training_windows(steps, [(1, 1), (2, 2)], reference_end=1)
    assert result["weights"] == {"common": 1}
    assert result["reference_coverage"] == 0.8
    assert result["excluded_families"] == ["rare"]
    assert [r["fixed_family_mix_accuracy"] for r in result["windows"]] == [0.5, 0.5]
