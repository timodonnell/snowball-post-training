"""Protect signed torsion conventions used to audit model reasoning."""

import math

import numpy as np
import pytest
from analyze_reasoning_audit import atoms_for, torsion, torsion_bin


@pytest.mark.parametrize("angle", [-179, -130, -60, 30, 60, 130, 179])
def test_torsion_known_geometry_and_rigid_transform(angle):
    theta = math.radians(angle)
    # The middle bond is deliberately not unit length.
    points = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2], [math.cos(theta), math.sin(theta), 2]])
    assert torsion(points.tolist())["angle"] == pytest.approx(angle)
    # Proper rotation, translation and uniform scale must preserve a dihedral.
    rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    moved = 3.5 * (points @ rotation) + [17, -5, 2]
    assert torsion(moved.tolist())["angle"] == pytest.approx(angle)
    assert torsion(points.tolist())["bin"] == torsion_bin(angle)


def test_missing_norm_can_change_bin_outside_observed_validation_support():
    theta = math.radians(115)
    points = [[1, 0, 0], [0, 0, 0], [0, 0, 1.53], [math.cos(theta), math.sin(theta), 1.53]]
    result = torsion(points)
    assert result["bin"] == "g+"
    assert result["missing_bond_norm_bin"] == "t"


def test_fixed_column_parser_keeps_signed_coordinates_and_atom_names():
    line = "ATOM      7  NZ  LYS A  25      -8.282  -2.818  11.482  1.00  0.00           N"
    assert atoms_for([{"content": line}]) == {("A", 25, "NZ"): [-8.282, -2.818, 11.482]}
