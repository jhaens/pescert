"""The production substrate builders: geometry that is physical, generic and repeatable.

The probes assume a cluster that is compact, non-linear and a *plausible* structure for
the element -- a substrate deep in a repulsive extrapolation regime turns every probe
into a measurement of the model outside its domain.  These tests pin the properties the
probes rely on, for a bcc metal, close-packed metals, covalent elements, and an element
ASE has no reference crystal for (the covalent fallback).
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk as ase_bulk

from pescert.substrates import RATTLE_STDEV, _nn_distance, _reference_crystal, cluster


def _distances(atoms):
    d = atoms.get_all_distances()
    np.fill_diagonal(d, np.inf)
    return d


@pytest.mark.parametrize(
    "element,expected_coordination",
    [("Na", 8), ("Al", 12), ("Mg", 12), ("C", 4), ("Si", 4), ("S", 4)],
)
def test_cluster_has_the_elements_own_coordination(element, expected_coordination):
    """Cutting the reference crystal keeps distance and coordination consistent.

    This is the point of the builder: a generic close-packed cluster puts twelve
    neighbours at carbon's single-bond length, a geometry no potential is fitted for.
    """
    d = _distances(cluster(element))
    assert int((d[0] < d.min() * 1.15).sum()) == expected_coordination


@pytest.mark.parametrize("element", ["Na", "Al", "C"])
def test_cluster_spacing_is_the_reference_crystals(element):
    crystal = ase_bulk(element).repeat(3)
    d_crystal = crystal.get_all_distances(mic=True)
    np.fill_diagonal(d_crystal, np.inf)
    assert _nn_distance(element) == pytest.approx(float(d_crystal.min()), rel=1e-9)
    # the symmetry-breaking displacement moves individual contacts by a few times its
    # own width, so the bound is stated in terms of it rather than as a bare percentage
    assert _distances(cluster(element)).min() == pytest.approx(
        _nn_distance(element), abs=4 * RATTLE_STDEV
    )


def test_cluster_is_deterministic_and_symmetry_broken():
    """Every model must be handed the identical geometry, and it must not be a saddle."""
    a, b = cluster("Al"), cluster("Al")
    assert np.allclose(a.get_positions(), b.get_positions())
    assert abs(_distances(a).min() - _nn_distance("Al")) < 6 * RATTLE_STDEV


@pytest.mark.parametrize("element", ["Li", "C", "S"])
def test_cluster_is_compact_and_non_linear(element):
    """Six rigid-body zero modes need three non-degenerate principal axes."""
    atoms = cluster(element)
    assert len(atoms) == 13
    pos = atoms.get_positions() - atoms.get_positions().mean(0)
    assert np.linalg.eigvalsh(pos.T @ pos).min() > 0.05 * np.linalg.eigvalsh(pos.T @ pos).max()
    d = _distances(atoms)
    assert d.min(axis=1).max() < 1.6 * d.min()  # no atom left behind by the cut


def test_fallback_for_elements_without_a_reference_crystal():
    """Sulfur has no ASE reference crystal; it still gets a consistent structure."""
    assert _reference_crystal("S") is None
    d = _distances(cluster("S"))
    assert d.min() == pytest.approx(_nn_distance("S"), abs=4 * RATTLE_STDEV)
    assert int((d[0] < d.min() * 1.15).sum()) == 4  # diamond, not close packed
