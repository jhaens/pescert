"""Shared fixtures: analytic engines and small substrates scaled to the LJ minimum."""

from __future__ import annotations

import numpy as np
import pytest
from _potentials import AnchoredHarmonic, clean_lj
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic

from pescert import cluster as build_cluster
from pescert import from_ase_calculator
from pescert import trimer as build_trimer
from pescert.evals._geometry import closest_pair

RMIN = 2.0 ** (1.0 / 6.0)  # LJ minimum for sigma = 1


def _rescale_to_rmin(atoms: Atoms) -> Atoms:
    """Uniformly rescale so the closest pair sits at the LJ minimum (fast relaxation)."""
    i, j = closest_pair(atoms)
    d = np.linalg.norm(atoms.get_positions()[i] - atoms.get_positions()[j])
    atoms.set_positions(atoms.get_positions() * (RMIN / d))
    return atoms


@pytest.fixture
def lj_calc():
    return clean_lj(rc=8.0)


@pytest.fixture
def lj_engine():
    return from_ase_calculator(clean_lj(rc=8.0))


@pytest.fixture
def cluster13() -> Atoms:
    """13-atom icosahedron scaled to the LJ minimum (relaxes to a genuine minimum)."""
    return _rescale_to_rmin(build_cluster("Ar", 13))


@pytest.fixture
def trimer3() -> Atoms:
    return _rescale_to_rmin(build_trimer("Ar"))


@pytest.fixture
def bulk_ar() -> Atoms:
    """Periodic FCC Ar at the LJ lattice constant (for stress / Gamma checks)."""
    b = FaceCenteredCubic("Cu", size=(2, 2, 2))
    b.set_chemical_symbols(["Ar"] * len(b))
    b.set_cell(b.cell * (RMIN * np.sqrt(2.0) / b.cell[0, 0]), scale_atoms=True)
    return b


@pytest.fixture
def harmonic_engine_factory(cluster13):
    """Factory for an exactly-quadratic engine anchored at the cluster geometry.

    Exact equipartition and configurational temperature hold, so it is the clean
    reference for the statistical proxies.
    """

    def make(k: float = 3.0):
        return from_ase_calculator(AnchoredHarmonic(k=k, anchors=cluster13.get_positions()))

    return make
