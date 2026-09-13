"""Shared fixtures: analytic engines and small substrates scaled to the LJ minimum."""

from __future__ import annotations

import numpy as np
import pytest
from _potentials import AnchoredHarmonic, clean_lj
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic

from pescert import from_ase_calculator
from pescert import p2mm_crystal as build_p2mm
from pescert import trimer as build_trimer
from pescert.evals._geometry import closest_pair

RMIN = 2.0 ** (1.0 / 6.0)  # LJ minimum for sigma = 1

#: Ar covalent diameter.  These fixtures deliberately keep their own geometry rather than
#: calling the production builders unchanged: the production substrates follow each
#: element's experimental reference crystal (3.72 A for Ar, fcc), whereas the sensitivity
#: tests are calibrated against a ``sigma = 1`` analytic potential on a 13-atom
#: icosahedron -- the geometry the "blind spot" and kink cases were constructed on.
#: Pinning it here keeps the probe tests measuring the probes, not the substrate defaults.
D_AR = 2.12


def _rescale_to_rmin(atoms: Atoms) -> Atoms:
    """Uniformly rescale so the closest pair sits at the LJ minimum (fast relaxation)."""
    i, j = closest_pair(atoms)
    d = np.linalg.norm(atoms.get_positions()[i] - atoms.get_positions()[j])
    atoms.set_positions(atoms.get_positions() * (RMIN / d))
    return atoms


def _icosahedron(element: str = "Ar", d_nn: float = D_AR) -> Atoms:
    """13-atom Mackay icosahedron at spacing ``d_nn`` (the tests' own substrate)."""
    from ase.cluster import Icosahedron

    ico = Icosahedron(element, noshells=2, latticeconstant=d_nn * np.sqrt(2.0))
    p = ico.get_positions()
    nn = min(
        np.linalg.norm(p[a] - p[b]) for a in range(len(p)) for b in range(a + 1, len(p))
    )
    ico.set_positions((p - p.mean(0)) * (d_nn / nn))
    ico.pbc = False
    ico.center(vacuum=8.0)
    return ico


@pytest.fixture
def lj_calc():
    return clean_lj(rc=8.0)


@pytest.fixture
def lj_engine():
    return from_ase_calculator(clean_lj(rc=8.0))


@pytest.fixture
def cluster13() -> Atoms:
    """13-atom Mackay icosahedron at the LJ minimum.

    Built here rather than by :func:`pescert.substrates.cluster`, and deliberately so.
    The production builder cuts a fragment of the element's *experimental reference
    crystal*, whose spacing (3.72 A for argon) has nothing to do with the ``sigma = 1``
    analytic potentials these tests use, and whose neighbour shells differ per element.
    These tests certify the *probes* -- their blind spots, what each one does and does not
    fire on -- which needs one controlled geometry commensurate with the test potential,
    not the production substrate rule.  ``tests/test_substrates.py`` covers the builders.
    """
    from ase.cluster import Icosahedron

    d_nn = 2 * 1.06  # covalent diameter of argon
    ico = Icosahedron("Ar", noshells=2, latticeconstant=d_nn * np.sqrt(2.0))
    p = ico.get_positions()
    nn = min(
        np.linalg.norm(p[a] - p[b]) for a in range(len(p)) for b in range(a + 1, len(p))
    )
    ico.set_positions((p - p.mean(0)) * (d_nn / nn))
    ico.pbc = False
    ico.center(vacuum=8.0)
    return _rescale_to_rmin(ico)


@pytest.fixture
def trimer3() -> Atoms:
    return _rescale_to_rmin(build_trimer("Ar"))


@pytest.fixture
def bulk_ar() -> Atoms:
    """Periodic FCC Ar at the LJ pair minimum (for stress / Gamma checks).

    A 2x2x2 supercell has two conventional cells per side, so the nearest-neighbour
    distance is cell[0,0] / (2*sqrt(2)); we set it to RMIN (a physical, moderately
    stressed crystal rather than an overlapping one).
    """
    b = FaceCenteredCubic("Cu", size=(2, 2, 2))
    b.set_chemical_symbols(["Ar"] * len(b))
    b.set_cell(b.cell * (2 * np.sqrt(2.0) * RMIN / b.cell[0, 0]), scale_atoms=True)
    return b


@pytest.fixture
def p2mm_ar() -> Atoms:
    """Four-atom Pmm2 crystal (the parity / representation / cross-Maxwell substrate).

    Rescaled to the covalent spacing the production builder used before it moved to
    reference crystals, which is the geometry these probe tests were calibrated on.
    """
    a = build_p2mm("Ar")
    d = a.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    a.set_cell(np.array(a.cell) * (D_AR / d.min()), scale_atoms=True)
    return a


@pytest.fixture
def harmonic_engine_factory(cluster13):
    """Factory for an exactly-quadratic engine anchored at the cluster geometry.

    Exact equipartition and configurational temperature hold, so it is the clean
    reference for the statistical proxies.
    """

    def make(k: float = 3.0):
        return from_ase_calculator(AnchoredHarmonic(k=k, anchors=cluster13.get_positions()))

    return make
