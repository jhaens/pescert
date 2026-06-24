"""Broken-fires sensitivity matrix.

Each deliberately-defective model must drop the *relevant* proxy's score sharply while
leaving unrelated proxies clean.  The expected catch/no-catch table (score thresholds):

model        fires (score low)                              stays clean (score high)
-----------  ---------------------------------------------  -------------------------------
CURL         equivariance, trimer(NEW-3iii), betti(OOB-1),  conservativeness*, smoothness,
  (curl)     equipartition, config_temperature              reversibility**
KINKED       smoothness                                     conservativeness, equivariance,
  (rough)                                                   zero_modes
SYMBREAK     equivariance, trimer(NEW-3i), zero_modes        conservativeness, smoothness
BADSTRESS    stress_consistency                             everything else

  * 1D radial conservativeness is structurally blind to a COM-curl (the force is
    transverse to the bond) -- exactly the gap NEW-3iii (the trimer transverse probe)
    was designed to close.  This test asserts that blind spot on purpose.
  ** Velocity-Verlet is time-reversible for any smooth position-dependent force, so the
    OOB-3 round-trip catches roughness/discontinuity, not smooth non-conservativeness
    (the energy-drift diagnostic is reported separately).
"""

from __future__ import annotations

import pytest
from _potentials import (
    BadStress,
    Kinked,
    NonConservative,
    SymmetryBreaking,
    clean_lj,
)

from pescert import from_ase_calculator, get_eval

RMIN = 2.0 ** (1.0 / 6.0)
FIRE = 0.5  # a fired proxy scores below this
CLEAN = 0.85  # an unrelated proxy stays above this


def _eng(calc):
    return from_ase_calculator(calc)


def _score(name, calc, atoms, **cfg):
    return get_eval(name).run(_eng(calc), atoms, seed=0, **cfg).score


# -- CURL: transverse non-conservative ------------------------------------------------
def test_curl_caught_by_jacobian_family(cluster13, trimer3):
    calc = lambda: NonConservative(clean_lj(rc=8.0), strength=0.4)  # noqa: E731
    assert _score("equivariance", calc(), cluster13) < FIRE
    assert _score("trimer", calc(), trimer3) < FIRE  # NEW-3iii transverse probe
    assert _score("betti", calc(), cluster13) < 0.95  # OOB-1, softer signal
    assert _score("equipartition", calc(), cluster13) < FIRE  # NEW-2b
    assert _score("config_temperature", calc(), cluster13) < FIRE  # NEW-2a


def test_curl_blind_spots(cluster13, bulk_ar):
    calc = lambda: NonConservative(clean_lj(rc=8.0), strength=0.4)  # noqa: E731
    # 1D radial conservativeness cannot see a transverse curl (motivation for NEW-3iii)
    assert _score("conservativeness", calc(), cluster13) > CLEAN
    # smoothness and Verlet reversibility are unaffected by a smooth curl
    assert _score("smoothness", calc(), cluster13) > CLEAN
    assert _score("reversibility", calc(), cluster13) > CLEAN
    # energy and stress are untouched -> stress consistency stays clean
    assert _score("stress_consistency", calc(), bulk_ar) > CLEAN


# -- KINKED: conservative but non-smooth ----------------------------------------------
def test_kinked_caught_by_smoothness(cluster13):
    calc = lambda: Kinked(clean_lj(rc=8.0), amplitude=0.6, d0=RMIN + 0.28)  # noqa: E731
    assert _score("smoothness", calc(), cluster13) < FIRE


def test_kinked_does_not_fire_unrelated(cluster13):
    calc = lambda: Kinked(clean_lj(rc=8.0), amplitude=0.6, d0=RMIN + 0.28)  # noqa: E731
    # the kink is conservative and lives outside the conservativeness window / minimum
    assert _score("conservativeness", calc(), cluster13) > CLEAN
    assert _score("equivariance", calc(), cluster13) > CLEAN
    assert _score("zero_modes", calc(), cluster13) > CLEAN


# -- SYMBREAK: orientation-dependent term ---------------------------------------------
def test_symbreak_caught_by_symmetry_family(cluster13, trimer3):
    calc = lambda: SymmetryBreaking(clean_lj(rc=8.0), c=1.0)  # noqa: E731
    assert _score("equivariance", calc(), cluster13) < FIRE
    assert _score("trimer", calc(), trimer3) < FIRE  # NEW-3i reflection
    assert _score("zero_modes", calc(), cluster13) < 0.95  # NEW-1 rotational residual


def test_symbreak_does_not_fire_unrelated(cluster13):
    calc = lambda: SymmetryBreaking(clean_lj(rc=8.0), c=1.0)  # noqa: E731
    # the symmetry-breaking term is conservative
    assert _score("conservativeness", calc(), cluster13) > CLEAN
    assert _score("smoothness", calc(), cluster13) > CLEAN


# -- BADSTRESS: wrong stress head -----------------------------------------------------
def test_badstress_caught_only_by_stress(cluster13, bulk_ar):
    calc = lambda: BadStress(clean_lj(rc=8.0), factor=1.4)  # noqa: E731
    assert _score("stress_consistency", calc(), bulk_ar) < FIRE
    # energy and forces are correct -> nothing else fires
    assert _score("conservativeness", calc(), cluster13) > CLEAN
    assert _score("equivariance", calc(), cluster13) > CLEAN
    assert _score("zero_modes", calc(), cluster13) > CLEAN


@pytest.mark.parametrize("seed", [0, 7])
def test_clean_lj_never_fires(lj_engine, cluster13, trimer3, bulk_ar, seed):
    """Sanity: the clean reference keeps every proxy well above the fire threshold."""
    checks = {
        "conservativeness": cluster13,
        "equivariance": cluster13,
        "smoothness": cluster13,
        "betti": cluster13,
        "zero_modes": cluster13,
        "trimer": trimer3,
        "reversibility": cluster13,
        "stress_consistency": bulk_ar,
    }
    for name, atoms in checks.items():
        r = get_eval(name).run(lj_engine, atoms, seed=seed)
        assert r.score > CLEAN, f"{name} unexpectedly low: {r.score}"
        if r.gate is not None:
            assert r.gate is True
