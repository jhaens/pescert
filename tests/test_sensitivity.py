"""Broken-fires sensitivity matrix.

Each deliberately-defective model must drop the *relevant* proxy's score sharply while
leaving unrelated proxies clean.  The expected catch/no-catch table (score thresholds):

model        fires (score low)                              stays clean (score high)
-----------  ---------------------------------------------  -------------------------------
CURL         equivariance, trimer (transverse), betti,      conservativeness*, smoothness,
  (curl)     equipartition, config_temperature              reversibility**
KINKED       smoothness                                     conservativeness, equivariance,
  (rough)                                                   zero_modes
SYMBREAK     equivariance, trimer (3-body), zero_modes       conservativeness, smoothness
BADSTRESS    stress_consistency                             everything else

  * 1D radial conservativeness is structurally blind to a COM-curl (the force is
    transverse to the bond) -- exactly the gap the trimer's transverse channel
    was designed to close.  This test asserts that blind spot on purpose.
  ** Velocity-Verlet is time-reversible for any smooth position-dependent force, so the
    reversibility round-trip catches roughness/discontinuity, not smooth non-conservativeness
    (the energy-drift diagnostic is reported separately).
"""

from __future__ import annotations

import pytest
from _potentials import (
    BadStress,
    Kinked,
    NonConservative,
    ParityBreakingStress,
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
    calc = lambda: NonConservative(clean_lj(rc=8.0), strength=1.0)  # noqa: E731
    assert _score("equivariance", calc(), cluster13) < FIRE
    assert _score("trimer", calc(), trimer3) < FIRE  # transverse channel
    assert _score("betti", calc(), cluster13) < 0.95  # softer signal
    assert _score("equipartition", calc(), cluster13) < FIRE
    assert _score("config_temperature", calc(), cluster13) < FIRE


def test_curl_blind_spots(cluster13, bulk_ar):
    calc = lambda: NonConservative(clean_lj(rc=8.0), strength=0.4)  # noqa: E731
    # 1D radial conservativeness cannot see a transverse curl: the motivation for the
    # trimer's transverse channel
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
    assert _score("trimer", calc(), trimer3) < FIRE  # 3-body reflection
    assert _score("zero_modes", calc(), cluster13) < 0.95  # rotational residual


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


# -- PARITYLEAK: improper-symmetry stress residual ------------------------------------
def test_parity_leak_caught_by_parity(p2mm_ar):
    calc = lambda: ParityBreakingStress(clean_lj(rc=8.0), amplitude=0.02)  # noqa: E731
    r = get_eval("parity").run(_eng(calc()), p2mm_ar, seed=0)
    assert r.score < FIRE
    # the leak is localised to the improper channel sigma_xy, not the proper controls
    assert r.details["improper_channel_xy"] > 10 * r.details["proper_control_shear"]


def test_parity_leak_does_not_fire_unrelated(cluster13, bulk_ar):
    calc = lambda: ParityBreakingStress(clean_lj(rc=8.0), amplitude=0.02)  # noqa: E731
    # only the improper stress channel is touched; energy/forces are untouched
    assert _score("conservativeness", calc(), cluster13) > CLEAN
    assert _score("equivariance", calc(), cluster13) > CLEAN


# -- NONLOCAL / CURL: representation, cross-Maxwell, virial ----------------------------
def test_representation_caught_by_global_com_force(p2mm_ar):
    # NonConservative uses a global centre-of-mass force -> non-extensive / non-local
    calc = lambda: NonConservative(clean_lj(rc=8.0), strength=0.4)  # noqa: E731
    assert _score("representation", calc(), p2mm_ar) < FIRE


def test_cross_maxwell_clean_gates_and_broken_degrades(lj_engine, p2mm_ar):
    clean = get_eval("cross_maxwell").run(lj_engine, p2mm_ar, seed=0, n_pairs=12)
    assert clean.score > 0.9 and clean.gate is True  # no false positive
    noncons = get_eval("cross_maxwell").run(
        _eng(NonConservative(clean_lj(rc=8.0), strength=0.6)), p2mm_ar, seed=0, n_pairs=12
    )
    badstress = get_eval("cross_maxwell").run(
        _eng(BadStress(clean_lj(rc=8.0), factor=2.0)), p2mm_ar, seed=0, n_pairs=12
    )
    # an inconsistent stress head / non-integrable force breaks strain-position reciprocity
    assert noncons.score < clean.score
    assert badstress.score < clean.score


def test_virial_caught_by_nonconservative(cluster13):
    calc = lambda: NonConservative(clean_lj(rc=8.0), strength=0.4)  # noqa: E731
    r = get_eval("virial").run(_eng(calc()), cluster13, seed=0, n_steps=1200)
    assert r.score < FIRE  # no Boltzmann distribution -> virial ratio drifts off 1


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
