"""Clean-pass: a smooth, conservative, symmetric model satisfies every identity.

Every "=0" defect must be ~0 and every "=1" ratio ~1 on the clean references; all
correctness gates pass.  This proves the math and finite-differencing are correct.
"""

from __future__ import annotations

from pescert import get_eval


def test_conservativeness_clean(lj_engine, cluster13):
    r = get_eval("conservativeness").run(lj_engine, cluster13, seed=0)
    assert r.target == 0.0
    assert r.score > 0.95
    assert r.gate is True


def test_equivariance_clean(lj_engine, cluster13):
    r = get_eval("equivariance").run(lj_engine, cluster13, seed=0)
    assert r.score > 0.999
    assert r.gate is True


def test_smoothness_clean(lj_engine, cluster13):
    r = get_eval("smoothness").run(lj_engine, cluster13, seed=0)
    assert r.score > 0.9
    assert r.details["force_flips_excess"] == 0
    assert r.details["spurious_minima"] == 0


def test_stress_consistency_clean(lj_engine, bulk_ar):
    r = get_eval("stress_consistency").run(lj_engine, bulk_ar, seed=0)
    assert r.score > 0.99
    assert r.gate is True


def test_betti_clean(lj_engine, cluster13):
    r = get_eval("betti").run(lj_engine, cluster13, seed=0)
    assert r.score > 0.99
    assert r.gate is True


def test_zero_modes_clean(lj_engine, cluster13):
    r = get_eval("zero_modes").run(lj_engine, cluster13, seed=0)
    assert r.details["n_negative_modes"] == 0
    assert r.details["mode_count_error"] == 0
    assert r.details["n_near_zero"] == 6
    assert r.score > 0.99
    assert r.gate is True


def test_trimer_clean(lj_engine, trimer3):
    r = get_eval("trimer").run(lj_engine, trimer3, seed=0)
    assert r.score > 0.99
    # pairwise LJ has exactly zero three-body term at all separations
    assert r.details["sub_scores"]["three_body_vanishing"] > 0.99
    assert r.gate is True


def test_reversibility_clean(lj_engine, cluster13):
    r = get_eval("reversibility").run(lj_engine, cluster13, seed=0)
    assert r.score > 0.99
    assert r.gate is True


def test_config_temperature_clean(harmonic_engine_factory, cluster13):
    # exact-harmonic reference: T_config == T_kin exactly
    engine = harmonic_engine_factory(k=3.0)
    r = get_eval("config_temperature").run(engine, cluster13, seed=1)
    assert r.target == 1.0
    assert abs(r.details["ratio"] - 1.0) < 0.25
    assert r.score > 0.6


def test_equipartition_clean(lj_engine, cluster13):
    r = get_eval("equipartition").run(lj_engine, cluster13, seed=2)
    assert r.target == 1.0
    assert abs(r.details["R_median"] - 1.0) < 0.35
    assert r.score > 0.5
