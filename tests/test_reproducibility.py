"""Reproducibility: identical seed -> identical numbers; different seed -> close."""

from __future__ import annotations

import numpy as np
import pytest

from pescert import get_eval

# proxies that draw randomness (rotations, Hutchinson vectors, MD, random pairs)
SEEDED = ["equivariance", "betti", "trimer", "reversibility", "config_temperature", "equipartition"]


@pytest.mark.parametrize("name", SEEDED)
def test_same_seed_identical(lj_engine, cluster13, trimer3, name):
    # Same seed reuses the exact RNG streams (rotations, Hutchinson vectors, velocities),
    # so results reproduce to ~1e-6.  (Bit-exactness is not asserted: threaded BLAS uses a
    # non-deterministic reduction order, which shows up in MD averages and in the
    # machine-epsilon cancellation of the reversibility round-trip.)
    atoms = trimer3 if name == "trimer" else cluster13
    r1 = get_eval(name).run(lj_engine, atoms, seed=3)
    r2 = get_eval(name).run(lj_engine, atoms, seed=3)
    assert np.isclose(r1.raw_defect, r2.raw_defect, rtol=1e-6, atol=1e-12)
    assert np.isclose(r1.score, r2.score, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("name", ["equivariance", "betti", "trimer"])
def test_different_seed_statistically_close(lj_engine, cluster13, trimer3, name):
    atoms = trimer3 if name == "trimer" else cluster13
    s = [get_eval(name).run(lj_engine, atoms, seed=k).score for k in range(3)]
    # all near 1 for a clean model regardless of seed
    assert np.std(s) < 0.05
    assert min(s) > 0.9


def test_deterministic_eval_seed_independent(lj_engine, cluster13):
    # zero_modes has no randomness; result must not depend on seed
    r0 = get_eval("zero_modes").run(lj_engine, cluster13, seed=0)
    r1 = get_eval("zero_modes").run(lj_engine, cluster13, seed=99)
    assert r0.raw_defect == r1.raw_defect
