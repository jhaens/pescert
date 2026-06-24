"""Budget: each proxy reports its call count and honors a max_calls cap."""

from __future__ import annotations

import pytest

from pescert import get_eval

# documented upper bounds on model calls for the default settings (small substrates)
BOUNDS = {
    "conservativeness": 40,
    "equivariance": 20,
    "smoothness": 90,
    "betti": 60,
    "zero_modes": 250,
    "trimer": 220,
    "reversibility": 220,
}


@pytest.mark.parametrize("name,bound", list(BOUNDS.items()))
def test_call_count_within_bound(lj_engine, cluster13, trimer3, name, bound):
    atoms = trimer3 if name == "trimer" else cluster13
    lj_engine.reset_counter()
    r = get_eval(name).run(lj_engine, atoms, seed=0)
    assert r.n_model_calls == lj_engine.n_calls
    assert r.n_model_calls <= bound, f"{name} used {r.n_model_calls} > {bound}"


@pytest.mark.parametrize("name", ["smoothness", "conservativeness", "betti"])
def test_max_calls_honored(lj_engine, cluster13, name):
    cap = 24
    r = get_eval(name).run(lj_engine, cluster13, seed=0, max_calls=cap)
    assert r.n_model_calls <= cap


def test_stress_skips_without_pbc(lj_engine, cluster13):
    r = get_eval("stress_consistency").run(lj_engine, cluster13, seed=0)
    assert r.details.get("skipped") is True
    assert r.gate is None
    assert r.score != r.score  # NaN: not applicable
