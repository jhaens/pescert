"""The equilibrium proxies share one Langevin trajectory: same numbers, fewer calls.

``config_temperature`` runs the relaxation, the equilibration and the MD; ``equipartition``
and ``virial`` subsample the stored frames.  Sharing must change *cost only*.

Exactness is asserted against the anchored-harmonic engine rather than the LJ one on
purpose: ASE's ``LennardJones`` caches a neighbour list, so relaxing the same geometry
twice can land ~1e-11 apart depending on what the calculator evaluated before, and a
1500-step Langevin trajectory amplifies that chaotically.  The analytic harmonic
potential is history-independent, so ``shared`` and ``per-proxy`` agree bit for bit --
which is the property this refactor is actually claiming.
"""

from __future__ import annotations

import pytest

from pescert import Suite, get_eval
from pescert.evals._trajectory import Thermostat, Trajectory, TrajectoryCache, n_samples_for
from pescert.suite import _order_for_trajectory_reuse

EQUILIBRIUM = ["config_temperature", "equipartition", "virial"]


def _by_name(report):
    return {r.name: r for r in report.results}


def test_sharing_does_not_change_the_numbers(harmonic_engine_factory, cluster13):
    suite = Suite.from_names(EQUILIBRIUM)
    alone = _by_name(
        suite.run(harmonic_engine_factory(), cluster13, seed=0, share_trajectory=False)
    )
    shared = _by_name(
        suite.run(harmonic_engine_factory(), cluster13, seed=0, share_trajectory=True)
    )
    for name in EQUILIBRIUM:
        assert alone[name].raw_defect == shared[name].raw_defect, name
        assert alone[name].score == shared[name].score, name


def test_sharing_saves_the_model_calls(lj_engine, cluster13):
    """Only config_temperature pays for the trajectory."""
    suite = Suite.from_names(EQUILIBRIUM)
    alone = _by_name(suite.run(lj_engine, cluster13, seed=0, share_trajectory=False))
    shared = _by_name(suite.run(lj_engine, cluster13, seed=0, share_trajectory=True))

    assert shared["virial"].n_model_calls == 0  # frames only, nothing of its own
    assert shared["equipartition"].n_model_calls <= 6 * len(cluster13) + 1  # its Hessian
    # The producer pays the same either way, to within the one call the suite may spend
    # settling the model's precision -- that call warms the calculator's result cache, so
    # whichever run makes it saves a call on the next relaxation.
    assert abs(
        shared["config_temperature"].n_model_calls
        - alone["config_temperature"].n_model_calls
    ) <= 1
    total = sum(r.n_model_calls for r in shared.values())
    assert total < 0.6 * sum(r.n_model_calls for r in alone.values())


def test_metadata_records_the_owner_and_its_consumers(lj_engine, cluster13):
    report = Suite.from_names(EQUILIBRIUM).run(lj_engine, cluster13, seed=0)
    stats = report.metadata["shared_trajectory"]
    assert stats["n_trajectories"] == 1
    assert stats["owners"] == ["config_temperature"]
    assert sorted(stats["reused_by"]) == ["equipartition", "virial"]
    for name, result in _by_name(report).items():
        assert result.details["trajectory"]["owner"] == "config_temperature", name


def test_producer_runs_before_its_consumers(lj_engine, cluster13):
    # listed last, config_temperature is still the one that runs (and pays) first
    report = Suite.from_names(["virial", "equipartition", "config_temperature"]).run(
        lj_engine, cluster13, seed=0
    )
    assert [r.name for r in report.results][0] == "config_temperature"
    assert _by_name(report)["virial"].n_model_calls == 0


@pytest.mark.parametrize(
    "names,expected",
    [
        (["config_temperature", "equipartition", "virial"], None),  # already in order
        (["virial", "config_temperature"], ["config_temperature", "virial"]),
        (
            ["smoothness", "virial", "betti", "config_temperature"],
            ["smoothness", "config_temperature", "virial", "betti"],
        ),
        (["equipartition", "virial"], None),  # no producer selected: leave it alone
        (["betti", "config_temperature"], None),  # no consumer selected
    ],
)
def test_ordering_only_moves_the_producer(names, expected):
    assert _order_for_trajectory_reuse(names) == (expected or names)


def test_shorter_run_truncates_to_the_same_prefix(harmonic_engine_factory, cluster13):
    """A consumer wanting fewer steps reads a prefix -- not a fresh trajectory."""
    cfg = {"virial": {"n_steps": 500}}
    shared = _by_name(
        Suite.from_names(EQUILIBRIUM).run(harmonic_engine_factory(), cluster13, seed=0, configs=cfg)
    )["virial"]
    alone = get_eval("virial").run(harmonic_engine_factory(), cluster13, seed=0, n_steps=500)
    assert shared.n_model_calls == 0
    assert shared.raw_defect == alone.raw_defect


def test_longer_run_continues_instead_of_restarting(harmonic_engine_factory, cluster13):
    """A consumer wanting more steps extends the same dynamics rather than redoing it."""
    cfg = {"virial": {"n_steps": 1400}}
    shared = _by_name(
        Suite.from_names(EQUILIBRIUM).run(harmonic_engine_factory(), cluster13, seed=0, configs=cfg)
    )["virial"]
    alone = get_eval("virial").run(harmonic_engine_factory(), cluster13, seed=0, n_steps=1400)
    # only the 300 extra steps are paid for (+1 for the force evaluation on resuming)
    assert shared.n_model_calls <= 301
    assert shared.raw_defect == alone.raw_defect


def test_mismatched_settings_get_their_own_trajectory(harmonic_engine_factory, cluster13):
    """A per-eval override that changes the thermostat must not reuse the wrong frames."""
    cfg = {"virial": {"temperature_K": 60.0}}
    report = Suite.from_names(EQUILIBRIUM).run(
        harmonic_engine_factory(), cluster13, seed=0, configs=cfg
    )
    assert report.metadata["shared_trajectory"]["n_trajectories"] == 2
    shared = _by_name(report)["virial"]
    alone = get_eval("virial").run(harmonic_engine_factory(), cluster13, seed=0, temperature_K=60.0)
    assert shared.details["trajectory"]["reused"] is False
    assert shared.raw_defect == alone.raw_defect


def test_frames_match_ase_observer_intervals(lj_engine, cluster13):
    """Frame selection reproduces what ``dyn.attach(..., interval=k)`` would have fired."""
    traj = Trajectory(lj_engine, cluster13, Thermostat(seed=0, warmup_steps=20), owner="test")
    traj.extend(40)
    assert traj.n_steps == 40
    for every in (1, 3, 4, 10):
        frames = traj.frames(sample_every=every)
        assert len(frames) == n_samples_for(20, 40, every)
        assert all(s % every == 0 for s in frames.steps)
        assert frames.steps.min() > 20 and frames.steps.max() <= 60
        assert frames.positions.shape == (len(frames), len(cluster13), 3)
    # truncation keeps a strict prefix of the sampled steps
    assert list(traj.frames(sample_every=4, n_steps=20).steps) == [24, 28, 32, 36, 40]


def test_cache_keys_on_the_substrate(lj_engine, cluster13, trimer3):
    """Different substrates never share a trajectory."""
    cache = TrajectoryCache()
    thermostat = Thermostat(seed=0, warmup_steps=5)
    _, reused_a = cache.get(lj_engine, cluster13, thermostat, owner="a")
    _, reused_b = cache.get(lj_engine, trimer3, thermostat, owner="b")
    _, reused_c = cache.get(lj_engine, cluster13, thermostat, owner="c")
    assert (reused_a, reused_b, reused_c) == (False, False, True)
    assert cache.stats()["n_trajectories"] == 2


def test_standalone_evals_are_unaffected(lj_engine, cluster13):
    """Without a cache each proxy still runs its own trajectory (the documented default)."""
    result = get_eval("virial").run(lj_engine, cluster13, seed=0)
    assert result.details["trajectory"]["reused"] is False
    assert result.n_model_calls > 1000
