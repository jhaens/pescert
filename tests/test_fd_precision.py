"""Finite-difference steps must respect the model's working float precision.

A central difference balances truncation, ``O(h**accuracy)``, against round-off,
``O(noise / h**order)``.  A float32 model's noise is ~1e9 times a float64 model's, so a
step tuned for float64 measures round-off rather than the potential energy surface.  The
contract asserted here: float64 keeps the well-tested 1e-3 defaults exactly, float32 gets
a larger step, and every probe records the step it actually used.
"""

from __future__ import annotations

import numpy as np
import pytest
from _potentials import clean_lj
from ase.calculators.calculator import Calculator, all_changes

from pescert import Suite, from_ase_calculator, get_eval
from pescert.engine import FD_MIN_STEP, ModelEngine


class Float32LJ(Calculator):
    """Clean LJ that returns float32 arrays, as a float32 network would."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self._inner = clean_lj(rc=8.0)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        work = atoms.copy()
        work.calc = self._inner
        self.results = {
            "energy": float(work.get_potential_energy()),
            "free_energy": float(work.get_potential_energy()),
            "forces": np.asarray(work.get_forces(), dtype=np.float32),
        }


def test_float64_defaults_are_unchanged(lj_engine):
    """The float64 step must stay at the historical default, or every number moves."""
    assert lj_engine.fd_step(order=1, accuracy=2) == FD_MIN_STEP
    assert lj_engine.fd_step(order=2, accuracy=2) == FD_MIN_STEP


def test_float32_step_is_larger_and_ordered():
    f64 = ModelEngine(clean_lj(rc=8.0), precision="float64")
    f32 = ModelEngine(clean_lj(rc=8.0), precision="float32")
    assert f32.fd_step(order=1, accuracy=2) > f64.fd_step(order=1, accuracy=2)
    # a second derivative tolerates round-off less well, so it needs a coarser step
    assert f32.fd_step(order=2, accuracy=2) > f32.fd_step(order=1, accuracy=2)
    # ... and a higher-order stencil coarser still
    assert f32.fd_step(order=1, accuracy=4) > f32.fd_step(order=1, accuracy=2)


def test_precision_is_detected_from_what_the_model_returns(cluster13):
    engine = from_ase_calculator(Float32LJ())
    assert not engine.precision_is_known  # nothing observed yet
    assert engine.precision == "float64"  # the safe assumption until evidence
    assert engine.detect_precision(cluster13) == "float32"
    assert engine.precision_is_known
    assert engine.fd_step(order=1, accuracy=2) > FD_MIN_STEP


def test_declared_precision_wins_over_observation(cluster13):
    engine = ModelEngine(Float32LJ(), precision="float64")
    engine.forces(cluster13)  # observes float32, but the declaration is authoritative
    assert engine.precision == "float64"
    assert engine.fd_step(order=1, accuracy=2) == FD_MIN_STEP


@pytest.mark.parametrize(
    "name,key",
    [("zero_modes", "eps"), ("betti", "delta"), ("conservativeness", "step")],
)
def test_probes_record_a_larger_step_for_float32(cluster13, name, key):
    f64 = get_eval(name).run(ModelEngine(clean_lj(rc=8.0), precision="float64"), cluster13, seed=0)
    f32 = get_eval(name).run(ModelEngine(clean_lj(rc=8.0), precision="float32"), cluster13, seed=0)
    assert f32.details[key] > f64.details[key]
    assert f64.details["precision"] == "float64"
    assert f32.details["precision"] == "float32"


def test_explicit_step_is_still_honoured(cluster13):
    """A caller pinning eps must get exactly that, whatever the precision."""
    engine = ModelEngine(clean_lj(rc=8.0), precision="float32")
    r = get_eval("zero_modes").run(engine, cluster13, seed=0, eps=2.5e-3)
    assert r.details["eps"] == 2.5e-3


def test_conservativeness_flags_a_round_off_limited_scan(cluster13):
    """When the physical scan cannot hold a big enough spacing, say so rather than lie."""
    f64 = get_eval("conservativeness").run(
        ModelEngine(clean_lj(rc=8.0), precision="float64"), cluster13, seed=0
    )
    f32 = get_eval("conservativeness").run(
        ModelEngine(clean_lj(rc=8.0), precision="float32"), cluster13, seed=0
    )
    assert f64.details["round_off_limited"] is False
    assert f32.details["round_off_limited"] is True
    # the grid is thinned, never widened: the scan range is a physical choice
    assert f32.details["n_points"] < f64.details["n_points"]
    assert f32.details["scan_range"] == f64.details["scan_range"]


def test_suite_settles_precision_before_running(cluster13):
    """The report records the precision, and it is detected even for an auto engine."""
    engine = from_ase_calculator(Float32LJ())
    report = Suite.from_names(["zero_modes", "betti"]).run(engine, cluster13, seed=0)
    assert report.metadata["precision"] == "float32"
    step = report.results[0].details["eps"]
    assert step > FD_MIN_STEP  # not the float64 default, despite running first
