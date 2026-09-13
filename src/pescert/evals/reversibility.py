"""Time-reversibility: an NVE round-trip (classic MD code-validation trick).

Section: Self-consistency.

Identity.  Integrate NVE forward N steps, flip all velocities, integrate N steps back.
A smooth, conservative PES with a symplectic integrator returns near the start.

Target: exactly **0** (round-trip return error ``||R0' - R0||``).  This is distinct
from secular energy drift (also reported): a model can have small per-step drift yet
large irreversibility from discontinuities, or vice versa.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._md import seeded_velocities
from .base import Budget, Eval


@register("reversibility")
class Reversibility(Eval):
    section = "Self-consistency"
    target = 0.0
    substrate_kind = "cluster"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        n_steps: int = 100,
        dt: float = 1.0,
        temperature_K: float = 30.0,
        scale: float = 0.001,
        **cfg,
    ) -> EvalResult:
        from ase.md.verlet import VelocityVerlet

        budget = Budget(engine, max_calls)
        while budget.would_exceed(2 * n_steps + 2) and n_steps > 10:
            n_steps -= 5

        work = engine.attach(atoms)
        work.set_velocities(seeded_velocities(atoms, temperature_K, seed))
        r0 = work.get_positions().copy()

        energies: list[float] = []

        def record():
            energies.append(float(work.get_total_energy()))

        dyn = VelocityVerlet(work, timestep=dt * units.fs)
        dyn.attach(record, interval=1)
        dyn.run(n_steps)

        work.set_velocities(-work.get_velocities())  # reverse and integrate back
        dyn.run(n_steps)

        r1 = work.get_positions()
        roundtrip = float(np.sqrt(np.mean(np.sum((r1 - r0) ** 2, axis=1))))
        score = score_from_defect(roundtrip, scale)

        steps = np.arange(len(energies))
        drift_slope = (
            float(np.polyfit(steps, energies, 1)[0]) if len(energies) > 2 else float("nan")
        )

        return self._result(
            raw_defect=roundtrip,
            score=score,
            n_model_calls=budget.used,
            gate=roundtrip < 5 * scale,
            details={
                "n_steps": int(n_steps),
                "dt_fs": dt,
                "temperature_K": temperature_K,
                "scale": scale,
                "roundtrip_rms_disp": roundtrip,
                "energy_drift_per_step": drift_slope,
                "energy_first": energies[0] if energies else None,
                "energy_max_forward": max(energies) if energies else None,
                "seed": seed,
            },
        )
