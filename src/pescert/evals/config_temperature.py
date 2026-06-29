"""NEW-2a -- configurational temperature (Rugh; Butler-Ayton-Jepps-Evans).

Spec: section 2, NEW-2a.

Identity.  A purely *configurational* thermometer, with no reference to momenta:
``k_B T_config = <||F||^2> / <Tr H>``.  At equilibrium at temperature ``T``,
``T_config = T`` exactly, so along the model's own thermostatted trajectory the ratio
``T_config / T_kin -> 1``.

Target: exactly **1**.  ``<||F||^2>`` is free (forces are evaluated anyway); ``<Tr H>``
is a matrix-free Hutchinson estimate ``<z^T H z>`` over a few seeded +/-1 vectors.  A
non-conservative model samples no Boltzmann distribution, so the ratio drifts off 1
even with a perfect thermostat.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._md import seeded_velocities
from .base import Budget, Eval


@register("config_temperature")
class ConfigTemperature(Eval):
    target = 1.0
    substrate_kind = "cluster"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        temperature_K: float = 40.0,
        warmup_steps: int = 250,
        n_steps: int = 400,
        sample_every: int = 4,
        n_hutchinson: int = 4,
        dt: float = 0.5,
        friction: float = 0.1,
        eps: float = 1e-3,
        scale: float = 1,
        do_relax: bool = True,
        **cfg,
    ) -> EvalResult:
        from ase.md.langevin import Langevin

        budget = Budget(engine, max_calls)
        rng = np.random.default_rng(seed + 12345)

        # start near the model's own minimum so the thermostatted trajectory samples
        # small equilibrium fluctuations rather than collapsing from a poor geometry.
        start = engine.relax(atoms, fmax=1e-2) if do_relax else atoms
        n_atoms = len(atoms)
        work = engine.attach(start)
        work.set_velocities(seeded_velocities(start, temperature_K, seed))
        dyn = Langevin(
            work,
            timestep=dt * units.fs,
            temperature_K=temperature_K,
            friction=friction/units.fs,
            rng=np.random.default_rng(seed),
        )

        f2_samples: list[float] = []
        trh_samples: list[float] = []
        tkin_samples: list[float] = []

        def sample():
            forces = work.get_forces()
            f2_samples.append(float(np.sum(forces**2)))
            snap = start.copy()
            snap.set_positions(work.get_positions())
            trh = 0.0
            for _ in range(n_hutchinson):
                z = rng.choice([-1.0, 1.0], size=3 * n_atoms)
                trh += float(z @ engine.hvp(snap, z, eps=eps))
            trh_samples.append(trh / n_hutchinson)
            # kinetic temperature with the full 3N dof, consistent with the dof-free
            # configurational temperature (ASE's get_temperature removes 3 COM dof,
            # biasing the ratio by 3N/(3N-3)).
            tkin_samples.append(2.0 * work.get_kinetic_energy() / (3 * n_atoms * units.kB))

        # budget-aware step count
        per_sample = 1 + 2 * n_hutchinson
        n_samples_planned = n_steps // sample_every
        while budget.would_exceed(
            warmup_steps + n_steps + per_sample * n_samples_planned
        ) and n_steps > 40:
            n_steps -= 20
            n_samples_planned = n_steps // sample_every
        dyn.run(warmup_steps)  # equilibrate before sampling
        dyn.attach(sample, interval=sample_every)
        dyn.run(n_steps)

        mean_f2 = float(np.mean(f2_samples))
        mean_trh = float(np.mean(trh_samples))
        t_config = mean_f2 / (units.kB * mean_trh) if mean_trh > 0 else float("nan")
        t_kin = float(np.mean(tkin_samples))
        # Self-normalizing ratio T_config / T_kin: robust to the actual sampled
        # temperature (both scale together).  A non-conservative model samples no
        # Boltzmann distribution, so T_config / T_kin drifts off 1.
        ratio = t_config / t_kin if t_kin > 0 else float("nan")
        defect = abs(ratio - 1.0)
        score = score_from_defect(defect, scale)

        return self._result(
            raw_defect=defect,
            score=score,
            n_model_calls=budget.used,
            gate=None,
            details={
                "T_config": t_config,
                "T_kin": t_kin,
                "ratio": ratio,
                "mean_F2": mean_f2,
                "mean_TrH": mean_trh,
                "n_samples": len(f2_samples),
                "temperature_K": temperature_K,
                "n_hutchinson": n_hutchinson,
                "eps": eps,
                "scale": scale,
                "ratio_series": [
                    float(f2 / (units.kB * trh * tk))
                    for f2, trh, tk in zip(f2_samples, trh_samples, tkin_samples)
                    if trh > 0 and tk > 0
                ],
            },
        )
