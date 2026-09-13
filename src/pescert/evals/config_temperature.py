"""Configurational temperature (Rugh; Butler-Ayton-Jepps-Evans).

Section: Statistical mechanics.

Identity.  A purely *configurational* thermometer, with no reference to momenta:
``k_B T_config = <||F||^2> / <Tr H>``.  At equilibrium at temperature ``T``,
``T_config = T`` exactly, so along the model's own thermostatted trajectory the ratio
``T_config / T_kin -> 1``.

Target: exactly **1**.  ``<||F||^2>`` is free (forces are evaluated anyway); ``<Tr H>``
is a matrix-free Hutchinson estimate ``<z^T H z>`` over a few seeded +/-1 vectors.  A
non-conservative model samples no Boltzmann distribution, so the ratio drifts off 1
even with a perfect thermostat.

This proxy owns the shared Langevin trajectory (:mod:`._trajectory`); only the
Hutchinson Hessian-vector products are its own cost.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._trajectory import (
    DT,
    FRICTION,
    N_STEPS,
    RELAX_FMAX,
    TEMPERATURE_K,
    WARMUP_STEPS,
    Thermostat,
    TrajectoryCache,
    get_trajectory,
    n_samples_for,
    stored_steps,
)
from .base import Budget, Eval


@register("config_temperature")
class ConfigTemperature(Eval):
    section = "Statistical mechanics"
    target = 1.0
    substrate_kind = "cluster"
    uses_trajectory = True

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        temperature_K: float = TEMPERATURE_K,
        warmup_steps: int = WARMUP_STEPS,
        n_steps: int = N_STEPS,
        sample_every: int = 10,
        n_hutchinson: int = 4,
        dt: float = DT,
        friction: float = FRICTION,
        eps: float | None = None,
        scale: float = 1,
        do_relax: bool = True,
        relax_fmax: float = RELAX_FMAX,
        trajectory_cache: TrajectoryCache | None = None,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        eps = engine.fd_step(order=1, accuracy=2) if eps is None else eps
        rng = np.random.default_rng(seed + 12345)
        n_atoms = len(atoms)

        # start from the model's own minimum, so the trajectory samples small
        # equilibrium fluctuations rather than collapsing from a poor geometry
        thermostat = Thermostat(
            seed=seed,
            temperature_K=temperature_K,
            warmup_steps=warmup_steps,
            dt=dt,
            friction=friction,
            relax_fmax=relax_fmax,
            do_relax=do_relax,
        )
        have = stored_steps(trajectory_cache, atoms, thermostat)
        traj, reused = get_trajectory(
            engine, atoms, thermostat, owner=self.name, cache=trajectory_cache
        )

        # only MD steps not already on the shared trajectory cost anything; the sampled
        # forces come with the frames, the Hutchinson products (2 calls each) do not
        def projected(steps: int) -> int:
            new_md = max(0, steps - traj.n_steps)
            n_samples = n_samples_for(warmup_steps, steps, sample_every)
            return new_md + (1 + 2 * n_hutchinson) * n_samples

        while budget.would_exceed(projected(n_steps)) and n_steps > 40:
            n_steps -= 20
        traj.extend(n_steps)
        frames = traj.frames(sample_every=sample_every, n_steps=n_steps)

        f2_samples: list[float] = []
        trh_samples: list[float] = []
        tkin_samples: list[float] = []
        masses = traj.reference.get_masses()[:, None]
        for pos, forces, vel in zip(frames.positions, frames.forces, frames.velocities):
            f2_samples.append(float(np.sum(forces**2)))
            snap = traj.reference.copy()
            snap.set_positions(pos)
            trh = 0.0
            for _ in range(n_hutchinson):
                z = rng.choice([-1.0, 1.0], size=3 * n_atoms)
                trh += float(z @ engine.hvp(snap, z, eps=eps))
            trh_samples.append(trh / n_hutchinson)
            # full 3N dof, to match the dof-free configurational temperature (ASE's
            # get_temperature removes 3 COM dof, biasing the ratio by 3N/(3N-3))
            kinetic = 0.5 * float(np.sum(masses * vel**2))
            tkin_samples.append(2.0 * kinetic / (3 * n_atoms * units.kB))

        mean_f2 = float(np.mean(f2_samples)) if f2_samples else float("nan")
        mean_trh = float(np.mean(trh_samples)) if trh_samples else float("nan")
        t_config = mean_f2 / (units.kB * mean_trh) if mean_trh > 0 else float("nan")
        t_kin = float(np.mean(tkin_samples)) if tkin_samples else float("nan")
        # the ratio is self-normalizing: both temperatures scale with the sampled one
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
                "precision": engine.precision,
                "scale": scale,
                "trajectory": {
                    "reused": reused,
                    "owner": traj.owner,
                    "steps_available_before": have,
                    "n_steps": n_steps,
                    "sample_every": sample_every,
                    "md_calls": int(traj.n_model_calls),
                },
                "ratio_series": [
                    float(f2 / (units.kB * trh * tk))
                    for f2, trh, tk in zip(f2_samples, trh_samples, tkin_samples)
                    if trh > 0 and tk > 0
                ],
            },
        )
