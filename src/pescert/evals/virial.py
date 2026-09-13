"""Virial theorem (Clausius) -- a Hessian-free equilibrium thermometer.

Section: Statistical mechanics.

Identity.  The Clausius virial theorem is exact for any bound classical system in a
stationary state, harmonic or not: the virial of the internal forces balances twice the
internal kinetic energy,

    rho_vir = < -sum_i (R_i - R_ref) . F_i > / ((3N-6) k_B T_kin^int)  ->  1 .

We read the shared Langevin trajectory (:mod:`._trajectory`) and at each snapshot
accumulate the numerator and the internal kinetic temperature obtained after projecting
rigid translation and rotation out of the velocities (the six unconfined DOF carry
kinetic energy but no confining virial).

Variance reduction.  Positions are referenced to the model's own relaxed structure rather
than the instantaneous centre of mass.  Because the internal forces are orthogonal to the
rigid-body modes this has the *same* expectation, but it drops the large zero-mean
``R^eq . F`` term that otherwise swamps the signal on a stiff cluster.

Being Hessian-free -- no finite-difference curvature, no trace estimator -- this ties
another independent pair of channels, positions and forces, to the kinetic energy.  Like
the other equilibrium probes it is a finite-time average; the exact target holds in the
long-trajectory limit.

Target: exactly **1**.
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
    stored_steps,
)
from .base import Budget, Eval


def _internal_kinetic_energy(masses: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> float:
    """Kinetic energy with rigid translation *and* rotation projected out (3N-6 DOF)."""
    m = masses[:, None]
    total_m = masses.sum()
    rcm = (m * pos).sum(0) / total_m
    vcm = (m * vel).sum(0) / total_m
    rel = pos - rcm
    vrel = vel - vcm
    # angular momentum and inertia tensor about the centre of mass
    ang_mom = (m * np.cross(rel, vrel)).sum(0)
    inertia = np.zeros((3, 3))
    for i in range(len(masses)):
        r = rel[i]
        inertia += masses[i] * (r @ r * np.eye(3) - np.outer(r, r))
    try:
        omega = np.linalg.solve(inertia, ang_mom)
    except np.linalg.LinAlgError:
        omega = np.zeros(3)
    v_internal = vrel - np.cross(np.tile(omega, (len(masses), 1)), rel)
    return 0.5 * float(np.sum(m * v_internal**2))


@register("virial")
class Virial(Eval):
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
        sample_every: int = 4,
        dt: float = DT,
        friction: float = FRICTION,
        scale: float = 1.0,
        do_relax: bool = True,
        relax_fmax: float = RELAX_FMAX,
        trajectory_cache: TrajectoryCache | None = None,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        n_atoms = len(atoms)
        if n_atoms < 3:
            return self._skip("need at least 3 atoms for a bound cluster virial", budget)

        # relax to the model's own minimum so R^eq is a genuine stationary point
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
        r_eq = traj.reference.get_positions()
        masses = traj.reference.get_masses()
        dof = 3 * n_atoms - 6

        # only MD steps not already on the shared trajectory cost anything; the sampled
        # positions, forces and velocities come with the frames
        while (
            budget.would_exceed(max(0, n_steps - traj.n_steps)) and n_steps > 200
        ):
            n_steps -= 100
        traj.extend(n_steps)
        frames = traj.frames(sample_every=sample_every, n_steps=n_steps)

        virial_samples: list[float] = []
        kin_int_samples: list[float] = []
        for pos, forces, vel in zip(frames.positions, frames.forces, frames.velocities):
            virial_samples.append(-float(np.sum((pos - r_eq) * forces)))
            kin_int_samples.append(_internal_kinetic_energy(masses, pos, vel))

        mean_virial = float(np.mean(virial_samples)) if virial_samples else float("nan")
        mean_kin_int = float(np.mean(kin_int_samples)) if kin_int_samples else float("nan")
        # rho = <numerator> / (2 <KE_internal>) = <numerator> / ((3N-6) k_B T_int)
        rho = mean_virial / (2.0 * mean_kin_int) if mean_kin_int > 0 else float("nan")
        t_int = 2.0 * mean_kin_int / (dof * units.kB) if dof > 0 else float("nan")
        defect = abs(rho - 1.0)
        score = score_from_defect(defect, scale)

        return self._result(
            raw_defect=defect,
            score=score,
            n_model_calls=budget.used,
            gate=None,
            details={
                "rho_vir": rho,
                "T_int_kin": t_int,
                "mean_virial": mean_virial,
                "mean_kinetic_internal": mean_kin_int,
                "n_samples": len(virial_samples),
                "temperature_K": temperature_K,
                "dof": dof,
                "scale": scale,
                "trajectory": {
                    "reused": reused,
                    "owner": traj.owner,
                    "steps_available_before": have,
                    "n_steps": n_steps,
                    "sample_every": sample_every,
                    "md_calls": int(traj.n_model_calls),
                },
                "rho_series": [
                    v / (2.0 * k)
                    for v, k in zip(virial_samples, kin_int_samples)
                    if k > 0
                ],
            },
        )

    def _skip(self, message: str, budget: Budget) -> EvalResult:
        return self._result(
            raw_defect=float("nan"),
            score=float("nan"),
            n_model_calls=budget.used,
            gate=None,
            details={"skipped": True, "message": message},
        )
