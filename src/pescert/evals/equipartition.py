"""Per-mode equipartition.

Section: Statistical mechanics.

Identity.  Near a minimum, in mass-weighted normal coordinates ``Q_alpha`` with
frequencies ``omega_alpha``, equipartition demands for *every* non-zero mode
``R_alpha = omega_alpha^2 <Q_alpha^2> / (k_B T) -> 1``.  Two independent estimates feed
it: ``omega_alpha^2`` from the **Hessian** (curvature channel) and ``<Q_alpha^2>`` from
**fluctuations** along ``e_alpha`` in a short trajectory (dynamics channel).

Target: exactly **1** per mode.  The whole spectrum ``{R_alpha}`` must collapse onto 1;
the *spread* about 1 is the discriminating, mode-resolved signal a single E/F RMSE
cannot capture.  Run at low T / small displacement so anharmonicity is negligible.

The dynamics channel reads the shared Langevin trajectory (:mod:`._trajectory`); only
the finite-difference Hessian (6N calls) is this proxy's own cost.
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


@register("equipartition")
class Equipartition(Eval):
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
        sample_every: int = 1,
        dt: float = DT,
        friction: float = FRICTION,
        eps: float | None = None,
        relax_fmax: float = RELAX_FMAX,
        tol_rel: float = 1e-3,
        scale: float = 1.0,
        do_relax: bool = True,
        trajectory_cache: TrajectoryCache | None = None,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        eps = engine.fd_step(order=1, accuracy=2) if eps is None else eps

        # the trajectory's reference geometry is the relaxed structure both channels
        # need, so curvature and fluctuations are taken about the same minimum
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
        relaxed = traj.reference
        r0 = relaxed.get_positions()
        masses = relaxed.get_masses()
        sqrt_m = np.repeat(np.sqrt(masses), 3)

        h = engine.hessian(relaxed, eps=eps, method="fd_forces")
        h = 0.5 * (h + h.T)
        h_mw = h / np.outer(sqrt_m, sqrt_m)
        eigvals, eigvecs = np.linalg.eigh(h_mw)

        lam_max = float(np.max(np.abs(eigvals))) + 1e-30
        keep = np.abs(eigvals) > tol_rel * lam_max
        keep &= eigvals > 0  # only real (stable) modes carry equipartition
        omega2 = eigvals[keep]
        modes = eigvecs[:, keep]  # columns are mass-weighted eigenvectors

        # accumulate <Q_alpha^2>.  Sampling starts after the warmup so <Q^2> reflects the
        # equilibrated amplitude: the run starts at the minimum, which would bias it low.
        traj.extend(n_steps)
        frames = traj.frames(sample_every=sample_every, n_steps=n_steps)
        dr = (frames.positions - r0).reshape(len(frames), -1)
        q = (dr * sqrt_m) @ modes  # projection of each frame onto each mode
        q2_mean = (
            (q**2).mean(axis=0) if len(frames) else np.full(omega2.shape, np.nan)
        )

        kt = units.kB * temperature_K
        r_alpha = omega2 * q2_mean / kt
        # Per-mode R_alpha carries irreducible finite-sampling scatter (~0.5 on a short
        # trajectory) even for a perfect model, so the scalar defect is how well the
        # spectrum *centers* on 1.  The mode-resolved {R_alpha} and its spread are
        # reported as the discriminating diagnostic.
        r_median = float(np.median(r_alpha)) if r_alpha.size else float("nan")
        median_abs_dev = float(np.median(np.abs(r_alpha - 1.0))) if r_alpha.size else float("nan")
        defect = abs(r_median - 1.0)
        score = score_from_defect(defect, scale)

        pcts = np.percentile(r_alpha, [5, 25, 50, 75, 95]) if r_alpha.size else np.full(5, np.nan)
        return self._result(
            raw_defect=defect,
            score=score,
            n_model_calls=budget.used,
            gate=None,
            details={
                "n_modes": int(r_alpha.size),
                "R_alpha": r_alpha.tolist(),
                "omega2": omega2.tolist(),
                "R_median": r_median,
                "R_median_abs_dev": median_abs_dev,
                "R_spread_std": float(np.std(r_alpha)) if r_alpha.size else float("nan"),
                "R_percentiles_5_25_50_75_95": pcts.tolist(),
                "temperature_K": temperature_K,
                "n_samples": len(frames),
                "eps": eps,
                "precision": engine.precision,
                "scale": scale,
                "relax_fmax_reached": float(relaxed.info.get("relax_fmax", np.nan)),
                "trajectory": {
                    "reused": reused,
                    "owner": traj.owner,
                    "steps_available_before": have,
                    "n_steps": n_steps,
                    "sample_every": sample_every,
                    "md_calls": int(traj.n_model_calls),
                },
            },
        )
