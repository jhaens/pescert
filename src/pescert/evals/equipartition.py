"""NEW-2b -- per-mode equipartition.

Spec: section 2, NEW-2b.

Identity.  Near a minimum, in mass-weighted normal coordinates ``Q_alpha`` with
frequencies ``omega_alpha``, equipartition demands for *every* non-zero mode
``R_alpha = omega_alpha^2 <Q_alpha^2> / (k_B T) -> 1``.  Two independent estimates feed
it: ``omega_alpha^2`` from the **Hessian** (curvature channel) and ``<Q_alpha^2>`` from
**fluctuations** along ``e_alpha`` in a short trajectory (dynamics channel).

Target: exactly **1** per mode.  The whole spectrum ``{R_alpha}`` must collapse onto 1;
the *spread* about 1 is the discriminating, mode-resolved signal a single E/F RMSE
cannot capture.  Run at low T / small displacement so anharmonicity is negligible.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._md import seeded_velocities
from .base import Budget, Eval


@register("equipartition")
class Equipartition(Eval):
    target = 1.0
    substrate_kind = "cluster"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        temperature_K: float = 20.0,
        warmup_steps: int = 250,
        n_steps: int = 350,
        sample_every: int = 1,
        dt: float = 1.0,
        friction: float = 0.1,
        eps: float = 1e-3,
        relax_fmax: float = 1e-3,
        tol_rel: float = 1e-3,
        scale: float = 0.5,
        do_relax: bool = True,
        **cfg,
    ) -> EvalResult:
        from ase.md.langevin import Langevin

        budget = Budget(engine, max_calls)
        relaxed = engine.relax(atoms, fmax=relax_fmax) if do_relax else atoms.copy()
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

        # short trajectory; accumulate <Q_alpha^2>
        work = engine.attach(relaxed)
        work.set_velocities(seeded_velocities(relaxed, temperature_K, seed))
        dyn = Langevin(
            work,
            timestep=dt * units.fs,
            temperature_K=temperature_K,
            friction=friction/units.fs,
            rng=np.random.default_rng(seed),
        )
        q2_acc = np.zeros(omega2.shape)
        n_acc = [0]

        def sample():
            dr = (work.get_positions() - r0).reshape(-1)
            q = (sqrt_m * dr) @ modes  # projection onto each mode
            q2_acc[:] += q**2
            n_acc[0] += 1

        # warm up first so <Q^2> reflects the equilibrated amplitude (the trajectory
        # starts at the minimum with zero displacement, which would bias <Q^2> low).
        dyn.run(warmup_steps)
        dyn.attach(sample, interval=sample_every)
        dyn.run(n_steps)

        q2_mean = q2_acc / max(n_acc[0], 1)
        kt = units.kB * temperature_K
        r_alpha = omega2 * q2_mean / kt
        # NOTE: per-mode R_alpha carries irreducible finite-sampling scatter (~0.5 for a
        # short trajectory) even for a perfect model, so the robust scalar defect is how
        # well the spectrum *centers* on 1: |median(R_alpha) - 1| (exact target 1
        # preserved).  The mode-resolved {R_alpha}, its spread and percentiles are
        # reported as the expressive, discriminating diagnostic the spec calls for.
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
                "n_samples": int(n_acc[0]),
                "eps": eps,
                "scale": scale,
                "relax_fmax_reached": float(relaxed.info.get("relax_fmax", np.nan)),
            },
        )
