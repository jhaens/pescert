"""Collinear conservativeness (MLIP-Arena "conservation deviation").

Section: Self-consistency.

Identity.  Along a 1D bond stretch parameterised by ``s`` (displace one atom of a
bond along the bond axis), the projected force must equal minus the derivative of the
energy: ``F_j . u_hat = -dE/ds`` for a conservative model.  We compare the model's
projected force to a central difference of its own energy.

Target: exactly **0** (mean absolute deviation between projected force and
``-dE/ds``).  The trimer probe generalises this to transverse directions; this proxy
stays strictly collinear.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._geometry import closest_pair
from .base import Budget, Eval


@register("conservativeness")
class Conservativeness(Eval):
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
        bond: tuple[int, int] | None = None,
        scan_range: float = 0.15,
        n_points: int = 31,
        min_points: int = 9,  # floor when thinning the grid for a low-precision model
        scale: float = 0.05,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        i, j = bond if bond is not None else closest_pair(atoms)
        r0 = atoms.get_positions()
        u = r0[j] - r0[i]
        u = u / np.linalg.norm(u)

        if n_points % 2 == 0:
            n_points += 1
        while budget.would_exceed(n_points) and n_points > 5:
            n_points -= 2

        # This probe's finite-difference step is the scan spacing, not a free eps, so
        # respect the model's precision by thinning the grid rather than by shrinking it:
        # the scan range is a physical choice and must stay put.  A float32 model may not
        # reach the required spacing at all within the range, which is recorded rather
        # than hidden -- the residual is then round-off, not non-conservativeness.
        h_min = engine.fd_step(order=1, accuracy=4)
        round_off_limited = False
        if 2.0 * scan_range / (n_points - 1) < h_min:
            n_fit = int(2.0 * scan_range / h_min) + 1
            n_points = max(min_points, n_fit if n_fit % 2 else n_fit - 1)
            round_off_limited = 2.0 * scan_range / (n_points - 1) < h_min

        s_grid = np.linspace(-scan_range, scan_range, n_points)
        h = s_grid[1] - s_grid[0]

        energies = np.empty(n_points)
        fproj = np.empty(n_points)
        for idx, s in enumerate(s_grid):
            a = atoms.copy()
            pos = r0.copy()
            pos[j] = r0[j] + s * u
            a.set_positions(pos)
            e, f = engine.energy_forces(a)
            energies[idx] = e
            fproj[idx] = float(f[j] @ u)

        # 4th-order central difference of E.  The high-order stencil keeps truncation
        # error negligible on steep regions, so a finite defect is genuine
        # non-conservativeness rather than discretization -- provided the spacing is
        # above the model's round-off floor, which is what h_min enforces above.
        f_from_e = (
            -energies[:-4] + 8 * energies[1:-3] - 8 * energies[3:-1] + energies[4:]
        ) / (12 * h)
        residual = np.abs(fproj[2:-2] - f_from_e)
        defect = float(np.mean(residual))
        score = score_from_defect(defect, scale)
        gate = defect < 5 * scale

        return self._result(
            raw_defect=defect,
            score=score,
            n_model_calls=budget.used,
            gate=gate,
            details={
                "bond": (int(i), int(j)),
                "scan_range": scan_range,
                "n_points": int(n_points),
                "step": float(h),
                "step_min_for_precision": float(h_min),
                "precision": engine.precision,
                "round_off_limited": bool(round_off_limited),
                "scale": scale,
                "max_residual": float(residual.max()),
                "s_grid": s_grid.tolist(),
                "projected_force": fproj.tolist(),
                "force_from_energy": f_from_e.tolist(),
                "budget_limited": budget.would_exceed(0) if max_calls else False,
            },
        )
