"""KNOWN -- 1D conservativeness (MLIP-Arena "conservation deviation").

Spec: section 1 (prior art) / section 0 family 2 (self-consistency).

Identity.  Along a 1D bond stretch parameterised by ``s`` (displace one atom of a
bond along the bond axis), the projected force must equal minus the derivative of the
energy: ``F_j . u_hat = -dE/ds`` for a conservative model.  We compare the model's
projected force to a central difference of its own energy.

Target: exactly **0** (mean absolute deviation between projected force and
``-dE/ds``).  NEW-3(iii) generalises this to transverse directions; this proxy stays
strictly collinear.
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
    section = "KNOWN"
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

        # 4th-order central difference of energy: F.u_hat = -dE/ds.  The high-order
        # stencil keeps truncation error negligible on steep (e.g. repulsive-wall)
        # regions, so a finite defect reflects genuine non-conservativeness (force not
        # equal to -dE/ds, independent of step) rather than discretization.
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
                "scale": scale,
                "max_residual": float(residual.max()),
                "s_grid": s_grid.tolist(),
                "projected_force": fproj.tolist(),
                "force_from_energy": f_from_e.tolist(),
                "budget_limited": budget.would_exceed(0) if max_calls else False,
            },
        )
