"""KNOWN -- equivariance under global rotation.

Spec: section 1 (prior art) / section 0 family 1 (symmetry).

Identity.  The energy is invariant and the forces are equivariant under any global
rotation ``Q``: ``E(QR) = E(R)`` and ``F(QR) = Q F(R)``.

Target: exactly **0** for both the energy-invariance residual and the
force-equivariance residual (combined into the primary defect).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._geometry import random_rotations
from .base import Budget, Eval


@register("equivariance")
class Equivariance(Eval):
    target = 0.0
    substrate_kind = "cluster"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        n_rot: int = 8,
        scale_energy: float = 1e-3,
        scale_force: float = 1e-3,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        while budget.would_exceed(1 + n_rot) and n_rot > 1:
            n_rot -= 1

        pos = atoms.get_positions()
        com = pos.mean(0)
        rel = pos - com
        e0, f0 = engine.energy_forces(atoms)

        rotations = random_rotations(n_rot, seed)
        e_res = []
        f_res = []
        for q in rotations:
            rotated = atoms.copy()
            rotated.set_positions(rel @ q.T + com)
            e, f = engine.energy_forces(rotated)
            e_res.append(abs(e - e0))
            f_res.append(float(np.linalg.norm(f - f0 @ q.T)))

        e_defect = float(np.mean(e_res))
        f_defect = float(np.mean(f_res))
        # combine the two normalized residuals; primary defect is their sum
        defect = e_defect / scale_energy + f_defect / scale_force
        score = score_from_defect(defect, scale=1.0)
        gate = (e_defect < 5 * scale_energy) and (f_defect < 5 * scale_force)

        return self._result(
            raw_defect=defect,
            score=score,
            n_model_calls=budget.used,
            gate=gate,
            details={
                "n_rot": int(n_rot),
                "seed": seed,
                "energy_residual": e_defect,
                "force_residual": f_defect,
                "scale_energy": scale_energy,
                "scale_force": scale_force,
                "energy_residuals": e_res,
                "force_residuals": f_res,
            },
        )
