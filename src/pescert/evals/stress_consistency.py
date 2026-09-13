"""Stress--gradient consistency (virial / thermodynamic).

Section: Self-consistency.

Identity.  The stress is the volume-normalised strain derivative of the energy:
``sigma_ij = (1/V) dE/deps_ij``.  We finite-difference the energy over the six
independent strains and compare the full tensor to the model's own stress head -- two
outputs that *should* be derivatives of one scalar, cross-checked with no DFT.

Target: exactly **0** (relative full-tensor residual).  Skips gracefully (``gate`` and
score left undefined) when the model exposes no stress.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._strain import apply_strain, strain_tensor
from .base import Budget, Eval


@register("stress_consistency")
class StressConsistency(Eval):
    section = "Self-consistency"
    target = 0.0
    substrate_kind = "bulk"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        delta: float | None = None,
        scale: float = 0.05,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        delta = engine.fd_step(order=1, accuracy=2) if delta is None else delta
        if not np.any(atoms.pbc):
            return self._skip("substrate is not periodic; stress is undefined", budget)
        if not engine.has_stress(atoms):
            return self._skip("model does not expose a stress", budget)

        sigma_pred = engine.stress_voigt(atoms)  # 1 call
        cell0 = np.array(atoms.get_cell())
        volume = atoms.get_volume()

        sigma_fd = np.empty(6)
        for k in range(6):
            voigt = np.zeros(6)
            voigt[k] = delta
            eps = strain_tensor(voigt)
            ep = engine.energy(apply_strain(atoms, cell0, eps))
            em = engine.energy(apply_strain(atoms, cell0, -eps))
            sigma_fd[k] = (ep - em) / (2 * delta) / volume

        # Frobenius norm of the full (3x3) tensor from its Voigt vector: the three shear
        # components are off-diagonal and therefore enter twice.
        def _frob(v):
            return float(np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2
                                 + 2.0 * (v[3] ** 2 + v[4] ** 2 + v[5] ** 2)))

        denom = _frob(sigma_fd) + 1e-12
        defect = _frob(sigma_pred - sigma_fd) / denom
        score = score_from_defect(defect, scale)
        gate = defect < 5 * scale

        return self._result(
            raw_defect=defect,
            score=score,
            n_model_calls=budget.used,
            gate=gate,
            details={
                "delta": delta,
                "precision": engine.precision,
                "scale": scale,
                "stress_pred_voigt": sigma_pred.tolist(),
                "stress_fd_voigt": sigma_fd.tolist(),
                "abs_residual": float(np.linalg.norm(sigma_pred - sigma_fd)),
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
