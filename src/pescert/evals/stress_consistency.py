"""OOB-2 -- stress vs energy-gradient (virial / thermodynamic) consistency.

Spec: section 3, OOB-2.

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
from .base import Budget, Eval


def _strain_tensor(voigt_strain: np.ndarray) -> np.ndarray:
    """Symmetric strain tensor from an engineering Voigt strain vector."""
    e1, e2, e3, e4, e5, e6 = voigt_strain
    return np.array(
        [
            [e1, e6 / 2.0, e5 / 2.0],
            [e6 / 2.0, e2, e4 / 2.0],
            [e5 / 2.0, e4 / 2.0, e3],
        ]
    )


@register("stress_consistency")
class StressConsistency(Eval):
    section = "OOB-2"
    target = 0.0
    substrate_kind = "bulk"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        delta: float = 1e-3,
        scale: float = 0.05,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        if not np.any(atoms.pbc):
            return self._skip("substrate is not periodic; stress is undefined", budget)
        if not engine.has_stress(atoms):
            return self._skip("model does not expose a stress; OOB-2 not applicable", budget)

        sigma_pred = engine.stress_voigt(atoms)  # 1 call
        cell0 = np.array(atoms.get_cell())
        volume = atoms.get_volume()

        sigma_fd = np.empty(6)
        for k in range(6):
            voigt = np.zeros(6)
            voigt[k] = delta
            eps = _strain_tensor(voigt)
            ep = self._strained_energy(engine, atoms, cell0, eps)
            em = self._strained_energy(engine, atoms, cell0, -eps)
            sigma_fd[k] = (ep - em) / (2 * delta) / volume

        denom = np.linalg.norm(sigma_fd) + 1e-12
        defect = float(np.linalg.norm(sigma_pred - sigma_fd) / denom)
        score = score_from_defect(defect, scale)
        gate = defect < 5 * scale

        return self._result(
            raw_defect=defect,
            score=score,
            n_model_calls=budget.used,
            gate=gate,
            details={
                "delta": delta,
                "scale": scale,
                "stress_pred_voigt": sigma_pred.tolist(),
                "stress_fd_voigt": sigma_fd.tolist(),
                "abs_residual": float(np.linalg.norm(sigma_pred - sigma_fd)),
            },
        )

    @staticmethod
    def _strained_energy(engine, atoms, cell0, eps_tensor):
        f = np.eye(3) + eps_tensor
        work = atoms.copy()
        work.set_cell(cell0 @ f.T, scale_atoms=True)
        return engine.energy(work)

    def _skip(self, message: str, budget: Budget) -> EvalResult:
        return self._result(
            raw_defect=float("nan"),
            score=float("nan"),
            n_model_calls=budget.used,
            gate=None,
            details={"skipped": True, "message": message},
        )
