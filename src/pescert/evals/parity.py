"""Improper-symmetry stress selection rule (parity of the stress tensor).

Section: Symmetry & invariance.

Identity.  The exact surface is invariant under the full orthogonal group O(3), not only
its proper part SO(3).  Under an orthogonal ``Q`` the stress transforms as a rank-two
tensor, ``sigma(Q R) = Q sigma(R) Q^T``; for the inversion ``P = -I`` the two sign factors
cancel, giving ``sigma(P R) = sigma(R)`` exactly.

We evaluate the stress of the substrate and of its parity image (two calls).  The
parity-even average ``sigma_bar`` restores the identity by construction, and the deviation
``d_P = ||sigma - sigma_bar||_F / ||sigma_bar||_F`` is the parity-odd remainder.

On the Pmm2 substrate the shear ``sigma_xy`` is even under the proper C2z and eliminated
only by the improper mirrors, so it is a pure improper-symmetry residual, structurally
unprotected in an SE(3)-but-not-E(3) architecture.  The shears ``sigma_xz`` and
``sigma_yz`` are odd under C2z and vanish for any rotation-equivariant model, serving as
built-in controls.

Target: exactly **0**.  Skips when the model exposes no stress.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from .base import Budget, Eval


@register("parity")
class ParityStress(Eval):
    section = "Symmetry & invariance"
    target = 0.0
    substrate_kind = "p2mm"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        lam: float = 1e-3,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        if not np.any(atoms.pbc):
            return self._skip("substrate is not periodic; stress is undefined", budget)
        if not engine.has_stress(atoms):
            return self._skip("model does not expose a stress", budget)

        sigma = engine.stress(atoms)  # (3, 3), 1 call

        # parity image: invert the fractional coordinates through the cell centre.  The
        # Bravais lattice is centrosymmetric, so the cell itself is unchanged.
        image = atoms.copy()
        frac = atoms.get_scaled_positions()
        image.set_scaled_positions((1.0 - frac) % 1.0)
        sigma_image = engine.stress(image)  # 1 call

        sigma_bar = 0.5 * (sigma + sigma_image)  # parity-even reference

        def _frob(m: np.ndarray) -> float:
            return float(np.linalg.norm(m))

        denom = _frob(sigma_bar) + 1e-30
        d_parity = _frob(sigma - sigma_bar) / denom
        score = score_from_defect(d_parity, lam)
        gate = d_parity < 5 * lam

        # sigma_xy is the improper channel, the other two shears the proper control;
        # a parity-blind model leaks into sigma_xy specifically
        odd = sigma - sigma_bar
        improper_xy = abs(odd[0, 1]) / denom
        proper_shear = max(abs(odd[0, 2]), abs(odd[1, 2])) / denom

        from ..engine import tensor_to_voigt

        return self._result(
            raw_defect=d_parity,
            score=score,
            n_model_calls=budget.used,
            gate=gate,
            details={
                "lambda": lam,
                "d_parity": d_parity,
                "improper_channel_xy": improper_xy,
                "proper_control_shear": proper_shear,
                "stress_pred_voigt": tensor_to_voigt(sigma).tolist(),
                "stress_image_voigt": tensor_to_voigt(sigma_image).tolist(),
                "stress_sym_voigt": tensor_to_voigt(sigma_bar).tolist(),
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
