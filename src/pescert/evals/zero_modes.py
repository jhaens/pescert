"""NEW-1 -- second-order symmetry residual (zero-mode / acoustic-sum-rule defect).

Spec: section 2, NEW-1.

Identity.  At the model's *own* relaxed structure the Hessian ``H = d2E/dR2`` is forced
by symmetry to have a known null space, with eigenvectors known analytically from
geometry alone:

* isolated cluster: exactly **6** zero modes (5 if linear) -- 3 translations, 3
  rotations ``e_a x (R_i - R_cm)``;
* periodic crystal at Gamma: exactly **3** acoustic zero modes (uniform translation),
  i.e. the acoustic sum rule ``sum_j Phi_{i,j} = 0``.

Targets: the continuous residual ``max_a ||H v_a|| / ||H||_F`` is exactly **0**, and the
near-zero mode count is exactly **6 / 5 / 3**.  Pure self-consistency: no DFT, exact
targets.  A negative (imaginary) eigenvalue at a relaxed minimum is flagged as a
spurious soft mode.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._geometry import is_linear, rigid_zero_modes
from .base import Budget, Eval


@register("zero_modes")
class ZeroModes(Eval):
    section = "NEW-1"
    target = 0.0
    substrate_kind = "cluster"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        eps: float = 1e-3,
        relax_fmax: float = 1e-3,
        relax_steps: int = 200,
        tol_rel: float = 1e-2,
        scale: float = 5e-3,
        do_relax: bool = True,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        periodic = bool(np.any(atoms.pbc))

        if do_relax:
            relaxed = engine.relax(atoms, fmax=relax_fmax, steps=relax_steps)
        else:
            relaxed = atoms.copy()
            relaxed.info["relax_fmax"] = float(np.linalg.norm(engine.forces(relaxed), axis=1).max())
        fmax_reached = float(relaxed.info.get("relax_fmax", np.nan))

        pos = relaxed.get_positions()
        # build analytic zero modes (non-mass-weighted: exact null vectors of plain H)
        rotations = not periodic
        modes, labels = rigid_zero_modes(pos, masses=None, rotations=rotations)
        if periodic:
            expected = 3
        else:
            expected = 5 if is_linear(pos) else 6

        h = engine.hessian(relaxed, eps=eps, method="fd_forces")  # 6N calls
        h_sym = 0.5 * (h + h.T)
        h_fro = np.linalg.norm(h_sym) + 1e-30

        # continuous defect: each known zero mode must be annihilated by H
        residuals = {
            lab: float(np.linalg.norm(h_sym @ v) / h_fro)
            for lab, v in zip(labels, modes, strict=True)
        }
        continuous_defect = max(residuals.values()) if residuals else 0.0

        eigvals = np.linalg.eigvalsh(h_sym)
        lam_max = float(np.max(np.abs(eigvals))) + 1e-30
        tol = tol_rel * lam_max
        n_near_zero = int(np.sum(np.abs(eigvals) < tol))
        n_negative = int(np.sum(eigvals < -tol))
        mode_count_error = abs(n_near_zero - expected)
        # eigenvalue gap above the zero block (reproducibility of the count)
        sorted_abs = np.sort(np.abs(eigvals))
        gap = (
            float(sorted_abs[expected] - sorted_abs[expected - 1])
            if len(sorted_abs) > expected
            else 0.0
        )

        cont_score = score_from_defect(continuous_defect, scale)
        score = cont_score * np.exp(-(mode_count_error + n_negative))
        gate = continuous_defect < 1e-3 and mode_count_error == 0 and n_negative == 0

        return self._result(
            raw_defect=continuous_defect,
            score=float(score),
            n_model_calls=budget.used,
            gate=gate,
            details={
                "periodic": periodic,
                "expected_modes": int(expected),
                "n_near_zero": n_near_zero,
                "mode_count_error": int(mode_count_error),
                "n_negative_modes": n_negative,
                "continuous_defect": continuous_defect,
                "per_mode_residual": residuals,
                "raw_asr_magnitude": continuous_defect * float(h_fro),
                "lowest_eigenvalues": np.sort(eigvals)[: expected + 3].tolist(),
                "eigenvalue_gap": gap,
                "relax_fmax_reached": fmax_reached,
                "eps": eps,
                "tol_rel": tol_rel,
                "scale": scale,
            },
        )
