"""Strain--position reciprocity (the mixed cross-Maxwell / internal-strain block).

Section: Self-consistency.

Identity.  For a periodic system a conservative model is a single scalar ``E(R, eps)``
on positions and cell strain, so Schwarz's theorem constrains the mixed second-derivative
block connecting the two sectors,

    V d sigma_mn / d R_ja = d^2 E / d R_ja d eps_mn = - d F_ja / d eps_mn ,

the internal-strain tensor ``Lambda`` of lattice dynamics.  A model whose force and
stress heads disagree on it couples cell and internal degrees of freedom inconsistently.

For ``N_p`` random pairs of a Cartesian DOF and a Voigt component we central-difference
both sides -- the stress under a displacement, the force under a strain, using the
*identical* deformation convention as the stress--gradient probe (shared :mod:`._strain`)
-- and average the relative residual into ``d_Lambda``.  At marginal extra cost the
strain--strain block ``B_mn = d sigma_m / d eps_n`` (12 calls) tests the stress head's own
integrability through its Voigt symmetry, ``d_B = ||B - B^T||_F / ||B||_F``.

Target: exactly **0**.  A gate for models whose stress is the exact strain derivative of
the predicted energy; discriminating for direct or separately trained stress heads.
Skips when the model exposes no stress.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._strain import apply_strain, strain_tensor
from .base import Budget, Eval


@register("cross_maxwell")
class CrossMaxwell(Eval):
    section = "Self-consistency"
    target = 0.0
    substrate_kind = "p2mm"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        n_pairs: int = 6,
        delta: float | None = None,
        floor: float = 1e-3,
        lam: float = 0.05,
        do_relax: bool = True,
        relax_fmax: float = 1e-2,
        relax_steps: int = 200,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        delta = engine.fd_step(order=1, accuracy=2) if delta is None else delta
        if not np.any(atoms.pbc):
            return self._skip("substrate is not periodic; stress is undefined", budget)
        if not engine.has_stress(atoms):
            return self._skip("model does not expose a stress", budget)

        # On a prestressed cell the strain--strain block picks up a spurious asymmetric
        # d(1/V)/deps term, so d_B is only clean near mechanical equilibrium.
        work_atoms = self._relax_cell(engine, atoms, relax_fmax, relax_steps) if do_relax else atoms

        ndof = 3 * len(work_atoms)
        cell0 = np.array(work_atoms.get_cell())
        volume = work_atoms.get_volume()
        rng = np.random.default_rng(seed)
        atoms = work_atoms

        # -- cross block Lambda: V dsigma_m/dR_j  vs  -dF_j/deps_m --------------------
        residuals = []
        pair_detail = []
        for _ in range(n_pairs):
            j = int(rng.integers(ndof))
            m = int(rng.integers(6))
            lhs = volume * self._d_stress_d_pos(engine, atoms, j, m, delta)
            rhs = self._d_force_d_strain(engine, atoms, cell0, j, m, delta)
            num = abs(lhs + rhs)
            den = max(abs(lhs), abs(rhs), floor)
            residuals.append(num / den)
            pair_detail.append((j, m, float(lhs), float(rhs)))
        d_lambda = float(np.mean(residuals)) if residuals else 0.0

        # -- strain--strain block B_mn = dsigma_m/deps_n; check Voigt symmetry ----------
        b_mat = np.empty((6, 6))
        for nidx in range(6):
            voigt = np.zeros(6)
            voigt[nidx] = delta
            sp = engine.stress_voigt(apply_strain(atoms, cell0, strain_tensor(voigt)))
            sm = engine.stress_voigt(apply_strain(atoms, cell0, strain_tensor(-voigt)))
            b_mat[:, nidx] = (sp - sm) / (2 * delta)
        d_b = float(np.linalg.norm(b_mat - b_mat.T) / (np.linalg.norm(b_mat) + 1e-12))

        raw_defect = 0.5 * (d_lambda + d_b)
        score = score_from_defect(raw_defect, lam)  # exp(-1/2 (d_L + d_B)/lambda)
        gate = (d_lambda < 5 * lam) and (d_b < 5 * lam)

        return self._result(
            raw_defect=raw_defect,
            score=score,
            n_model_calls=budget.used,
            gate=gate,
            details={
                "n_pairs": int(n_pairs),
                "delta": delta,
                "precision": engine.precision,
                "floor": floor,
                "lambda": lam,
                "relaxed": bool(do_relax),
                "d_Lambda": d_lambda,
                "d_B": d_b,
                "strain_block_B": b_mat.tolist(),
                "pairs": [(int(a), int(b), x, y) for a, b, x, y in pair_detail],
            },
        )

    @staticmethod
    def _relax_cell(engine, atoms, fmax: float, steps: int):
        """Relax positions and cell to near-zero stress (model's own equilibrium)."""
        from ase.optimize import BFGS

        try:
            from ase.filters import FrechetCellFilter as CellFilter
        except ImportError:  # older ASE
            from ase.constraints import ExpCellFilter as CellFilter

        work = engine.attach(atoms)
        opt = BFGS(CellFilter(work), logfile=None)
        opt.run(fmax=fmax, steps=steps)
        out = atoms.copy()
        out.set_cell(work.get_cell(), scale_atoms=False)
        out.set_positions(work.get_positions())
        return out

    @staticmethod
    def _d_stress_d_pos(engine, atoms, j: int, m: int, delta: float) -> float:
        n = len(atoms)
        r0 = atoms.get_positions()
        disp = np.zeros(3 * n)
        disp[j] = delta
        plus = atoms.copy()
        plus.set_positions(r0 + disp.reshape(n, 3))
        minus = atoms.copy()
        minus.set_positions(r0 - disp.reshape(n, 3))
        return (engine.stress_voigt(plus)[m] - engine.stress_voigt(minus)[m]) / (2 * delta)

    @staticmethod
    def _d_force_d_strain(engine, atoms, cell0, j: int, m: int, delta: float) -> float:
        voigt = np.zeros(6)
        voigt[m] = delta
        fp = engine.forces(apply_strain(atoms, cell0, strain_tensor(voigt))).reshape(-1)[j]
        fm = engine.forces(apply_strain(atoms, cell0, strain_tensor(-voigt))).reshape(-1)[j]
        return (fp - fm) / (2 * delta)

    def _skip(self, message: str, budget: Budget) -> EvalResult:
        return self._result(
            raw_defect=float("nan"),
            score=float("nan"),
            n_model_calls=budget.used,
            gate=None,
            details={"skipped": True, "message": message},
        )
