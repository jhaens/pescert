"""OOB-1 -- Maxwell-Betti reciprocity (structural mechanics, 1870s).

Spec: section 3, OOB-1.

Identity.  The reciprocal theorem: the force induced at DOF ``(j, beta)`` by a unit
displacement at ``(i, alpha)`` equals the force induced at ``(i, alpha)`` by a unit
displacement at ``(j, beta)``.  This *is* symmetry of the Hessian / compliance, but
operationalised as a transparent finite-displacement response experiment.

Target: exactly **0** (cross-response asymmetry).  Near-0 by construction for
autodiff-conservative models (a correctness gate); discriminating for the
unconstrained / direct-force class.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from .base import Budget, Eval


@register("betti")
class MaxwellBetti(Eval):
    target = 0.0
    substrate_kind = "cluster"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        n_pairs: int = 12,
        delta: float = 1e-3,
        scale: float = 0.02,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        ndof = 3 * len(atoms)
        rng = np.random.default_rng(seed)

        while budget.would_exceed(4 * n_pairs) and n_pairs > 2:
            n_pairs -= 1

        r0 = atoms.get_positions()
        asyms = []
        responses = []
        for _ in range(n_pairs):
            ia = int(rng.integers(ndof))
            jb = int(rng.integers(ndof))
            while jb == ia:
                jb = int(rng.integers(ndof))
            # response of jb to a push at ia  == H_{jb, ia}
            r1 = self._response(engine, atoms, r0, push=ia, read=jb, delta=delta)
            # response of ia to a push at jb  == H_{ia, jb}
            r2 = self._response(engine, atoms, r0, push=jb, read=ia, delta=delta)
            asyms.append(abs(r1 - r2))
            responses.append((ia, jb, r1, r2))

        defect = float(np.mean(asyms))
        score = score_from_defect(defect, scale)
        gate = defect < 5 * scale

        return self._result(
            raw_defect=defect,
            score=score,
            n_model_calls=budget.used,
            gate=gate,
            details={
                "n_pairs": int(n_pairs),
                "seed": seed,
                "delta": delta,
                "scale": scale,
                "max_asymmetry": float(np.max(asyms)),
                "pairs": [(int(a), int(b), float(x), float(y)) for a, b, x, y in responses],
            },
        )

    @staticmethod
    def _response(engine, atoms, r0, *, push: int, read: int, delta: float) -> float:
        """Induced force at DOF ``read`` from a +/- push at DOF ``push`` (== H_{read,push})."""
        n = len(atoms)
        plus = atoms.copy()
        disp = np.zeros(3 * n)
        disp[push] = delta
        plus.set_positions(r0 + disp.reshape(n, 3))
        minus = atoms.copy()
        minus.set_positions(r0 - disp.reshape(n, 3))
        f_plus = engine.forces(plus).reshape(-1)[read]
        f_minus = engine.forces(minus).reshape(-1)[read]
        return -(f_plus - f_minus) / (2 * delta)
