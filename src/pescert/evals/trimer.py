"""NEW-3 -- many-body & transverse self-consistency (the trimer probe).

Spec: section 2, NEW-3.

Two complementary checks on a symmetric trimer A-B-C (A and C the same species):

(i) **Three-body-term vanishing** (target 0 *limit*): the many-body-expansion term
     ``dE3 = E(ABC) - [E(AB)+E(BC)+E(CA)] + [E(A)+E(B)+E(C)]`` is *supposed* to be
     nonzero at contact (real physics), so we never score its value -- only that it
     decays smoothly and monotonically to 0 as a vertex is removed to infinity.

(ii) **Transverse conservativeness** (target 0): for random unit vectors ``u, w``,
      ``<w, J u> - <u, J w> = <w, (J - J^T) u>`` must vanish.  This is the cheap,
      matrix-free realisation of the Jacobian-symmetry probe the Ceriotti group called
      "too expensive", and it sees off-diagonal/angular non-conservativeness invisible
      to collinear dimer stretches.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from .base import Budget, Eval


@register("trimer")
class Trimer(Eval):
    target = 0.0
    substrate_kind = "trimer"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        scale_transverse: float = 0.05,
        n_separation: int = 16,
        sep_max: float = 12.0,
        n_pairs: int = 16,
        eps: float = 1e-3,
        eps_sweep: tuple[float, ...] = (),
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        rng = np.random.default_rng(seed)

        d_i, det_i = self._three_body_vanishing(engine, atoms, n_separation, sep_max)
        d_ii, det_ii = self._transverse(engine, atoms, rng, n_pairs, eps, eps_sweep, scale_transverse)

        s_i = score_from_defect(d_i, 0.1)
        s_ii = score_from_defect(d_ii, scale_transverse)
        score = float((s_i * s_ii) ** (1.0 / 2.0))
        raw_defect = float(d_i + d_ii / scale_transverse)
        gate =  (d_ii < 5 * scale_transverse)

        details = {
            "sub_scores": {"three_body_vanishing": s_i, "transverse": s_ii},
            "sub_defects": {"three_body_vanishing": d_i, "transverse": d_ii},
            "three_body_vanishing": det_i,
            "transverse": det_ii,
        }
        return self._result(
            raw_defect=raw_defect,
            score=score,
            n_model_calls=budget.used,
            gate=gate,
            details=details,
        )

    # -- (i) -----------------------------------------------------------------
    def _three_body_vanishing(self, engine, atoms, n_sep, sep_max):
        symbols = atoms.get_chemical_symbols()
        pos0 = atoms.get_positions()

        def mono(k):
            a = Atoms([symbols[k]], positions=[[0, 0, 0]], pbc=False)
            return engine.energy(a)

        e_mono = [mono(0), mono(1), mono(2)]
        # CA pair (atoms 0,2) does not move when vertex B (index 1) recedes
        e_ca = engine.energy(atoms[[0, 2]])

        b_dir = np.array([0.0, 1.0, 0.0])  # recede the apex along +y
        offsets = np.linspace(0.0, sep_max, n_sep)
        de3 = np.empty(n_sep)
        for idx, off in enumerate(offsets):
            pos = pos0.copy()
            pos[1] = pos0[1] + off * b_dir
            full = atoms.copy()
            full.set_positions(pos)
            e_abc = engine.energy(full)
            e_ab = engine.energy(self._sub(atoms, [0, 1], pos))
            e_bc = engine.energy(self._sub(atoms, [1, 2], pos))
            de3[idx] = e_abc - (e_ab + e_bc + e_ca) + sum(e_mono)

        scale = float(np.max(np.abs(de3))) + 1e-12
        residual_rel = float(abs(de3[-1])) / scale
        tv = float(np.sum(np.abs(np.diff(de3))))
        span = float(abs(de3[0] - de3[-1]))
        smooth_penalty = max(0.0, tv / (span + 1e-12) - 1.0) if span > 1e-12 else 0.0
        defect = residual_rel + smooth_penalty
        return defect, {
            "separations": offsets.tolist(),
            "delta_e3": de3.tolist(),
            "residual_at_max_sep": float(de3[-1]),
            "residual_relative": residual_rel,
            "smoothness_penalty": smooth_penalty,
        }

    @staticmethod
    def _sub(atoms, idx, positions):
        sub = atoms[idx]
        sub.set_positions(positions[idx])
        return sub

    # -- (ii) ---------------------------------------------------------------
    def _transverse(self, engine, atoms, rng, n_pairs, eps, eps_sweep, scale):
        ndof = 3 * len(atoms)

        def mean_antisym(step):
            vals = []
            for _ in range(n_pairs):
                u = rng.normal(size=ndof)
                u /= np.linalg.norm(u)
                w = rng.normal(size=ndof)
                w /= np.linalg.norm(w)
                ju = engine.jvp(atoms, u, eps=step)
                jw = engine.jvp(atoms, w, eps=step)
                vals.append(abs(float(w @ ju) - float(u @ jw)))
            return float(np.mean(vals))

        defect = mean_antisym(eps)
        sweep = {}
        for s in eps_sweep:
            sweep[f"{s:g}"] = mean_antisym(s)
        return defect, {
            "n_pairs": int(n_pairs),
            "eps": eps,
            "mean_abs_antisymmetry": defect,
            "scale": scale,
            "eps_sweep": sweep,
        }
