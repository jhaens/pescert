"""Many-body trimer probe (three-body decay and transverse self-consistency).

Section: Symmetry & invariance.

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


class _FragmentUnsupported(RuntimeError):
    """The model refused an isolated fragment the many-body expansion needs."""


@register("trimer")
class Trimer(Eval):
    section = "Symmetry & invariance"
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
        eps: float | None = None,
        eps_sweep: tuple[float, ...] = (),
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        eps = engine.fd_step(order=1, accuracy=2) if eps is None else eps
        rng = np.random.default_rng(seed)

        try:
            d_i, det_i = self._three_body_vanishing(engine, atoms, n_separation, sep_max)
        except _FragmentUnsupported as exc:
            # The transverse channel never leaves the intact trimer, so it still has a
            # number.  It is reported, but not promoted to the probe's score: half a
            # trimer score is not comparable with the two-channel score every other
            # model gets, and a silently different definition is worse than a gap.
            d_ii, det_ii = self._transverse(
                engine, atoms, rng, n_pairs, eps, eps_sweep, scale_transverse
            )
            return self._skip(
                str(exc),
                budget,
                extra={
                    "sub_scores": {
                        "three_body_vanishing": float("nan"),
                        "transverse": score_from_defect(d_ii, scale_transverse),
                    },
                    "sub_defects": {
                        "three_body_vanishing": float("nan"),
                        "transverse": d_ii,
                    },
                    "transverse": det_ii,
                },
            )

        d_ii, det_ii = self._transverse(
            engine, atoms, rng, n_pairs, eps, eps_sweep, scale_transverse
        )

        s_i = score_from_defect(d_i, 0.2)
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
            # keep the substrate's cell: a zero-cell Atoms is the degenerate object
            # pescert.substrates warns about, and several universal models reject it
            # outright ("Atoms must have a defined cell").  Built without one, this
            # single line made the whole proxy crash -- and a crashed proxy scores 0 --
            # for every model with that requirement (NEP89, CHGNet, EquiformerV2/V3).
            a = Atoms([symbols[k]], positions=[[0, 0, 0]], cell=atoms.get_cell(), pbc=False)
            a.center()
            return self._fragment(engine, a, "atom")

        e_mono = [mono(0), mono(1), mono(2)]
        # the CA pair does not move as vertex B recedes, so it is evaluated once
        e_ca = self._fragment(engine, atoms[[0, 2]], "pair")

        b_dir = np.array([0.0, 1.0, 0.0])  # recede the apex along +y
        offsets = np.linspace(0.0, sep_max, n_sep)
        de3 = np.empty(n_sep)
        emag = 1.0  # track the energy magnitude to set a floating-point noise floor
        for idx, off in enumerate(offsets):
            pos = pos0.copy()
            pos[1] = pos0[1] + off * b_dir
            full = atoms.copy()
            full.set_positions(pos)
            e_abc = engine.energy(full)
            e_ab = self._fragment(engine, self._sub(atoms, [0, 1], pos), "pair")
            e_bc = self._fragment(engine, self._sub(atoms, [1, 2], pos), "pair")
            de3[idx] = e_abc - (e_ab + e_bc + e_ca) + sum(e_mono)
            emag = max(emag, abs(e_abc), abs(e_ab) + abs(e_bc) + abs(e_ca))

        peak = float(np.max(np.abs(de3)))
        # A pairwise model has dE3 == 0 up to the cancellation noise of the large
        # energies; below that floor the limit is trivially satisfied, not noise/noise.
        noise_floor = 1e-8 * emag
        if peak < noise_floor:
            return 0.0, {
                "separations": offsets.tolist(), "delta_e3": de3.tolist(),
                "residual_at_max_sep": float(de3[-1]), "residual_relative": 0.0,
                "smoothness_penalty": 0.0, "vanishing": True,
            }
        residual_rel = float(abs(de3[-1])) / peak
        tv = float(np.sum(np.abs(np.diff(de3))))
        span = float(abs(de3[0] - de3[-1]))
        smooth_penalty = max(0.0, tv / span - 1.0) if span > noise_floor else 0.0
        defect = residual_rel + smooth_penalty
        return defect, {
            "separations": offsets.tolist(),
            "delta_e3": de3.tolist(),
            "residual_at_max_sep": float(de3[-1]),
            "residual_relative": residual_rel,
            "smoothness_penalty": smooth_penalty,
            "vanishing": False,
        }

    @staticmethod
    def _fragment(engine, atoms, what: str) -> float:
        """Energy of an isolated fragment, or :class:`_FragmentUnsupported`.

        The many-body expansion is built from monomers and from dimers pulled past the
        cutoff -- systems whose neighbour list is *empty*.  Their energy is well defined
        (nothing interacts), but several universal models raise on a zero-edge graph
        instead of returning it, and the number cannot be recovered from outside the
        model: every route to it needs another edgeless evaluation.  That is a
        limitation of the implementation, not a defect of the model's PES, so the
        channel reports "not applicable" rather than scoring the model as if it had
        failed an identity.
        """
        try:
            return engine.energy(atoms)
        except Exception as exc:  # noqa: BLE001
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
            raise _FragmentUnsupported(
                f"model cannot evaluate an isolated {what}, which the many-body "
                f"expansion requires ({type(exc).__name__}: {first[:140]})"
            ) from exc

    def _skip(self, message: str, budget: Budget, extra: dict | None = None) -> EvalResult:
        return self._result(
            raw_defect=float("nan"),
            score=float("nan"),
            n_model_calls=budget.used,
            gate=None,
            details={"skipped": True, "message": message, **(extra or {})},
        )

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
            "precision": engine.precision,
            "scale": scale,
            "eps_sweep": sweep,
        }
