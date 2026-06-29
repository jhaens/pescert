"""KNOWN -- PES smoothness along a bond deformation (BSCT-style).

Spec: section 1 (prior art); BSCT bond deformation (arXiv:2602.04861).

Identity.  A physical PES is a smooth single well along a bond stretch: it has no
energy discontinuities, its projected force is monotone through a single zero
crossing, it has one minimum, and the force-vs-displacement curve has low total
variation ("tortuosity").  We scan one bond from far-compressed to far-stretched and
combine four diagnostics -- localized energy jump, excess force-flips, spurious
minima, and force tortuosity -- into a single score.

Target: a smooth single well -> defect 0 -> **score 1**.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from ._geometry import closest_pair
from .base import Budget, Eval


def _count_sign_changes(x: np.ndarray, tol: float) -> int:
    s = np.sign(x)
    s[np.abs(x) < tol] = 0
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def _count_local_minima(y: np.ndarray) -> int:
    interior = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
    return int(np.sum(interior))


@register("smoothness")
class Smoothness(Eval):
    target = 0.0  # defect target; a perfectly smooth single well scores 1

    substrate_kind = "cluster"

    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        bond: tuple[int, int] | None = None,
        compress_frac: float = 0.2,
        stretch: float = 0.5,
        n_points: int = 81,
        scale: float = 1.0,
        jolt_baseline: float = 4.0,
        jolt_weight: float = 0.25,
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        i, j = bond if bond is not None else closest_pair(atoms)
        r0 = atoms.get_positions()
        bond_vec = r0[j] - r0[i]
        bond_len = float(np.linalg.norm(bond_vec))
        u = bond_vec / bond_len

        while budget.would_exceed(n_points) and n_points > 11:
            n_points -= 2
        # asymmetric scan: cap compression at compress_frac * bond_len so we probe the
        # physical Pauli wall and the dissociation tail without diving into the
        # numerically-vertical (r << r0) part of a steep repulsion.
        s_grid = np.linspace(-compress_frac * bond_len, stretch, n_points)

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

        f_span = float(fproj.max() - fproj.min()) + 1e-12

        # (1) force-jolt ratio: a localized second difference of the projected force,
        # normalized by the typical step.  A smooth (even steep) PES gives O(1); a kink
        # or step in the force gives a large outlier.  This is the discontinuity / energy-
        # jump diagnostic, measured on the force (the derivative of E) for scale-freeness.
        df = np.diff(fproj)
        if df.size >= 3:
            jolt = df[1:-1] - 0.5 * (df[:-2] + df[2:])
            jolt_ratio = float(np.max(np.abs(jolt))) / (float(np.median(np.abs(df))) + 1e-9)
        else:
            jolt_ratio = 0.0
        jolt_excess = max(0.0, jolt_ratio - jolt_baseline)

        # (2) force flips beyond the single crossing of a single well
        tol = 1e-6 * (abs(fproj).max() + 1e-12)
        flips = _count_sign_changes(fproj, tol)
        flips_excess = max(0, flips - 1)

        # (3) spurious minima beyond the single well
        n_min = _count_local_minima(energies)
        spurious_minima = max(0, n_min - 1)

        # (4) force tortuosity: total variation normalized by force range (0 for monotone)
        tv = float(np.sum(np.abs(df)))
        tortuosity = max(0.0, tv / f_span - 1.0)

        defect = (
            jolt_weight * jolt_excess
            + 1.0 * flips_excess
            + 1.0 * spurious_minima
            + 1.0 * tortuosity
        )
        score = score_from_defect(defect, scale)

        return self._result(
            raw_defect=defect,
            score=score,
            n_model_calls=budget.used,
            gate=None,
            details={
                "bond": (int(i), int(j)),
                "bond_len": bond_len,
                "compress_frac": compress_frac,
                "stretch": stretch,
                "n_points": int(n_points),
                "scale": scale,
                "force_jolt_ratio": jolt_ratio,
                "force_jolt_excess": jolt_excess,
                "force_flips": int(flips),
                "force_flips_excess": int(flips_excess),
                "n_minima": int(n_min),
                "spurious_minima": int(spurious_minima),
                "tortuosity": tortuosity,
                "s_grid": s_grid.tolist(),
                "energies": energies.tolist(),
                "projected_force": fproj.tolist(),
            },
        )
