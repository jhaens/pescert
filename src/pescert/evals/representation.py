"""Extensivity and representation invariance.

Section: Symmetry & invariance.

A periodic calculation refers to a crystal, not to the bookkeeping chosen to describe it.
Four exact invariances are tested on one periodic substrate:

(i)   **Lattice-basis gauge**: for a unimodular integer ``M`` the basis ``A' = M A`` with
      atoms wrapped describes the identical crystal (we use the shear ``a1' = a1 + a2``).
(ii)  **Extensivity**: an ``n``-fold supercell returns ``n`` times the energy and
      replicated forces.
(iii) **Translation (egg-box)**: the energy is invariant under a rigid translation modulo
      the cell; a dependence is the egg-box effect of grid-based codes.
(iv)  **Permutation**: relabelling atoms leaves the energy invariant and permutes forces.

Each comparison yields a per-atom energy residual and an RMS force residual, normalised
and summed into ``d_rep``.  Strictly local models sit at machine precision (a gate); it
becomes discriminating for global pooling, cell-conditioned features, or grid-based
long-range modules.

Target: exactly **0** (summed normalised residual).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..engine import ModelEngine
from ..registry import register
from ..result import EvalResult, score_from_defect
from .base import Budget, Eval

_E_SCALE = 1e-3  # eV, per-atom energy normaliser
_F_SCALE = 1e-3  # eV/A, force normaliser


def _match_by_fraction(ref_cell: np.ndarray, ref_pos: np.ndarray, new_pos: np.ndarray):
    """Map each atom of ``new_pos`` to the ``ref_pos`` atom with the same wrapped
    fractional coordinate in ``ref_cell`` (minimum image), returning an index array."""
    inv = np.linalg.inv(np.asarray(ref_cell))
    ref_frac = (ref_pos @ inv) % 1.0
    new_frac = (new_pos @ inv) % 1.0
    idx = np.empty(len(new_frac), dtype=int)
    for a in range(len(new_frac)):
        d = new_frac[a] - ref_frac
        d -= np.round(d)  # minimum image in fractional space
        idx[a] = int(np.argmin((d**2).sum(1)))
    return idx


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


@register("representation")
class Representation(Eval):
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
        n_translations: int = 16,
        supercell: tuple[int, int, int] = (2, 2, 2),
        **cfg,
    ) -> EvalResult:
        budget = Budget(engine, max_calls)
        if not np.any(atoms.pbc):
            return self._skip("substrate is not periodic", budget)

        n = len(atoms)
        cell0 = np.array(atoms.get_cell())
        e0, f0 = engine.energy_forces(atoms)  # 1 call
        e0_atom = e0 / n
        rng = np.random.default_rng(seed)

        comparisons: dict[str, tuple[float, float]] = {}

        # (i) unimodular gauge: a1' = a1 + a2 (det M = 1), atoms wrapped
        m_uni = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        sheared = atoms.copy()
        sheared.set_cell(m_uni @ cell0, scale_atoms=False)
        sheared.wrap()
        e_s, f_s = engine.energy_forces(sheared)
        idx = _match_by_fraction(cell0, atoms.get_positions(), sheared.get_positions())
        comparisons["unimodular"] = (
            abs(e_s / n - e0_atom),
            _rms(f_s - f0[idx]),
        )

        # (ii) extensivity: n-fold supercell, per-atom energy and replicated forces
        rep_factor = int(np.prod(supercell))
        rep = atoms.repeat(supercell)
        e_r, f_r = engine.energy_forces(rep)
        idx_r = _match_by_fraction(cell0, atoms.get_positions(), rep.get_positions())
        comparisons["supercell"] = (
            abs(e_r / rep_factor / n - e0_atom),
            _rms(f_r - f0[idx_r]),
        )

        # (iii) translation / egg-box: sweep rigid shifts along a random direction over one
        # lattice period; energy and forces are exactly invariant for a periodic crystal.
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        disp_full = u @ cell0  # Cartesian shift for +1 in fractional along u
        while budget.would_exceed(n_translations + rep_factor) and n_translations > 4:
            n_translations -= 2
        energies = np.empty(n_translations)
        f_shift_max = 0.0
        for k, s in enumerate(np.linspace(0.0, 1.0, n_translations)):
            shifted = atoms.copy()
            shifted.set_positions(atoms.get_positions() + s * disp_full)
            e_t, f_t = engine.energy_forces(shifted)
            energies[k] = e_t / n
            f_shift_max = max(f_shift_max, _rms(f_t - f0))
        comparisons["translation"] = (
            float(energies.max() - energies.min()),
            f_shift_max,
        )

        # (iv) permutation: shuffle indices, compare after undoing the permutation
        if n >= 2:
            perm = rng.permutation(n)
            permuted = atoms[perm]
            e_p, f_p = engine.energy_forces(permuted)
            f_unperm = np.empty_like(f0)
            f_unperm[perm] = f_p
            comparisons["permutation"] = (
                abs(e_p / n - e0_atom),
                _rms(f_unperm - f0),
            )

        d_rep = 0.0
        detail: dict[str, dict] = {}
        for name, (de, df) in comparisons.items():
            contrib = de / _E_SCALE + df / _F_SCALE
            d_rep += contrib
            detail[name] = {"d_energy_per_atom": de, "d_force_rms": df, "contribution": contrib}

        score = score_from_defect(d_rep, scale=1.0)  # S_rep = exp(-d_rep)
        gate = d_rep < 1.0

        return self._result(
            raw_defect=d_rep,
            score=score,
            n_model_calls=budget.used,
            gate=gate,
            details={
                "seed": seed,
                "n_atoms": n,
                "supercell": list(supercell),
                "n_translations": int(n_translations),
                "energy_scale_eV": _E_SCALE,
                "force_scale_eV_per_A": _F_SCALE,
                "comparisons": detail,
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
