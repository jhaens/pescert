"""Default in-domain substrate builders (spec section 6, "System choice").

The *identity* each proxy checks is universal; only the test *substrate* is supplied
per model.  Users should pass their own :class:`ase.Atoms`; these helpers build small,
physically reasonable defaults from element symbol(s) so universal foundation models
and narrow fine-tuned models are tested on equal footing.

Geometry is element-agnostic: nearest-neighbour distances come from covalent radii,
never from hardcoded chemistry.  Substrates are intentionally small (clusters
``<= ~20`` atoms, trimers, small crystals) to keep every proxy cheap.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

__all__ = ["cluster", "trimer", "bulk_crystal", "make_substrate"]


def _nn_distance(element: str, scale: float = 1.0) -> float:
    z = atomic_numbers[element]
    r = covalent_radii[z]
    if not np.isfinite(r) or r <= 0:
        r = 0.75
    return 2.0 * r * scale


def cluster(element: str = "Ar", n_atoms: int = 13, *, scale: float = 1.0) -> Atoms:
    """Build a compact cluster of ``n_atoms`` of one element.

    Nearest-neighbour spacing is ``2 * covalent_radius * scale``.  At the magic number
    13 a (Mackay) icosahedron is used -- the generic compact-cluster minimum (e.g. the
    LJ13 global minimum) -- so the default substrate relaxes to a genuine minimum, not a
    saddle.  Otherwise the ``n_atoms`` FCC sites nearest the lattice centre are taken.
    Either way the result is compact and non-linear, with well-defined translational and
    rotational zero modes.
    """
    d_nn = _nn_distance(element, scale)
    if n_atoms == 13:
        from ase.cluster import Icosahedron

        # pass an explicit lattice constant so it works for any element (ASE cannot guess
        # one for e.g. diamond-structure elements); the rescale below sets the real scale.
        ico = Icosahedron(element, noshells=2, latticeconstant=d_nn * np.sqrt(2.0))
        p = ico.get_positions()
        nn = min(
            np.linalg.norm(p[a] - p[b]) for a in range(len(p)) for b in range(a + 1, len(p))
        )
        ico.set_positions((p - p.mean(0)) * (d_nn / nn))
        ico.pbc = False
        return ico
    a_fcc = d_nn * np.sqrt(2.0)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pts = []
    rng = range(-3, 4)
    for i in rng:
        for j in rng:
            for k in rng:
                for b in basis:
                    pts.append((np.array([i, j, k]) + b) * a_fcc)
    pts = np.array(pts)
    # center on the site nearest the centroid, take the n closest sites
    center = pts[np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1))]
    order = np.argsort(np.linalg.norm(pts - center, axis=1))
    chosen = pts[order[:n_atoms]]
    chosen = chosen - chosen.mean(0)
    return Atoms(f"{element}{n_atoms}", positions=chosen, pbc=False)


def trimer(
    elements: str | tuple[str, str, str] = "O",
    *,
    d_bond: float | None = None,
    d_base: float | None = None,
    scale: float = 1.0,
) -> Atoms:
    """Build a symmetric A-B-C trimer with A and C the *same* species.

    The mirror plane ``x = 0`` swaps A (index 0) and C (index 2) and fixes B (index 1):
    exactly the reflection+permutation symmetry NEW-3(i) exploits.  ``elements`` may be
    a single symbol (A=B=C) or a 3-tuple ``(A, B, C)`` with ``A == C``.
    """
    if isinstance(elements, str):
        a = b = c = elements
    else:
        a, b, c = elements
        if a != c:
            raise ValueError("trimer requires A and C to be the same species (A == C)")
    if d_bond is None:
        d_bond = _nn_distance(b, scale)
    if d_base is None:
        d_base = d_bond  # equilateral by default
    half = d_base / 2.0
    height = float(np.sqrt(max(d_bond**2 - half**2, 1e-6)))
    positions = np.array(
        [
            [-half, 0.0, 0.0],  # A (index 0)
            [0.0, height, 0.0],  # B (index 1), apex, on the mirror plane
            [half, 0.0, 0.0],  # C (index 2)
        ]
    )
    positions -= positions.mean(0)
    atoms = Atoms([a, b, c], positions=positions, pbc=False)
    atoms.info["mirror_swap"] = (0, 2)  # which atoms the x-mirror exchanges
    return atoms


def bulk_crystal(element: str = "Ar", *, scale: float = 1.0, repeat: int = 1) -> Atoms:
    """Build a small periodic crystal for stress / Gamma-point checks.

    Tries :func:`ase.build.bulk`; falls back to an FCC cell sized from covalent radii
    for elements without an ASE reference structure.
    """
    from ase.build import bulk as ase_bulk

    try:
        atoms = ase_bulk(element)
    except Exception:  # noqa: BLE001 - element has no ASE reference state
        a_fcc = _nn_distance(element, scale) * np.sqrt(2.0)
        atoms = ase_bulk(element, crystalstructure="fcc", a=a_fcc)
    if repeat > 1:
        atoms = atoms.repeat(repeat)
    return atoms


def make_substrate(spec, kind: str) -> Atoms:
    """Resolve a substrate ``spec`` for a given ``kind`` ("cluster"/"trimer"/"bulk").

    ``spec`` may be an :class:`ase.Atoms` (returned as-is), an element symbol, or a
    tuple of element symbols.  Used by :class:`~pescert.suite.Suite` to build the
    right default substrate per proxy.
    """
    if isinstance(spec, Atoms):
        return spec
    if kind == "trimer":
        return trimer(spec)
    if kind == "bulk":
        element = spec if isinstance(spec, str) else spec[0]
        return bulk_crystal(element)
    element = spec if isinstance(spec, str) else spec[0]
    return cluster(element)
