"""Default in-domain substrate builders.

The *identity* each proxy checks is universal; only the test *substrate* is supplied
per model.  Users should pass their own :class:`ase.Atoms`; these helpers build small,
physically reasonable defaults from element symbol(s) so universal foundation models
and narrow fine-tuned models are tested on equal footing.

Geometry is element-agnostic: nearest-neighbour distances come from covalent radii,
never from hardcoded chemistry.  Substrates are intentionally small (clusters
``<= ~20`` atoms, trimers, small crystals) to keep every proxy cheap.
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

__all__ = ["cluster", "trimer", "bulk_crystal", "p2mm_crystal", "make_substrate"]

#: Vacuum padding, in Angstrom, put around every non-periodic substrate.
#:
#: An isolated cluster needs no cell physically, and with ``pbc=False`` none of it enters
#: the energy.  But a zero cell is a degenerate object: it inverts to a singular matrix
#: and produces empty neighbour lists, and several universal models reject it outright
#: ("Atoms must have a defined cell").  Enclosing the cluster in a box is the standard
#: way to describe it and costs nothing.  The padding is kept modest -- comfortably
#: beyond any model cutoff, but not so large that neighbour-list binning slows down.
VACUUM = 8.0


def _reference_crystal(element: str):
    """The element's ASE reference crystal, or ``None`` when ASE has no buildable one.

    ``ase.build.bulk`` resolves the experimental ground-state structure *and* its
    measured lattice constant from ``ase.data.reference_states``.  It covers 72 elements;
    the rest are molecular solids or structures ASE cannot build (H, B, N, O, F, P, S,
    Cl, Ga, Se, Br, ...), which is what the covalent fallback below is for.
    """
    from ase.build import bulk as ase_bulk

    try:
        return ase_bulk(element)
    except Exception:  # noqa: BLE001 - no reference state, or a symmetry ASE cannot build
        return None


def _covalent_diameter(element: str, scale: float = 1.0) -> float:
    """``2 * covalent_radius * scale``: the single-bond length of one element."""
    z = atomic_numbers[element]
    r = covalent_radii[z]
    if not np.isfinite(r) or r <= 0:
        r = 0.75
    return 2.0 * r * scale


def _nn_distance(element: str, scale: float = 1.0) -> float:
    """Nearest-neighbour distance for this element, from its reference crystal.

    The experimental crystal is the right source because it fixes distance *and*
    coordination together: carbon's 1.55 A is a four-coordinate diamond bond, sodium's
    3.66 A an eight-coordinate bcc contact.  Falls back to the covalent diameter for the
    elements ASE has no reference crystal for, where the covalent radius -- a
    single-bond quantity -- is the consistent choice.
    """
    crystal = _reference_crystal(element)
    if crystal is None:
        return _covalent_diameter(element, scale)
    d = crystal.repeat(3).get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min()) * scale


#: Seeded symmetry-breaking displacement applied to every cluster, in Angstrom.
#:
#: A fragment cut from a crystal inherits the crystal's point group, and a symmetric
#: geometry can be a *stationary* point that is not a minimum -- the 13-site fcc fragment
#: is the cuboctahedron, a saddle of the pair potential whose gradient vanishes by
#: symmetry, so a relaxation started there cannot leave it and the Hessian keeps a
#: negative eigenvalue.  0.05 A is a few percent of any bond length here, small enough to
#: leave coordination and spacing untouched and large enough to leave the saddle: 0.02 A
#: still left the bcc lithium fragment stalled on one (the relaxation converges to
#: fmax = 1e-3 with a negative eigenvalue and more steps do not help, because a shallow
#: saddle is a stationary point).  It is drawn from a fixed seed, so every model is handed
#: the identical geometry.
RATTLE_STDEV = 0.05
RATTLE_SEED = 0


def _compact_fragment(
    lattice: Atoms,
    element: str,
    n_atoms: int,
    scale: float,
    rattle: float = RATTLE_STDEV,
    seed: int = RATTLE_SEED,
) -> Atoms:
    """Cut the ``n_atoms`` sites closest to the centre out of a repeated ``lattice``."""
    reps = 2
    while len(lattice) * reps**3 < 8 * n_atoms:
        reps += 1
    pos = lattice.repeat(reps).get_positions()
    centre = pos[np.argmin(np.linalg.norm(pos - pos.mean(0), axis=1))]
    order = np.argsort(np.linalg.norm(pos - centre, axis=1))
    chosen = pos[order[:n_atoms]]
    chosen = (chosen - chosen.mean(0)) * scale
    out = Atoms(f"{element}{n_atoms}", positions=chosen, pbc=False)
    if rattle:
        out.rattle(stdev=rattle, seed=seed)
    out.center(vacuum=VACUUM)
    return out


def cluster(element: str = "Ar", n_atoms: int = 13, *, scale: float = 1.0) -> Atoms:
    """Build a compact cluster of ``n_atoms`` of one element.

    The cluster is the ``n_atoms`` sites nearest the centre of the element's **own
    experimental reference crystal** (``ase.build.bulk``, i.e. the measured ground-state
    structure at its measured lattice constant), rescaled by ``scale``.  Elements ASE has
    no reference crystal for get a diamond-structure fragment at the covalent diameter
    instead: a four-coordinate network is the structure consistent with a single-bond
    radius.

    Taking the geometry from the crystal rather than imposing one fixes distance and
    coordination *together*, which a generic close-packed cluster cannot do.  A 13-atom
    Mackay icosahedron rescaled to the covalent diameter -- the previous default -- puts
    twelve neighbours at the single-bond length, which is a physical geometry for a metal
    and an extreme compression for a covalent element: at 1.52 A with coordination 12,
    carbon reached +2.7 keV and 2.7 keV/A on a foundation model, i.e. an extrapolation
    regime no potential is fitted for.  Cutting diamond instead puts carbon's four
    neighbours at 1.55 A, sodium's eight at 3.66 A, aluminium's twelve at 2.86 A, each
    the coordination its own chemistry has.

    A seeded ``RATTLE_STDEV`` displacement then breaks the fragment's point symmetry --
    see that constant for why a perfectly symmetric fragment is not usable.  The result is
    compact, non-linear and non-periodic, with the six well-defined translational/
    rotational zero modes the probes assume.
    """
    crystal = _reference_crystal(element)
    if crystal is not None:
        return _compact_fragment(crystal, element, n_atoms, scale)

    # covalent fallback: diamond structure at the single-bond length, so distance and
    # coordination stay mutually consistent
    from ase.build import bulk as ase_bulk

    a_cubic = _covalent_diameter(element, 1.0) * 4.0 / np.sqrt(3.0)
    lattice = ase_bulk(element, crystalstructure="diamond", a=a_cubic)
    return _compact_fragment(lattice, element, n_atoms, scale)


def trimer(
    elements: str | tuple[str, str, str] = "Ar",
    *,
    d_bond: float | None = None,
    d_base: float | None = None,
    scale: float = 1.0,
) -> Atoms:
    """Build a symmetric A-B-C trimer with A, B and C the *same* species.
    """
    if isinstance(elements, str):
        a = b = c = elements
    else:
        raise ValueError("trimer requires A, B and C to be the same species (A == B == C)")
    if d_bond is None:
        d_bond = _nn_distance(b, scale)
    if d_base is None:
        d_base = d_bond  # equilateral by default
    half = d_base / 2.0
    height = float(np.sqrt(max(d_bond**2 - half**2, 1e-6)))
    positions = np.array(
        [
            [-half, 0.0, 0.0],  # A (index 0)
            [0.0, height, 0.0], # B (index 1), apex
            [half, 0.0, 0.0],   # C (index 2)
        ]
    )
    positions -= positions.mean(0)
    atoms = Atoms([a, b, c], positions=positions, pbc=False)
    atoms.center(vacuum=VACUUM)
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
    except Exception:  # Element has no ASE reference state
        a_fcc = _nn_distance(element, scale) * np.sqrt(2.0)
        atoms = ase_bulk(element, crystalstructure="fcc", a=a_fcc)
    if repeat > 1:
        atoms = atoms.repeat(repeat)
    return atoms


# Template for the improper-symmetry (parity) stress probe: four atoms of one element on
# Wyckoff sites (0,0,z), (1/2,0,z), (1/2,1/2,z) of space group Pmm2 (no. 25, point group
# mm2), with three unequal lattice parameters and generic heights.  In mm2 the shear
# sigma_xy is even under the proper C2z (killed only by the improper mirrors m_x, m_y),
# while sigma_xz, sigma_yz are odd under C2z (killed by any SE(3) model) -- so sigma_xy is
# a pure improper-symmetry residual and the other two shears are built-in controls.
_P2MM_CELL = np.array(
    [[4.32890106, 0.0, 0.0], [0.0, 4.46763875, 0.0], [0.0, 0.0, 7.40896034]]
)
_P2MM_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.8122234288060413],
        [2.16445053, 2.233819375, 4.24693875927123],
        [2.16445053, 0.0, 5.641866990811415],
        [2.16445053, 2.233819375, 1.7472833451210954],
    ]
)


def p2mm_crystal(element: str = "Si", *, scale: float = 1.0) -> Atoms:
    """Build the four-atom ``Pmm2`` (no. 25) crystal for the improper-symmetry probe.

    All four atoms are ``element``.  As with the other builders the geometry is
    element-agnostic: the fixed template cell is uniformly rescaled so the
    nearest-neighbour distance matches the covalent diameter ``2 * covalent_radius *
    scale``, keeping every model tested at a physically reasonable bond length.
    """
    atoms = Atoms(
        f"{element}4",
        positions=_P2MM_POSITIONS.copy(),
        cell=_P2MM_CELL.copy(),
        pbc=True,
    )
    dists = atoms.get_all_distances(mic=True)
    iu = np.triu_indices(len(atoms), 1)
    d_nn_template = float(dists[iu].min())
    target = _nn_distance(element, scale)
    atoms.set_cell(np.array(atoms.get_cell()) * (target / d_nn_template), scale_atoms=True)
    return atoms


def _is_element(spec) -> bool:
    return isinstance(spec, str) and atomic_numbers.get(spec, 0) > 0


def make_substrate(spec, kind: str) -> Atoms:
    """Resolve a substrate ``spec`` for a given ``kind`` ("cluster"/"trimer"/"bulk"/"p2mm").

    ``spec`` may be an :class:`ase.Atoms` (returned as-is), an element symbol, a tuple of
    element symbols, or the path of a structure file ASE can read (returned as read).  A
    symbol always means the element, even next to a file of that name, and a file that
    cannot be parsed raises ASE's own error.  Used by :class:`~pescert.suite.Suite` to
    build the right default substrate per proxy.
    """
    if isinstance(spec, Atoms):
        return spec
    if isinstance(spec, (list, tuple)) and len(spec) == 1:
        spec = spec[0]  # ["Si"] is the element Si
    if isinstance(spec, (str, os.PathLike)) and not _is_element(spec):
        if not os.path.isfile(spec):
            raise ValueError(f"{spec!r} is neither an element symbol nor a structure file")
        from ase.io import read

        return read(spec)
    if kind == "trimer":
        return trimer(spec)
    if kind == "bulk":
        element = spec if isinstance(spec, str) else spec[0]
        return bulk_crystal(element)
    if kind == "p2mm":
        element = spec if isinstance(spec, str) else spec[0]
        return p2mm_crystal(element)
    element = spec if isinstance(spec, str) else spec[0]
    return cluster(element)
