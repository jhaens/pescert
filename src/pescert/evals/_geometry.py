"""Shared geometry helpers: bond selection, rigid zero modes, random rotations."""

from __future__ import annotations

import numpy as np
from ase import Atoms


def closest_pair(atoms: Atoms) -> tuple[int, int]:
    """Return the indices of the closest pair of atoms (a representative bond)."""
    pos = atoms.get_positions()
    n = len(pos)
    if n < 2:
        raise ValueError("need at least 2 atoms to pick a bond")
    best = (0, 1)
    best_d = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < best_d:
                best_d = d
                best = (i, j)
    return best


def is_linear(positions: np.ndarray, tol: float = 1e-6) -> bool:
    """Whether all atoms are (nearly) collinear."""
    p = positions - positions.mean(0)
    if len(p) < 3:
        return True
    # rank of the coordinate spread
    s = np.linalg.svd(p, compute_uv=False)
    return s[1] < tol * max(s[0], 1e-30)


def rigid_zero_modes(
    positions: np.ndarray,
    masses: np.ndarray | None = None,
    *,
    rotations: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Analytic translational (and rotational) zero modes from geometry alone.

    Returns an orthonormal basis ``(n_modes, 3N)`` and a list of labels.  These are
    the exact null vectors of the (mass-weighted, if ``masses`` given) Hessian forced
    by translational/rotational invariance of the energy -- no diagonalisation, no
    reference needed (spec NEW-1).
    """
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    com = pos.mean(0)
    rel = pos - com
    if masses is None:
        sqrt_m = np.ones(n)
    else:
        sqrt_m = np.sqrt(np.asarray(masses, dtype=float))

    modes = []
    labels = []
    # translations
    for a, lab in enumerate(["Tx", "Ty", "Tz"]):
        v = np.zeros((n, 3))
        v[:, a] = 1.0
        v = v * sqrt_m[:, None]
        modes.append(v.reshape(-1))
        labels.append(lab)
    # rotations: v_i = e_a x r_i
    if rotations:
        for a, lab in zip(range(3), ["Rx", "Ry", "Rz"]):
            e = np.zeros(3)
            e[a] = 1.0
            v = np.cross(np.tile(e, (n, 1)), rel)
            v = v * sqrt_m[:, None]
            nrm = np.linalg.norm(v)
            if nrm > 1e-12:
                modes.append(v.reshape(-1))
                labels.append(lab)

    basis = np.array(modes)
    # orthonormalise (translations and rotations can be non-orthogonal once mass-weighted)
    q, _ = np.linalg.qr(basis.T)
    ortho = q.T[: len(basis)]
    # keep labels aligned with the number of retained modes
    return ortho, labels[: len(ortho)]


def random_rotations(n: int, seed: int) -> np.ndarray:
    """Return ``n`` proper random rotation matrices (3x3), reproducible from ``seed``."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, 3))
    for k in range(n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, 2 * np.pi)
        out[k] = _rodrigues(axis, angle)
    return out


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
