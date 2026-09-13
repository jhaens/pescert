"""Shared cell-strain helpers for the periodic probes.

The stress--gradient probe and the strain--position reciprocity probe
(cross-Maxwell) both deform the cell by an affine strain.  They *share this code path*
on purpose: the finite-difference stress and the strained forces must use the identical
deformation convention, otherwise the two probes could silently drift apart.

Convention: an engineering Voigt strain ``[e_xx, e_yy, e_zz, e_yz, e_xz, e_xy]`` maps to
the symmetric strain tensor ``eps``; the deformation gradient ``F = I + eps`` is applied
to the reference cell as ``cell0 @ F.T`` with the atoms scaled affinely
(``scale_atoms=True``).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


def strain_tensor(voigt_strain: np.ndarray) -> np.ndarray:
    """Symmetric strain tensor from an engineering Voigt strain vector."""
    e1, e2, e3, e4, e5, e6 = voigt_strain
    return np.array(
        [
            [e1, e6 / 2.0, e5 / 2.0],
            [e6 / 2.0, e2, e4 / 2.0],
            [e5 / 2.0, e4 / 2.0, e3],
        ]
    )


def apply_strain(atoms: Atoms, cell0: np.ndarray, eps_tensor: np.ndarray) -> Atoms:
    """Return a copy of ``atoms`` with the reference cell ``cell0`` affinely strained."""
    f = np.eye(3) + eps_tensor
    work = atoms.copy()
    work.set_cell(np.asarray(cell0) @ f.T, scale_atoms=True)
    return work
