"""Small seeded-MD helpers shared by the dynamics-based proxies."""

from __future__ import annotations

import numpy as np
from ase import Atoms, units


def seeded_velocities(atoms: Atoms, temperature_K: float, seed: int) -> np.ndarray:
    """Reproducible Maxwell-Boltzmann velocities with zero net momentum.

    Returns velocities in ASE units (Angstrom / (ASE time unit)).  Same ``seed`` gives
    identical velocities.
    """
    rng = np.random.default_rng(seed)
    masses = atoms.get_masses()[:, None]  # amu
    # momenta p ~ N(0, m kB T); velocity = p / m
    std = np.sqrt(masses * units.kB * temperature_K)
    momenta = rng.standard_normal((len(atoms), 3)) * std
    momenta -= momenta.mean(0)  # remove net translation
    return momenta / masses
