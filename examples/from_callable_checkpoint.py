"""Certify a raw checkpoint that exposes only energy/forces callables (no ASE calculator).

    python examples/from_callable_checkpoint.py

Here the "model" is a toy analytic potential, but the pattern is exactly how you wrap a
loaded neural-network checkpoint: provide ``energy_fn(atoms) -> float`` and
``forces_fn(atoms) -> (N, 3)`` (and optionally ``stress_fn``).
"""

from __future__ import annotations

import numpy as np

from pescert import Suite, from_callable


def energy_fn(atoms) -> float:
    """A smooth pairwise (Morse-like) energy over all pairs."""
    pos = atoms.get_positions()
    e = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            r = np.linalg.norm(pos[i] - pos[j])
            x = np.exp(-2.0 * (r - 1.2))
            e += x - 2.0 * np.sqrt(x)
    return float(e)


def forces_fn(atoms) -> np.ndarray:
    """Analytic forces consistent with ``energy_fn``."""
    pos = atoms.get_positions()
    f = np.zeros_like(pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = pos[i] - pos[j]
            r = np.linalg.norm(d)
            x = np.exp(-2.0 * (r - 1.2))
            dedr = -2.0 * x + 2.0 * np.exp(-(r - 1.2))
            g = dedr * d / r
            f[i] -= g
            f[j] += g
    return f


def main() -> None:
    engine = from_callable(energy_fn, forces_fn)  # stress_fn optional
    report = Suite.from_names(["zero_modes", "trimer", "reversibility"]).run(
        engine, substrates="C", seed=0
    )
    report.summary()


if __name__ == "__main__":
    main()
