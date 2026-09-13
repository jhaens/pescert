"""Proxy implementations.  Importing this package registers every proxy.

Grouped by the four certification sections; each module's docstring states the identity
it tests and its exact target.

**Symmetry & invariance**
    equivariance.py        global-rotation invariance of E, equivariance of F
    parity.py              improper-symmetry stress selection rule
    representation.py      extensivity, lattice gauge, egg-box, permutation
    zero_modes.py          Hessian null space / acoustic sum rule
    trimer.py              many-body decay and transverse self-consistency

**Self-consistency**
    conservativeness.py    collinear bond-stretch force vs dE
    betti.py               Maxwell-Betti reciprocity
    stress_consistency.py  stress vs energy gradient
    cross_maxwell.py       strain-position reciprocity (Lambda block)
    reversibility.py       NVE time-reversibility

**Statistical mechanics**
    config_temperature.py  configurational temperature
    virial.py              Clausius virial theorem
    equipartition.py       per-mode equipartition

**Regularity**
    smoothness.py          BSCT-style bond-scan smoothness
"""

from __future__ import annotations

from . import (  # noqa: F401
    betti,
    config_temperature,
    conservativeness,
    cross_maxwell,
    equipartition,
    equivariance,
    parity,
    representation,
    reversibility,
    smoothness,
    stress_consistency,
    trimer,
    virial,
    zero_modes,
)
