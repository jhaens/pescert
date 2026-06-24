"""Proxy implementations.  Importing this package registers every proxy.

Each module maps to a spec section (see its docstring):

============================  ========  ====================================
module                        section   proxy
============================  ========  ====================================
smoothness.py                 KNOWN     BSCT-style bond-scan smoothness
equivariance.py               KNOWN     global-rotation invariance of E, F
conservativeness.py           KNOWN     1D bond-stretch force vs dE
zero_modes.py                 NEW-1     zero-mode / acoustic-sum-rule defect
config_temperature.py         NEW-2a    configurational temperature
equipartition.py              NEW-2b    per-mode equipartition
trimer.py                     NEW-3     many-body / transverse self-consistency
betti.py                      OOB-1     Maxwell-Betti reciprocity
stress_consistency.py         OOB-2     stress vs energy-gradient
reversibility.py              OOB-3     NVE time-reversibility
============================  ========  ====================================
"""

from __future__ import annotations

from . import (  # noqa: F401
    betti,
    config_temperature,
    conservativeness,
    equipartition,
    equivariance,
    reversibility,
    smoothness,
    stress_consistency,
    trimer,
    zero_modes,
)
