"""pescert -- ground-truth-free certification proxies for MLIPs.

Each proxy certifies a model against an *exact* number (0, 1, or ``n``) that the true
Born-Oppenheimer PES satisfies by necessity -- with no DFT reference anywhere.  Each
proxy module's docstring states the identity it tests and its exact target.
"""

from __future__ import annotations

# importing the evals package populates the registry via @register
from . import evals as _evals  # noqa: E402,F401
from .engine import ModelEngine, from_ase_calculator, from_callable
from .reference import ElementLennardJones, lj_parameters
from .registry import REGISTRY, available, get_eval, register
from .result import (
    AVERAGE_METHODS,
    EvalResult,
    aggregate,
    combine_scores,
    score_from_defect,
)
from .substrates import bulk_crystal, cluster, make_substrate, p2mm_crystal, trimer
from .suite import Report, Suite  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "ModelEngine",
    "from_ase_calculator",
    "from_callable",
    "EvalResult",
    "score_from_defect",
    "aggregate",
    "combine_scores",
    "AVERAGE_METHODS",
    "register",
    "get_eval",
    "available",
    "REGISTRY",
    "cluster",
    "trimer",
    "bulk_crystal",
    "p2mm_crystal",
    "make_substrate",
    "ElementLennardJones",
    "lj_parameters",
    "Suite",
    "Report",
]
