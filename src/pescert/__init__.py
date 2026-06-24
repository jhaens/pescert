"""pescert -- ground-truth-free certification proxies for MLIPs.

Each proxy certifies a model against an *exact* number (0, 1, or ``n``) that the true
Born-Oppenheimer PES satisfies by necessity -- with no DFT reference anywhere.  See
``mlip_ground_truth_free_evals.md`` for the authoritative specification.
"""

from __future__ import annotations

# Importing the evals package populates the registry via @register decorators.
from . import evals as _evals  # noqa: E402,F401
from .engine import ModelEngine, from_ase_calculator, from_callable
from .registry import REGISTRY, available, get_eval, register
from .result import EvalResult, aggregate, score_from_defect
from .substrates import bulk_crystal, cluster, make_substrate, trimer
from .suite import Report, Suite  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "ModelEngine",
    "from_ase_calculator",
    "from_callable",
    "EvalResult",
    "score_from_defect",
    "aggregate",
    "register",
    "get_eval",
    "available",
    "REGISTRY",
    "cluster",
    "trimer",
    "bulk_crystal",
    "make_substrate",
    "Suite",
    "Report",
]
