"""Uniform result object, score normalization, and aggregation (spec section 6).

Every proxy returns the same :class:`EvalResult`: the exact target value (0, 1, or
``n``), the non-negative raw deviation from it, a normalized score in ``[0, 1]``
(1 == perfect), the model-call count, and a rich ``details`` dict.  Mapping a defect
to a score is done by :func:`score_from_defect`; each proxy chooses and *records* the
scale it used.  :func:`aggregate` implements the leaderboard aggregation: keep
sub-scores separate, also report an overall.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = ["EvalResult", "score_from_defect", "aggregate"]


def score_from_defect(defect: float, scale: float) -> float:
    """Map a non-negative ``defect`` to a score in ``(0, 1]`` via ``exp(-defect/scale)``.

    A defect of 0 scores 1; a defect equal to ``scale`` scores ``1/e ~ 0.37``.  The
    ``scale`` sets the sensitivity and **must be recorded** in the result details so
    scores are interpretable across models.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    d = abs(float(defect))
    if not math.isfinite(d):
        return 0.0
    return float(math.exp(-d / scale))


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy types/arrays so a result is JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


@dataclass
class EvalResult:
    """The result of running one proxy.

    Attributes
    ----------
    name:
        Registry name of the proxy (e.g. ``"zero_modes"``).
    target:
        Exact target value the true PES satisfies: ``0.0``, ``1.0``, or an int ``n``.
    raw_defect:
        Primary deviation from ``target``; non-negative by convention.
    score:
        Normalized quality in ``[0, 1]``; 1 == perfect.
    n_model_calls:
        Number of model single-point evaluations consumed.
    details:
        Arrays, sub-scores, per-mode values, ``eps``/``seed``/``scale`` used, etc.
    gate:
        ``True``/``False`` for correctness-gate metrics (near-0 by construction for
        symmetry-exact / autodiff-conservative models); ``None`` if purely graded.
    """

    name: str
    target: float
    raw_defect: float
    score: float
    n_model_calls: int
    details: dict = field(default_factory=dict)
    gate: bool | None = None

    def to_dict(self) -> dict:
        """Return a fully JSON-serializable dict."""
        return _jsonable(asdict(self))

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str:
        """Serialize to JSON; write to ``path`` if given and return the string."""
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def __str__(self) -> str:
        gate = "" if self.gate is None else f" gate={'PASS' if self.gate else 'FAIL'}"
        return (
            f"{self.name}: target={self.target:g} "
            f"defect={self.raw_defect:.3e} score={self.score:.3f} "
            f"calls={self.n_model_calls}{gate}"
        )


def aggregate(results: list[EvalResult]) -> dict:
    """Aggregate a list of results (spec section 6).

    Returns sub-scores per proxy *and* an overall (geometric mean of scores, which
    penalizes any single bad axis), plus the gate pass/fail tally.  Sub-scores are
    always kept separate so non-conservative vs conservative models separate cleanly.
    """
    if not results:
        return {"overall": float("nan"), "sub_scores": {}, "n_total_calls": 0}
    sub = {r.name: r.score for r in results}
    scores = np.array(list(sub.values()), dtype=float)
    scores = scores[np.isfinite(scores)]
    # geometric mean penalizes any single bad axis more than an arithmetic mean
    if scores.size:
        overall = float(np.exp(np.mean(np.log(np.clip(scores, 1e-12, 1.0)))))
    else:
        overall = float("nan")
    gates = {r.name: r.gate for r in results if r.gate is not None}
    return {
        "overall": overall,
        "sub_scores": sub,
        "gates": gates,
        "gates_passed": sum(1 for v in gates.values() if v),
        "gates_total": len(gates),
        "n_total_calls": int(sum(r.n_model_calls for r in results)),
    }
