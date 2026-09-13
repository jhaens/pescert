"""Uniform result object, score normalization, and aggregation.

Every proxy returns the same :class:`EvalResult`: the exact target (0, 1, or ``n``),
the non-negative deviation from it, a normalized score in ``[0, 1]``, the model-call
count, and a ``details`` dict.  :func:`score_from_defect` maps defect to score; each
proxy chooses and *records* the scale it used.  :func:`aggregate` keeps the sub-scores
separate and adds an overall.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = ["EvalResult", "score_from_defect", "aggregate", "AVERAGE_METHODS", "combine_scores"]

#: Overall-score averaging methods, in increasing severity towards a single bad axis.
AVERAGE_METHODS = ("arithmetic", "geometric", "harmonic")


def combine_scores(scores, method: str = "arithmetic") -> float:
    """Combine per-proxy ``scores`` into a single overall score in ``[0, 1]``.

    ``"arithmetic"`` (default) is the plain average, ``"geometric"`` penalises any
    single bad axis, ``"harmonic"`` is dominated by the worst.  Non-finite scores are
    dropped; an empty input returns NaN.
    """
    s = np.asarray(list(scores), dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    if method == "arithmetic":
        return float(np.mean(s))
    if method == "geometric":
        return float(np.exp(np.mean(np.log(np.clip(s, 1e-12, 1.0)))))
    if method == "harmonic":
        return float(s.size / np.sum(1.0 / np.clip(s, 1e-12, 1.0)))
    raise ValueError(f"unknown average method {method!r}; choose from {AVERAGE_METHODS}")


def score_from_defect(defect: float, scale: float) -> float:
    """Map a non-negative ``defect`` to a score in ``(0, 1]`` via ``exp(-defect/scale)``.

    A defect of 0 scores 1, a defect of ``scale`` scores ``1/e``.  ``scale`` sets the
    sensitivity and **must be recorded** in the details, or scores are not comparable
    across models.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    d = abs(float(defect))
    if not math.isfinite(d):
        return 0.0
    return float(math.exp(-d / scale))


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy types/arrays so a result is JSON-serializable.

    Non-finite values become ``null``.  ``NaN``/``Infinity`` are what Python's json
    module emits for them by default, but they are **not** valid JSON: a strict reader
    -- every browser's ``JSON.parse`` included -- rejects the whole document over one of
    them, so a single diverged probe would make the run output unreadable.  ``null``
    round-trips as "not measured", which is what a non-finite score means anyway.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()  # fall through: a numpy inf must meet the finiteness check too
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
        text = json.dumps(self.to_dict(), indent=indent, allow_nan=False)
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


def aggregate(results: list[EvalResult], method: str = "arithmetic") -> dict:
    """Aggregate a list of results into sub-scores, an overall, and the gate tally.

    Sub-scores are always kept separate, so conservative and non-conservative models
    separate cleanly.  ``method`` (see :func:`combine_scores`) is recorded under
    ``"overall_method"``.
    """
    if not results:
        return {
            "overall": float("nan"),
            "overall_method": method,
            "sub_scores": {},
            "n_total_calls": 0,
        }
    sub = {r.name: r.score for r in results}
    overall = combine_scores(sub.values(), method)
    gates = {r.name: r.gate for r in results if r.gate is not None}
    return {
        "overall": overall,
        "overall_method": method,
        "sub_scores": sub,
        "gates": gates,
        "gates_passed": sum(1 for v in gates.values() if v),
        "gates_total": len(gates),
        "n_total_calls": int(sum(r.n_model_calls for r in results)),
    }
