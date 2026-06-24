"""Eval abstract base class and the per-run model-call budget helper."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ase import Atoms

from ..engine import ModelEngine
from ..result import EvalResult

__all__ = ["Eval", "Budget"]


class Budget:
    """Track and enforce a per-run model-call budget.

    Snapshots :attr:`ModelEngine.n_calls` at construction; :attr:`used` is the delta
    since.  Metrics call :meth:`would_exceed` *before* an expensive batch so they can
    shrink sampling rather than silently overrun, and record ``budget_limited`` in
    their details when they do.
    """

    def __init__(self, engine: ModelEngine, max_calls: int | None):
        self.engine = engine
        self.start = engine.n_calls
        self.max_calls = max_calls

    @property
    def used(self) -> int:
        return self.engine.n_calls - self.start

    @property
    def remaining(self) -> float:
        if self.max_calls is None:
            return float("inf")
        return self.max_calls - self.used

    def would_exceed(self, extra: int) -> bool:
        """Whether spending ``extra`` more calls would overrun the budget."""
        if self.max_calls is None:
            return False
        return self.used + extra > self.max_calls


class Eval(ABC):
    """Abstract proxy.

    Subclasses set the class attributes :attr:`name` (via ``@register``),
    :attr:`section`, :attr:`target`, and :attr:`substrate_kind`, and implement
    :meth:`run`.  ``run`` must respect ``max_calls``, be deterministic given
    ``seed``, and set :attr:`EvalResult.n_model_calls` to the calls it consumed.
    """

    name: str = "eval"
    section: str = ""
    target: float = 0.0
    #: which default substrate the Suite should build: "cluster" | "trimer" | "bulk".
    substrate_kind: str = "cluster"

    @abstractmethod
    def run(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        *,
        max_calls: int | None = None,
        seed: int = 0,
        **cfg,
    ) -> EvalResult:
        """Run the proxy on ``atoms`` and return an :class:`EvalResult`."""

    # -- convenience for subclasses ----------------------------------------
    def _result(
        self,
        *,
        raw_defect: float,
        score: float,
        n_model_calls: int,
        details: dict,
        gate: bool | None = None,
    ) -> EvalResult:
        return EvalResult(
            name=self.name,
            section=self.section,
            target=float(self.target),
            raw_defect=float(raw_defect),
            score=float(score),
            n_model_calls=int(n_model_calls),
            details=details,
            gate=gate,
        )
