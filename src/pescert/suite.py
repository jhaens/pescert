"""Suite orchestration and reporting (spec sections 4 and 6).

:class:`Suite` selects proxies, builds the right default substrate per proxy (or uses
user-supplied :class:`ase.Atoms`), runs each within a per-eval call budget, and returns
a :class:`Report` that serializes to JSON and prints the section-4 results table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ase import Atoms

from .engine import ModelEngine
from .registry import REGISTRY, get_eval
from .result import EvalResult, aggregate

__all__ = ["Suite", "Report"]


def _resolve_substrate(substrates, kind: str, eval_name: str) -> Atoms:
    """Pick the substrate for one proxy from the user's ``substrates`` argument."""
    from .substrates import make_substrate

    if isinstance(substrates, dict):
        if eval_name in substrates:
            return substrates[eval_name]
        if kind in substrates:
            return substrates[kind]
        if "default" in substrates:
            return make_substrate(substrates["default"], kind)
        raise KeyError(f"no substrate provided for {eval_name!r} (kind {kind!r})")
    return make_substrate(substrates, kind)


@dataclass
class Report:
    """Collected results plus metadata; serializes and prints the section-4 table."""

    results: list[EvalResult]
    metadata: dict = field(default_factory=dict)

    @property
    def aggregate(self) -> dict:
        return aggregate(self.results)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
            "aggregate": self.aggregate,
        }

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def summary(self) -> str:
        """Build (and print) the markdown results table from spec section 4."""
        header = (
            "| Proxy | Target | Defect | Score | Calls | Gate |\n"
            "|------|---|---|---|---|"
        )
        rows = [header]
        for r in self.results:
            defect = "skip" if r.score != r.score else f"{r.raw_defect:.3e}"  # NaN check
            score = "skip" if r.score != r.score else f"{r.score:.3f}"
            gate = "" if r.gate is None else ("PASS" if r.gate else "FAIL")
            rows.append(
                f"| {r.name} | {r.target:g} | {defect} | {score} | "
                f"{r.n_model_calls} | {gate} |"
            )
        agg = self.aggregate
        rows.append(
            f"\n**overall score:** {agg['overall']:.3f}  ·  "
            f"**gates:** {agg.get('gates_passed', 0)}/{agg.get('gates_total', 0)}  ·  "
            f"**total model calls:** {agg['n_total_calls']}"
        )
        table = "\n".join(rows)
        print(table)
        return table

    def plot(self, directory: str) -> list[str]:
        """Write per-proxy diagnostic figures (requires the ``[plots]`` extra)."""
        from .plotting import plot_report

        return plot_report(self, directory)


class Suite:
    """A selected set of proxies to run against a model."""

    def __init__(self, names: list[str]):
        missing = [n for n in names if n not in REGISTRY]
        if missing:
            raise KeyError(f"unknown evals: {missing}; available: {sorted(REGISTRY)}")
        self.names = list(names)

    @classmethod
    def default(cls) -> Suite:
        """All registered proxies, in a stable, cheap-first order."""
        order = [
            "conservativeness",
            "equivariance",
            "smoothness",
            "stress_consistency",
            "betti",
            "zero_modes",
            "trimer",
            "reversibility",
            "config_temperature",
            "equipartition",
        ]
        names = [n for n in order if n in REGISTRY] + [n for n in REGISTRY if n not in order]
        return cls(names)

    @classmethod
    def from_names(cls, names: list[str]) -> Suite:
        return cls(names)

    def run(
        self,
        engine: ModelEngine,
        substrates,
        *,
        seed: int = 0,
        budget_per_eval: int | None = None,
        configs: dict[str, dict] | None = None,
    ) -> Report:
        """Run every selected proxy and return a :class:`Report`.

        Parameters
        ----------
        substrates:
            An :class:`ase.Atoms`, an element symbol/tuple, or a dict mapping eval name
            or substrate kind (``"cluster"``/``"trimer"``/``"bulk"``) to ``Atoms``.
        seed:
            Base seed forwarded to every proxy (reproducibility).
        budget_per_eval:
            Per-proxy ``max_calls`` budget (``None`` == unlimited).
        configs:
            Optional per-eval keyword overrides, keyed by eval name.
        """
        configs = configs or {}
        results: list[EvalResult] = []
        for name in self.names:
            ev = get_eval(name)
            atoms = _resolve_substrate(substrates, ev.substrate_kind, name)
            cfg = configs.get(name, {})
            res = ev.run(engine, atoms, max_calls=budget_per_eval, seed=seed, **cfg)
            results.append(res)
        meta = {
            "seed": seed,
            "budget_per_eval": budget_per_eval,
            "evals": list(self.names),
            "n_total_calls": int(sum(r.n_model_calls for r in results)),
        }
        return Report(results=results, metadata=meta)
