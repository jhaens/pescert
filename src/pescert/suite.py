"""Suite orchestration and reporting.

:class:`Suite` selects proxies, builds the right default substrate per proxy (or uses
user-supplied :class:`ase.Atoms`), runs each within a per-eval call budget, and returns
a :class:`Report` that serializes to JSON and prints the results table.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from ase import Atoms

from .engine import ModelEngine
from .evals._trajectory import TrajectoryCache
from .registry import REGISTRY, get_eval
from .result import EvalResult, _jsonable, aggregate

__all__ = ["Suite", "Report"]

#: The equilibrium proxies share one thermostatted trajectory: config_temperature runs
#: it, the others read its frames (see :mod:`pescert.evals._trajectory`).
TRAJECTORY_PRODUCER = "config_temperature"
TRAJECTORY_CONSUMERS = ("equipartition", "virial")


def _crashed_result(ev, exc: Exception) -> EvalResult:
    """Score a proxy whose measurement the model made impossible.

    A model that returns non-finite or absurd forces blows the proxy up rather than
    scoring badly -- exploding positions overflow a neighbour list, a solver fails to
    converge.  That is a defect of the model, not of the run, so it is recorded as an
    unbounded defect (score 0) and the remaining proxies still run.  ``gate`` stays
    ``None``: no gate was measured, so none is claimed either way.
    """
    _release_accelerator_memory()
    return EvalResult(
        name=ev.name,
        target=float(ev.target),
        raw_defect=float("inf"),
        score=0.0,
        n_model_calls=0,
        details={"error": f"{type(exc).__name__}: {exc}"[:500]},
        gate=None,
    )


def _release_accelerator_memory() -> None:
    """Give back GPU memory an out-of-memory crash left reserved, so later proxies run."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch is optional and this is best-effort
        pass


def _order_for_trajectory_reuse(names: list[str]) -> list[str]:
    """Move the trajectory producer ahead of its consumers, leaving the rest in place.

    Sharing works whichever proxy runs first; pinning the order keeps the MD charged to
    ``config_temperature`` so call counts are reproducible.
    """
    if TRAJECTORY_PRODUCER not in names:
        return list(names)
    first_consumer = [names.index(c) for c in TRAJECTORY_CONSUMERS if c in names]
    if not first_consumer or names.index(TRAJECTORY_PRODUCER) < min(first_consumer):
        return list(names)
    ordered = [n for n in names if n != TRAJECTORY_PRODUCER]
    ordered.insert(min(ordered.index(c) for c in TRAJECTORY_CONSUMERS if c in ordered),
                   TRAJECTORY_PRODUCER)
    return ordered


def _parse_elements(spec) -> list[str]:
    """Split an element spec into a list of symbols.

    Accepts a list/tuple of symbols or a string like ``"Si, C, H"`` / ``"Si C H"``.
    """
    if isinstance(spec, (list, tuple)):
        return [str(e) for e in spec]
    if isinstance(spec, str):
        return [tok for tok in re.split(r"[,\s]+", spec.strip()) if tok]
    raise TypeError(f"cannot parse elements from {spec!r}")


def _multi_elements(substrates) -> list[str] | None:
    """Return the element list if ``substrates`` denotes *several* elements, else None.

    A single symbol (``"Si"`` or ``["Si"]``) is not multi-element and returns None, so
    the ordinary per-proxy default-substrate path is used.
    """
    if isinstance(substrates, str):
        if os.path.isfile(substrates):
            return None  # a structure file, whatever its name contains
        toks = _parse_elements(substrates)
        return toks if len(toks) > 1 else None
    if (
        isinstance(substrates, (list, tuple))
        and len(substrates) > 1
        and all(isinstance(x, str) for x in substrates)
    ):
        return list(substrates)
    return None


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


def _skip_details(rows, elements) -> dict:
    """``{"skipped": True, "message": ...}`` when every element declined the proxy.

    A proxy skipped for one element and measured for another is not a skip -- the mean
    over the elements that ran is still meaningful -- so this only fires when the whole
    row declined, which is the case that must not be read as a score of zero.
    """
    flags = [(r.details or {}).get("skipped") for r in rows]
    if not flags or not all(flags):
        return {}
    messages = {(r.details or {}).get("message", "") for r in rows}
    messages.discard("")
    one = sorted(messages)[0] if len(messages) == 1 else "; ".join(sorted(messages))
    return {
        "skipped": True,
        "message": one or "not applicable to this model",
        "skipped_elements": list(elements),
    }


def _average_reports(per_element: dict[str, Report], names, agg_method: str) -> Report:
    """Average per-element Reports proxy-by-proxy into a single combined Report."""
    import math

    elements = list(per_element)
    averaged: list[EvalResult] = []
    for name in names:
        rows = [next(r for r in per_element[el].results if r.name == name) for el in elements]
        scores = [r.score for r in rows if r.score == r.score]  # drop NaN
        defects = [r.raw_defect for r in rows if r.raw_defect == r.raw_defect]
        mean_score = sum(scores) / len(scores) if scores else float("nan")
        mean_defect = sum(defects) / len(defects) if defects else float("nan")
        # True only if every element that has a gate passed; None if none do
        gate_vals = [r.gate for r in rows if r.gate is not None]
        gate = None if not gate_vals else all(gate_vals)
        averaged.append(
            EvalResult(
                name=name,
                target=rows[0].target,
                raw_defect=mean_defect,
                score=mean_score,
                n_model_calls=int(sum(r.n_model_calls for r in rows)),
                details={
                    "per_element": {el: rows[i].score for i, el in enumerate(elements)},
                    "per_element_defect": {
                        el: rows[i].raw_defect for i, el in enumerate(elements)
                    },
                    # a proxy that crashed scores 0 like a proxy that failed; without the
                    # reason travelling with it the two are indistinguishable downstream
                    "crashed": {
                        el: (rows[i].details or {}).get("error")
                        for i, el in enumerate(elements)
                        if (rows[i].details or {}).get("error")
                    },
                    # ... and a proxy that *declined* (no stress head, no isolated-atom
                    # support) is a third case again: it scores NaN, not 0, and the
                    # reason has to survive averaging or the table can only say "n/a"
                    **_skip_details(rows, elements),
                },
                gate=gate,
            )
        )
    per_overall = {el: per_element[el].aggregate["overall"] for el in elements}
    finite = [v for v in per_overall.values() if isinstance(v, float) and math.isfinite(v)]
    meta = {
        "elements": elements,
        "per_element_overall": per_overall,
        "mean_of_element_overalls": (sum(finite) / len(finite) if finite else float("nan")),
        "evals": list(names),
        "n_total_calls": int(sum(r.n_model_calls for r in averaged)),
    }
    return Report(results=averaged, metadata=meta, agg_method=agg_method)


@dataclass
class Report:
    """Collected results plus metadata and serializes."""

    results: list[EvalResult]
    metadata: dict = field(default_factory=dict)
    #: overall-score averaging method: "geometric" (default) | "arithmetic" | "harmonic".
    agg_method: str = "geometric"

    @property
    def aggregate(self) -> dict:
        return aggregate(self.results, self.agg_method)

    def to_dict(self) -> dict:
        # metadata and aggregate hold raw floats (a skipped proxy contributes NaN), so
        # they go through the same non-finite -> null mapping the results already get:
        # bare NaN/Infinity is not valid JSON and makes the whole file unreadable.
        return _jsonable(
            {
                "metadata": self.metadata,
                "results": [r.to_dict() for r in self.results],
                "aggregate": self.aggregate,
            }
        )

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, allow_nan=False)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def summary(self) -> str:
        """Build (and print) the results as a markdown table, aligned for the terminal."""
        header = ("Probe", "Target", "Defect", "Score", "Calls", "Gate")
        numeric = (False, True, True, True, True, False)
        body = []
        for r in self.results:
            if r.score != r.score:  # NaN: the probe declined to measure
                defect = score = "skip"
            elif (r.details or {}).get("error"):
                defect, score = "crash", f"{r.score:.3f}"
            else:
                defect, score = f"{r.raw_defect:.3e}", f"{r.score:.3f}"
            gate = "" if r.gate is None else ("PASS" if r.gate else "FAIL")
            body.append((r.name, f"{r.target:g}", defect, score, str(r.n_model_calls), gate))
        widths = [max(len(row[i]) for row in (header, *body)) for i in range(len(header))]

        def line(cells) -> str:
            padded = (
                c.rjust(w) if num else c.ljust(w) for c, w, num in zip(cells, widths, numeric)
            )
            return "| " + " | ".join(padded) + " |"

        # the separator's colons right-align the numbers when rendered as markdown, too
        rule = "|" + "|".join(
            "-" * (w + 1) + ":" if num else "-" * (w + 2) for w, num in zip(widths, numeric)
        ) + "|"
        rows = [line(header), rule, *(line(cells) for cells in body)]
        agg = self.aggregate
        if self.metadata.get("elements"):
            rows.append(f"\n_averaged over elements: {', '.join(self.metadata['elements'])}_")
        rows.append(
            f"\n**overall score ({agg.get('overall_method', 'geometric')}):** "
            f"{agg['overall']:.3f}  ·  "
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
            "parity",
            "representation",
            "cross_maxwell",
            "betti",
            "zero_modes",
            "trimer",
            "reversibility",
            "config_temperature",
            "equipartition",
            "virial",
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
        agg_method: str = "geometric",
        share_trajectory: bool = True,
    ) -> Report:
        """Run every selected proxy and return a :class:`Report`.

        Parameters
        ----------
        substrates:
            An :class:`ase.Atoms`, an element symbol/tuple, the path of a structure file,
            or a dict mapping eval name or substrate kind
            (``"cluster"``/``"trimer"``/``"bulk"``/``"p2mm"``) to ``Atoms``.  Passing
            *several* element symbols -- ``["Si", "C", "H"]`` or ``"Si, C, H"`` -- runs
            the suite once per element and returns their average (see
            :meth:`run_elements`).
        seed:
            Base seed forwarded to every proxy (reproducibility).
        budget_per_eval:
            Per-proxy ``max_calls`` budget (``None`` == unlimited).
        configs:
            Optional per-eval keyword overrides, keyed by eval name.
        agg_method:
            Overall-score averaging: ``"geometric"`` (default), ``"arithmetic"`` or
            ``"harmonic"``.
        share_trajectory:
            Run one Langevin trajectory for the equilibrium proxies instead of one each
            (default).  ``config_temperature`` pays for it; ``equipartition`` and
            ``virial`` subsample the same frames -- identical numbers, roughly half the
            calls.  ``False`` gives every proxy its own trajectory.
        """
        elements = _multi_elements(substrates)
        if elements is not None:
            return self.run_elements(
                engine,
                elements,
                seed=seed,
                budget_per_eval=budget_per_eval,
                configs=configs,
                agg_method=agg_method,
                share_trajectory=share_trajectory,
            )

        configs = configs or {}
        names = _order_for_trajectory_reuse(self.names) if share_trajectory else list(self.names)
        # Settle the model's working precision before any proxy runs: it sets every
        # finite-difference step, and a proxy that resolves one before the first model
        # call would otherwise get a different step depending on the run order.  Costs a
        # single model call, and nothing when the caller declared the precision.
        if not engine.precision_is_known and names:
            first = get_eval(names[0])
            engine.detect_precision(_resolve_substrate(substrates, first.substrate_kind, names[0]))
        cache = TrajectoryCache() if share_trajectory else None
        results: list[EvalResult] = []
        for name in names:
            ev = get_eval(name)
            atoms = _resolve_substrate(substrates, ev.substrate_kind, name)
            cfg = dict(configs.get(name, {}))
            if cache is not None and ev.uses_trajectory:
                cfg.setdefault("trajectory_cache", cache)
            try:
                res = ev.run(engine, atoms, max_calls=budget_per_eval, seed=seed, **cfg)
            except Exception as exc:  # noqa: BLE001 - a bad model must be scored, not fatal
                res = _crashed_result(ev, exc)
            results.append(res)
        meta = {
            "seed": seed,
            "budget_per_eval": budget_per_eval,
            "evals": list(names),
            "precision": engine.precision,
            "n_total_calls": int(sum(r.n_model_calls for r in results)),
        }
        if cache is not None:
            meta["shared_trajectory"] = cache.stats()
        return Report(results=results, metadata=meta, agg_method=agg_method)

    def run_elements(
        self,
        engine: ModelEngine,
        elements,
        *,
        seed: int = 0,
        budget_per_eval: int | None = None,
        configs: dict[str, dict] | None = None,
        agg_method: str = "geometric",
        share_trajectory: bool = True,
    ) -> Report:
        """Run the suite once per element and return the per-proxy average as a Report.

        ``elements`` is a list of symbols or a string like ``"Si, C, H"``.  Each proxy's
        default substrate is built for that element; the returned Report holds, for each
        proxy, the mean score/defect across elements, and records the per-element overall
        scores in its metadata.
        """
        els = _parse_elements(elements)
        per_element = {
            el: self.run(
                engine,
                el,
                seed=seed,
                budget_per_eval=budget_per_eval,
                configs=configs,
                agg_method=agg_method,
                share_trajectory=share_trajectory,
            )
            for el in els
        }
        return _average_reports(per_element, self.names, agg_method)
