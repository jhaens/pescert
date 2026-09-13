"""Console entry point: ``pescert run --calc <import.path> --element Si --evals all``."""

from __future__ import annotations

import argparse
import importlib
import sys

from .engine import from_ase_calculator
from .registry import REGISTRY
from .suite import Suite


def _load_calculator(path: str):
    """Import ``module:attr`` (or ``module.attr``) to a Calculator instance/class/factory.

    Coercion to an instance is handled centrally by :class:`ModelEngine`, so the CLI and
    the Python API resolve a calculator identically.
    """
    if ":" in path:
        mod_name, attr = path.split(":", 1)
    else:
        mod_name, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod_name), attr)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pescert", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run probes against a model")
    run.add_argument("--calc", required=True, help="import path to a calculator/factory")
    run.add_argument(
        "--element",
        default="Ar",
        help="element for default substrates; several ('Si,C,H') averages over elements",
    )
    run.add_argument(
        "--evals",
        default="all",
        help="'all' or a comma-separated list of eval names",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--budget", type=int, default=None, help="per-eval max model calls")
    run.add_argument(
        "--agg-method",
        default="arithmetic",
        choices=["arithmetic", "geometric", "harmonic"],
        help="how to combine sub-scores into the overall (default: arithmetic)",
    )
    run.add_argument("--json", default=None, help="write the JSON report to this path")
    run.add_argument(
        "--no-share-trajectory",
        dest="share_trajectory",
        action="store_false",
        help="give every equilibrium probe its own MD run instead of sharing one "
        "(config_temperature's trajectory is reused by equipartition and virial)",
    )

    sub.add_parser("list", help="list available probes")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "list":
        for name in sorted(REGISTRY):
            ev = REGISTRY[name]
            print(f"{name:22s} {ev.substrate_kind}")
        return 0

    if args.command == "run":
        calc = _load_calculator(args.calc)
        engine = from_ase_calculator(calc)
        suite = Suite.default() if args.evals == "all" else Suite.from_names(args.evals.split(","))
        report = suite.run(
            engine,
            substrates=args.element,
            seed=args.seed,
            budget_per_eval=args.budget,
            agg_method=args.agg_method,
            share_trajectory=args.share_trajectory,
        )
        report.summary()
        if args.json:
            report.to_json(args.json)
            print(f"\nwrote {args.json}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
