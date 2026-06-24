"""Run the full pescert suite on a stock ASE calculator and print the results table.

    python examples/run_full_suite.py

Uses EMT (ships with ASE) so it runs with no extra dependencies.  EMT is a real
many-body embedded-atom potential, so the proxies report meaningful, non-trivial
numbers (it is conservative but not symmetry-exact, and EMT has a cutoff).
"""

from __future__ import annotations

from ase.calculators.emt import EMT

from pescert import Suite, from_ase_calculator


def main() -> None:
    engine = from_ase_calculator(EMT())
    # Cu has an EMT parameterization; default substrates are built from the element.
    report = Suite.default().run(engine, substrates="Cu", seed=0, budget_per_eval=2000)
    report.summary()
    report.to_json("pescert_emt_Cu.json")
    print("\nwrote pescert_emt_Cu.json")


if __name__ == "__main__":
    main()
