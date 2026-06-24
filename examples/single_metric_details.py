"""Run a single proxy and inspect its rich diagnostics.

    python examples/single_metric_details.py

Demonstrates reading the per-mode equipartition spectrum and the zero-mode breakdown
out of ``EvalResult.details`` — the mode-resolved information a single RMSE cannot show.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.lj import LennardJones

from pescert import cluster, from_ase_calculator, get_eval


def main() -> None:
    engine = from_ase_calculator(LennardJones(epsilon=1.0, sigma=1.0, rc=8.0))
    atoms = cluster("Ar", 13)

    zm = get_eval("zero_modes").run(engine, atoms, seed=0)
    print(zm)
    d = zm.details
    print(f"  expected zero modes: {d['expected_modes']}  found near-zero: {d['n_near_zero']}")
    print(f"  imaginary (negative) modes: {d['n_negative_modes']}")
    print("  per-mode residuals (||H v|| / ||H||_F, should be ~0):")
    for mode, res in d["per_mode_residual"].items():
        print(f"    {mode}: {res:.2e}")

    eq = get_eval("equipartition").run(engine, atoms, seed=0)
    print(eq)
    r_alpha = np.array(eq.details["R_alpha"])
    pcts = np.round(eq.details["R_percentiles_5_25_50_75_95"], 3)
    print(f"  R_alpha median={eq.details['R_median']:.3f} over {r_alpha.size} modes")
    print(f"  percentiles (5/25/50/75/95): {pcts}")


if __name__ == "__main__":
    main()
