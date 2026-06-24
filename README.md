# pescert — ground-truth-free certification for MLIPs

`pescert` (PES certification) is a small, pip-installable suite of **ground-truth-free**
evaluation proxies for machine-learning interatomic potentials (MLIPs). Every proxy
certifies a model against an *exact* number — **0, 1, or n** — that the true
Born–Oppenheimer PES satisfies by mathematical or physical necessity. **No DFT
reference is used anywhere.** A model can ace energy/force RMSE on a fixed test set and
still fail these, because they measure self-consistency the loss never enforces.

The design and physics are specified in [`mlip_ground_truth_free_evals.md`](mlip_ground_truth_free_evals.md);
each module's docstring cites the section it implements.

> Rename the package by editing `name` in `pyproject.toml` and the `src/pescert/`
> directory — nothing else is hardcoded.

## Install

```bash
pip install -e .            # core (numpy, ase)
pip install -e .[dev]       # + pytest, ruff
pip install -e .[plots]     # + matplotlib (optional diagnostic figures)
```

## Three-line quickstart

```python
from pescert import Suite, from_ase_calculator
engine = from_ase_calculator(my_calc)                 # any ASE calculator
report = Suite.default().run(engine, substrates="Si") # or pass your own ase.Atoms
report.summary()                                       # prints the live results table
report.to_json("out.json")
```

`substrates` may be an element symbol (default substrates are auto-built from covalent
radii), your own `ase.Atoms`, or a dict mapping a proxy/substrate-kind to `Atoms`.

Raw checkpoint (no ASE calculator)? Use `from_callable`:

```python
from pescert import from_callable
engine = from_callable(energy_fn, forces_fn, stress_fn=None)  # each takes ase.Atoms
```

## Command line

```bash
pescert list
pescert run --calc ase.calculators.emt:EMT --element Cu --evals all --json out.json
pescert run --calc my_pkg.models:load_checkpoint --element Si --evals zero_modes,trimer
```

`--calc` is an import path (`module:attr` or `module.attr`) to a calculator instance, a
calculator class, or a zero-argument factory.

## The proxies

Each returns the same `EvalResult`: the exact `target`, a non-negative `raw_defect`, a
normalized `score` in `[0, 1]` (1 = perfect), the `n_model_calls`, rich `details`, and a
`gate` flag for correctness-gate metrics. Only `energy`, `forces`, and (for OOB-2)
`stress` are ever requested from the model; Hessians and Jacobian–vector products are
matrix-free finite differences.

| Proxy | Section | Exact target | What it catches |
|---|---|---|---|
| `conservativeness` | KNOWN | 0 | force ≠ −dE/ds along a bond (energy/force inconsistency) |
| `equivariance` | KNOWN | 0 | broken rotational invariance of E and equivariance of F |
| `smoothness` | KNOWN | score→1 | energy jumps, force kinks, spurious minima, tortuosity |
| `zero_modes` | NEW-1 | 0 ; 6/5/3 | rotational/translational zero-mode & acoustic-sum-rule defects; imaginary modes |
| `config_temperature` | NEW-2a | 1 | non-conservativeness (configurational vs kinetic temperature) |
| `equipartition` | NEW-2b | 1 (per mode) | curvature-vs-dynamics inconsistency, soft modes (mode-resolved) |
| `trimer` | NEW-3 | 0 (×3) | many-body symmetry breaking; 3-body coupling at range; transverse non-conservativeness |
| `betti` | OOB-1 | 0 | Maxwell–Betti response asymmetry (= Hessian asymmetry) |
| `stress_consistency` | OOB-2 | 0 | stress head inconsistent with the energy gradient |
| `reversibility` | OOB-3 | 0 | irreversibility from roughness/discontinuity (NVE round-trip) |

**Reading the scores.** Several proxies sit near machine precision *by construction* for
fully symmetry-constrained, autodiff-conservative models — there they act as
**correctness gates** (`gate=True/False`). Their discriminating power for that class
lives in the numerical-residual diagnostics (raw ASR magnitude, mode-resolved
equipartition spread, many-body decay) and against the fast-growing *unconstrained /
direct-force* class, where they bite hard. The package is honest about this:
`gate is not None` marks a correctness gate.

### Known scope notes (built into the tests)

- **1D radial `conservativeness` is structurally blind to a transverse (curl) force** —
  exactly the gap `trimer` (NEW-3iii, the matrix-free Jacobian-antisymmetry probe)
  closes. The sensitivity tests assert this on purpose.
- **`reversibility`'s round-trip is reversible for any smooth position-dependent force**
  (Velocity-Verlet symmetry), so it catches roughness/discontinuity, not smooth
  non-conservativeness; the energy-drift slope is reported separately.

## Testing

All ground truth is built from analytic ASE potentials — no DFT, no downloads, fully
deterministic:

```bash
pytest -q
```

- **clean pass** — a smooth, conservative, symmetric model (Lennard-Jones / anchored
  harmonic) drives every "=0" defect to ~0, every "=1" ratio to ~1, and every gate to
  pass (proves the math + finite-differencing).
- **broken fires** — a sensitivity matrix of deliberately defective wrappers
  (non-conservative curl, kinked, symmetry-breaking, wrong-stress) trips exactly the
  relevant proxies while unrelated ones stay clean.
- **budget / reproducibility** — call counts stay within documented bounds, `max_calls`
  is honored, and a fixed `seed` reproduces results.

## Design constraints

1. **Architecture-independent** — only `energy/forces/stress`; everything else is
   derived (matrix-free) by the engine.
2. **Training-set-independent** — the identity is universal; only the small in-domain
   substrate is supplied per model.
3. **Cheap** — every proxy tracks and budgets its model calls; substrates are small.
4. **Uniform output** — one `EvalResult` for every proxy.
5. **Reproducible** — every randomized probe takes an explicit `seed`.
