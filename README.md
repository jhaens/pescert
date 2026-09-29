<p align="center">
  <img src="assets/pescert.svg" width="68" alt="pescert">
</p>
<h1 align="center">pescert</h1>
<p align="center"><b>Reference-free certification for machine-learning interatomic potentials.</b></p>

`pescert` scores an interatomic potential without any DFT reference.

Each of its 14 probes tests one identity that the exact Born-Oppenheimer surface has to
satisfy, so the target is an exact number (0, 1, or an integer count) instead of a
computed reference. A model can reach excellent energy and force errors on a test set and
still fail these, because they measure consistency that the training loss never enforces.

Only `energy`, `forces` and `stress` are ever requested from the model. Hessians and
Jacobian-vector products are matrix-free finite differences, so any architecture works.

## Install

```bash
git clone https://github.com/jhaens/pescert.git && cd pescert
pip install .               # core (numpy, ase)
pip install ".[plots]"      # + matplotlib for diagnostic figures
pip install -e ".[dev]"     # editable, + pytest, ruff, for development
```

## Use

```python
from pescert import Suite, from_ase_calculator

engine = from_ase_calculator(my_calc)                  # any ASE calculator
report = Suite.default().run(engine, substrates="Si")  # or pass your own ase.Atoms
report.summary()                                       # prints the results table
report.to_json("out.json")
```

`substrates` takes an element symbol, your own `ase.Atoms`, or several symbols
(`["Si", "C"]`) to run per element and average. No ASE calculator? Use
`from_callable(energy_fn, forces_fn, stress_fn=None)`.

```bash
pescert list
pescert run --calc ase.calculators.emt:EMT --element Cu --json out.json
pescert run --calc my_pkg.models:load_checkpoint --element Si --evals zero_modes,trimer
```

`--calc` is an import path (`module:attr`) to a calculator instance, a calculator class,
or a zero-argument factory.

## Example: MACE-MP-0 (small)

```python
from mace.calculators import mace_mp
from pescert import Suite, from_ase_calculator

calc = mace_mp(model="small", default_dtype="float64", device="cuda")
engine = from_ase_calculator(calc, precision="float64")

report = Suite.default().run(engine, substrates="Si", seed=0)
report.summary()
report.to_json("mace_mp_0_small.json")
```

`--calc` cannot pass model options, so from the command line wrap the model in a
zero-argument factory:

```python
# mymodels.py
from mace.calculators import mace_mp

def mace_mp_small():
    return mace_mp(model="small", default_dtype="float64", device="cuda")
```

```bash
pescert run --calc mymodels:mace_mp_small --element Si --json mace_mp_0_small.json
```

## The probes

| Family | Probe | Target | What it catches |
|---|---|---|---|
| Symmetry and invariance | `equivariance` | 0 | rotational invariance of E, equivariance of F |
| | `parity` | 0 | improper-symmetry leakage in the stress |
| | `representation` | 0 | non-extensivity, lattice gauge, egg-box, permutation |
| | `zero_modes` | 0, 6/5/3 | broken rigid-body null space, imaginary modes |
| | `trimer` | 0 | many-body term surviving at range, Jacobian asymmetry |
| Self-consistency | `conservativeness` | 0 | force not equal to `-dE/ds` along a bond |
| | `betti` | 0 | Maxwell-Betti response asymmetry |
| | `stress_consistency` | 0 | stress head inconsistent with the energy gradient |
| | `cross_maxwell` | 0 | force and stress heads disagreeing on internal strain |
| | `reversibility` | 0 | irreversibility from roughness or internal state |
| Statistical mechanics | `config_temperature` | 1 | configurational against kinetic temperature |
| | `virial` | 1 | Clausius virial against internal kinetic energy |
| | `equipartition` | 1 | curvature against dynamics, mode resolved |
| Regularity | `smoothness` | 0 | energy jumps, force kinks, spurious minima |

Every probe returns the same `EvalResult`: the exact `target`, a non-negative
`raw_defect`, a `score` in `[0, 1]` (1 is ideal), `n_model_calls`, `details`, and a `gate`
flag. Each module docstring states the identity it tests and how the defect becomes a
score.

For a symmetry-constrained model whose forces come from differentiating an energy, many
probes sit near machine precision by construction and act as correctness gates. They bite
hardest on unconstrained and direct-force models.

## Suite parameters

A full run with every option, on the command line and in Python. The table below explains
each one.

```bash
pescert run --calc <module:attr> --element <element or list> --evals <all or probe list> \
            --seed <int> --budget <int> --agg-method <arithmetic|geometric|harmonic> \
            --no-share-trajectory --json <path>
```

```python
engine = from_ase_calculator(calc, precision=None)
report = Suite.from_names(names).run(  # Suite.default() selects all 14 probes
    engine,
    substrates,
    seed=0,
    budget_per_eval=None,
    configs=None,
    agg_method="geometric",
    share_trajectory=True,
)
report.to_json(path)
```

| Python | CLI | Default | What it does |
|---|---|---|---|
| `calc` | `--calc` | required | the model: an ASE calculator instance, class or zero-argument factory, on the command line as an import path; `from_callable()` wraps raw energy and force functions instead |
| `precision` | — | `None` (detected) | the model's float precision, `"float32"` or `"float64"`, which sizes the finite-difference steps; `None` detects it with one extra model call |
| `substrates` | `--element` | required (CLI: `Ar`) | the structures to probe: an element symbol or several (run per element, then averaged), or a structure file ASE can read; from Python also an `ase.Atoms`, or a dict mapping probe names or substrate kinds (`cluster`, `trimer`, `bulk`, `p2mm`) to `ase.Atoms` |
| `names` | `--evals` | all 14 (`Suite.default()`) | which probes run: a list of probe names, on the command line `all` or a comma-separated list such as `zero_modes,trimer` |
| `seed` | `--seed` | `0` | base seed of every randomized probe; the same seed reproduces a run |
| `budget_per_eval` | `--budget` | `None` (no cap) | soft cap on model calls per probe: sampling shrinks to fit, but relaxations, Hessians and MD warm-up are not capped |
| `configs` | — | `None` | per-probe keyword overrides keyed by probe name, e.g. `{"zero_modes": {"relax_fmax": 1e-4}}`; the options are the keyword arguments of each probe's `run()` |
| `agg_method` | `--agg-method` | `"geometric"` | how the probe scores combine into the overall score: `arithmetic`, `geometric` or `harmonic` |
| `share_trajectory` | `--no-share-trajectory` | `True` | one Langevin trajectory for the three statistical-mechanics probes instead of one each; the flag turns sharing off |
| `to_json(path)` | `--json` | not written | writes the report as JSON |

## Tests

```bash
pytest -q
```

Ground truth is analytic, so there are no downloads and no DFT. The tests check that a
clean model passes every probe, that deliberately broken models trip exactly the relevant
ones, and that call budgets and seeds hold.

## Notes

- Finite-difference steps are sized to the model's float precision, so a residual is the
  model's and not round-off. Declare it with `from_ase_calculator(calc, precision="float32")`
  to skip the probing call.
- The three statistical-mechanics probes share one Langevin trajectory in a suite run,
  which roughly halves their cost. `share_trajectory=False` gives each its own.
- Every randomized probe takes an explicit `seed`, so a run reproduces.

## License

MIT. See [LICENSE](LICENSE).
