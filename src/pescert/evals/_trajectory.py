"""One shared Langevin trajectory for the equilibrium proxies.

``config_temperature``, ``equipartition`` and ``virial`` each need a short thermostatted
trajectory started from the model's own minimum, and what they extract from a snapshot
(a Hutchinson trace, normal-mode projections, the Clausius virial) is cheap
post-processing of positions, forces and velocities the integrator has already produced.
This module runs that trajectory once per (substrate, thermostat) setting and lets each
proxy subsample it at its own stride.

Frames are selected by absolute MD step index, so reading at ``sample_every=k`` yields
exactly the snapshots ``dyn.attach(..., interval=k)`` would have fired -- reuse changes
cost, not numbers.  Storing a frame costs no model call: the Langevin step ends by
evaluating the forces at the new positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms, units

from ..engine import ModelEngine
from ._md import seeded_velocities

__all__ = [
    "Frames",
    "Thermostat",
    "Trajectory",
    "TrajectoryCache",
    "get_trajectory",
    "n_samples_for",
    "stored_steps",
]

# Thermostat defaults shared by config_temperature, equipartition and virial:
TEMPERATURE_K = 30.0
WARMUP_STEPS = 400
N_STEPS = 1100
DT = 1.0
FRICTION = 0.05
RELAX_FMAX = 1e-3


@dataclass(frozen=True)
class Thermostat:
    """Everything that defines a trajectory, and therefore its cache key.

    ``n_steps`` is deliberately not part of it: a shorter run is a prefix of a longer
    one, so a proxy wanting fewer steps truncates (:meth:`Trajectory.frames`) and one
    wanting more continues the same dynamics (:meth:`Trajectory.extend`).
    """

    seed: int = 0
    temperature_K: float = TEMPERATURE_K
    warmup_steps: int = WARMUP_STEPS
    dt: float = DT
    friction: float = FRICTION
    relax_fmax: float = RELAX_FMAX
    do_relax: bool = True


def n_samples_for(warmup_steps: int, n_steps: int, sample_every: int) -> int:
    """How many snapshots sampling every ``sample_every`` steps yields.

    ASE calls an observer when the absolute step counter is divisible by its interval.
    For budget planning only; :meth:`Trajectory.frames` selects the actual frames.
    """
    if sample_every <= 0:
        raise ValueError(f"sample_every must be positive, got {sample_every}")
    return (warmup_steps + n_steps) // sample_every - warmup_steps // sample_every


def _fingerprint(atoms: Atoms) -> tuple:
    """Hashable identity of a substrate: species, geometry and cell."""
    # + 0.0 normalizes -0.0 to 0.0 so equal geometries hash equal
    return (
        tuple(atoms.get_chemical_symbols()),
        np.ascontiguousarray(atoms.get_positions() + 0.0).tobytes(),
        np.ascontiguousarray(atoms.get_cell().array + 0.0).tobytes(),
        tuple(bool(p) for p in atoms.get_pbc()),
    )


@dataclass
class Frames:
    """A subsampled slice of a :class:`Trajectory`; arrays are ``(n_frames, N, 3)``."""

    steps: np.ndarray  # absolute MD step index of each frame
    positions: np.ndarray
    forces: np.ndarray
    velocities: np.ndarray

    def __len__(self) -> int:
        return int(self.steps.size)


class Trajectory:
    """A stored Langevin trajectory: :attr:`reference` plus every frame after warmup.

    Construction relaxes to the model's own minimum and equilibrates; sampling steps are
    added by :meth:`extend`, so a caller can size the run against its remaining budget
    once the fixed cost is known.  :attr:`reference` is the relaxed geometry the run
    started from -- the ``R^eq`` the virial and equipartition probes displace against.
    """

    def __init__(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        thermostat: Thermostat,
        *,
        owner: str = "",
    ):
        from ase.constraints import FixCom
        from ase.md.langevin import Langevin

        self.thermostat = thermostat
        self.owner = owner
        self._engine = engine
        calls0 = engine.n_calls

        self.reference = (
            engine.relax(atoms, fmax=thermostat.relax_fmax)
            if thermostat.do_relax
            else atoms.copy()
        )
        self._work = engine.attach(self.reference)
        # pin the centre of mass with a constraint: Langevin's own fixcm does not sample
        self._work.set_constraint(FixCom())
        # kinetic degrees of freedom the trajectory samples: 3N minus the pinned centre
        self.kinetic_dof = 3 * len(self.reference) - sum(
            c.get_removed_dof(self._work) for c in self._work.constraints
        )
        self._work.set_velocities(
            seeded_velocities(self.reference, thermostat.temperature_K, thermostat.seed)
        )
        self._dyn = Langevin(
            self._work,
            timestep=thermostat.dt * units.fs,
            temperature_K=thermostat.temperature_K,
            friction=thermostat.friction / units.fs,
            rng=np.random.default_rng(thermostat.seed),
            fixcm=False
        )
        self._steps: list[int] = []
        self._pos: list[np.ndarray] = []
        self._frc: list[np.ndarray] = []
        self._vel: list[np.ndarray] = []

        # equilibrate before storing anything, then record every step
        self._dyn.run(thermostat.warmup_steps)
        self._dyn.attach(self._store, interval=1)
        self.n_model_calls = engine.n_calls - calls0

    # -- recording ---------------------------------------------------------
    def _store(self) -> None:
        # get_forces() is a cache hit here: the Langevin step just evaluated them.  
        # The stored forces are the model's own: FixCom would strip their net component, 
        # and a net force is part of what the probes measure.
        self._steps.append(int(self._dyn.nsteps))
        self._pos.append(self._work.get_positions())
        self._frc.append(self._work.get_forces(apply_constraint=False))
        self._vel.append(self._work.get_velocities())

    @property
    def n_steps(self) -> int:
        """Number of MD steps stored (equilibration excluded)."""
        return len(self._steps)

    def extend(self, n_steps: int) -> int:
        """Run on until ``n_steps`` frames are stored; return the model calls spent.

        Continuing the stored dynamics reproduces what a single longer run would have
        given (same integrator, same RNG stream).  Resuming can cost one extra call: if
        another proxy has since evaluated a displaced geometry, ASE's single-slot result
        cache no longer holds the forces at the current frame.
        """
        missing = int(n_steps) - self.n_steps
        if missing <= 0:
            return 0
        calls0 = self._engine.n_calls
        self._dyn.run(missing)
        spent = self._engine.n_calls - calls0
        self.n_model_calls += spent
        return spent

    # -- consumption -------------------------------------------------------
    def frames(self, *, sample_every: int = 1, n_steps: int | None = None) -> Frames:
        """The frames a proxy sampling every ``sample_every`` steps would have seen.

        Truncated to the first ``n_steps`` sampling steps (all of them by default).
        """
        if sample_every <= 0:
            raise ValueError(f"sample_every must be positive, got {sample_every}")
        last = self.thermostat.warmup_steps + (
            self.n_steps if n_steps is None else int(n_steps)
        )
        idx = [i for i, s in enumerate(self._steps) if s <= last and s % sample_every == 0]
        n_atoms = len(self.reference)

        def stack(src: list[np.ndarray]) -> np.ndarray:
            if not idx:
                return np.empty((0, n_atoms, 3))
            return np.asarray([src[i] for i in idx], dtype=float)

        return Frames(
            steps=np.asarray([self._steps[i] for i in idx], dtype=int),
            positions=stack(self._pos),
            forces=stack(self._frc),
            velocities=stack(self._vel),
        )


class TrajectoryCache:
    """Store of Langevin trajectories, keyed by substrate and thermostat.

    The first equilibrium proxy to ask pays for the relaxation, the equilibration and
    the MD; later proxies with matching settings read the stored frames for free.
    Mismatched settings get their own entry, so per-eval overrides stay correct -- they
    just stop saving calls.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple, Trajectory] = {}
        #: names of the proxies that reused someone else's trajectory, in order.
        self.reused_by: list[str] = []

    def stored_steps(self, atoms: Atoms, thermostat: Thermostat) -> int:
        """Sampling steps already available for these settings (0 if none)."""
        traj = self._entries.get((_fingerprint(atoms), thermostat))
        return 0 if traj is None else traj.n_steps

    def get(
        self,
        engine: ModelEngine,
        atoms: Atoms,
        thermostat: Thermostat,
        *,
        owner: str,
    ) -> tuple[Trajectory, bool]:
        """Return ``(trajectory, reused)`` for these settings, running one if needed."""
        key = (_fingerprint(atoms), thermostat)
        traj = self._entries.get(key)
        if traj is not None:
            self.reused_by.append(owner)
            return traj, True
        traj = Trajectory(engine, atoms, thermostat, owner=owner)
        self._entries[key] = traj
        return traj, False

    def stats(self) -> dict:
        """A JSON-friendly summary of what sharing cost and who reused it."""
        trajectories = list(self._entries.values())
        return {
            "n_trajectories": len(trajectories),
            "owners": [t.owner for t in trajectories],
            "reused_by": list(self.reused_by),
            "n_model_calls": int(sum(t.n_model_calls for t in trajectories)),
        }


def get_trajectory(
    engine: ModelEngine,
    atoms: Atoms,
    thermostat: Thermostat,
    *,
    owner: str,
    cache: TrajectoryCache | None = None,
) -> tuple[Trajectory, bool]:
    """Fetch (or run) the relaxed-and-equilibrated trajectory for these settings.

    Without a ``cache`` the proxy runs its own.  Returns ``(trajectory, reused)``; call
    :meth:`Trajectory.extend` to size the sampling run before reading
    :meth:`Trajectory.frames`.
    """
    if cache is None:
        return Trajectory(engine, atoms, thermostat, owner=owner), False
    return cache.get(engine, atoms, thermostat, owner=owner)


def stored_steps(cache: TrajectoryCache | None, atoms: Atoms, thermostat: Thermostat) -> int:
    """Sampling steps a proxy can get for free from ``cache`` (0 without one)."""
    return 0 if cache is None else cache.stored_steps(atoms, thermostat)
