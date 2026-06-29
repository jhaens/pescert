"""Model engine: the only place that touches the MLIP.

The :class:`ModelEngine` wraps any ASE calculator (or a raw energy/forces/stress
callable) and exposes *only* physical quantities -- ``energy``, ``forces``,
``stress`` -- plus utilities derived from them by finite differences
(``hessian``, ``hvp``, ``jvp``) and a model-driven ``relax``.

This indirection enforces the package's non-negotiable constraint: every proxy is
**architecture-independent**.  No metric is allowed to reach inside the model; it
may only ask the engine for energy/forces/stress.  Hessians and Jacobian--vector
products are matrix-free finite differences of forces -- never autodiff into the
model, never format conversion.

Every underlying single-point model evaluation is counted (one wrapped
``Calculator.calculate`` call == one model call), so each proxy can report and
budget its cost.  Units are ASE units throughout: energy in eV, length in Angstrom,
forces in eV/Angstrom, stress in eV/Angstrom**3, temperature in K
(via :data:`ase.units.kB`).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

__all__ = ["ModelEngine", "from_ase_calculator", "from_callable"]


class _CountingCalculator(Calculator):
    """Wrap an inner ASE calculator and count every forward pass.

    A *model call* is one invocation of :meth:`calculate` (one geometry sent to
    the model).  Reading energy and forces from the *same* geometry triggers a
    single ``calculate`` and therefore counts once -- mirroring a real MLIP
    forward pass that yields energy and forces together.  ASE optimizers and MD
    integrators drive the calculator through this same path, so relaxation and
    dynamics are counted automatically.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, inner: Calculator, counter: list[int], count: bool = True):
        super().__init__()
        self.inner = inner
        self.counter = counter
        self._count = count

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self._count:
            self.counter[0] += 1
        work = atoms.copy()
        work.calc = self.inner
        results: dict[str, object] = {}
        results["energy"] = float(work.get_potential_energy())
        results["free_energy"] = results["energy"]
        if "forces" in properties:
            results["forces"] = np.asarray(work.get_forces(), dtype=float)
        if "stress" in properties:
            results["stress"] = np.asarray(work.get_stress(voigt=True), dtype=float)
        self.results = results


class _CallableCalculator(Calculator):
    """Adapt raw ``energy_fn`` / ``forces_fn`` / ``stress_fn`` callables to ASE.

    Each callable receives an :class:`ase.Atoms` and returns a float, an ``(N, 3)``
    array, and a ``(3, 3)`` (or 6-vector Voigt) array respectively.  This is the
    raw-checkpoint path: a model with no ASE calculator can still be certified.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        energy_fn: Callable[[Atoms], float],
        forces_fn: Callable[[Atoms], np.ndarray],
        stress_fn: Callable[[Atoms], np.ndarray] | None = None,
    ):
        super().__init__()
        self._energy_fn = energy_fn
        self._forces_fn = forces_fn
        self._stress_fn = stress_fn

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        results: dict[str, object] = {}
        results["energy"] = float(self._energy_fn(atoms))
        results["free_energy"] = results["energy"]
        if "forces" in properties:
            results["forces"] = np.asarray(self._forces_fn(atoms), dtype=float)
        if "stress" in properties:
            if self._stress_fn is None:
                raise NotImplementedError("No stress_fn was provided to from_callable().")
            s = np.asarray(self._stress_fn(atoms), dtype=float)
            if s.shape == (3, 3):
                s = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
            results["stress"] = s
        self.results = results


def _coerce_calculator(calculator) -> Calculator:
    """Accept a Calculator instance, a Calculator subclass, or a zero-argument factory.

    This makes the Python API forgiving in the same way the CLI is: passing the class
    ``LennardJones`` (rather than ``LennardJones()``) is instantiated for you instead of
    failing deep inside ASE.
    """
    if isinstance(calculator, Calculator):
        return calculator
    if isinstance(calculator, type) or callable(calculator):
        try:
            instance = calculator()
        except TypeError as exc:
            raise TypeError(
                "ModelEngine received a calculator class/factory that needs arguments; "
                "pass a constructed instance instead, e.g. "
                "from_ase_calculator(LennardJones(epsilon=1.0))."
            ) from exc
        if isinstance(instance, Calculator):
            return instance
    raise TypeError(
        "Expected an ASE Calculator instance, a Calculator subclass, or a zero-argument "
        f"factory returning one; got {calculator!r}."
    )


class ModelEngine:
    """Wrap a model and expose only physical quantities plus derived utilities.

    Parameters
    ----------
    calculator:
        An ASE :class:`~ase.calculators.calculator.Calculator` instance, a Calculator
        subclass, or a zero-argument factory returning one (the latter two are
        instantiated for you).
    count:
        If ``True`` (default) every underlying single-point evaluation is counted
        and exposed through :attr:`n_calls`.
    """

    def __init__(self, calculator: Calculator, *, count: bool = True):
        calculator = _coerce_calculator(calculator)
        self._counter: list[int] = [0]
        self._ccalc = _CountingCalculator(calculator, self._counter, count=count)
        self._inner = calculator

    # -- attachment ---------------------------------------------------------
    @property
    def calc(self) -> Calculator:
        """The counting calculator (attach this to drive optimizers / MD)."""
        return self._ccalc

    def attach(self, atoms: Atoms) -> Atoms:
        """Return a copy of ``atoms`` with the counting calculator attached."""
        work = atoms.copy()
        work.calc = self._ccalc
        return work

    # -- primitive physical quantities -------------------------------------
    def energy(self, atoms: Atoms) -> float:
        """Potential energy in eV (one model call)."""
        return float(self.attach(atoms).get_potential_energy())

    def forces(self, atoms: Atoms) -> np.ndarray:
        """Forces, shape ``(N, 3)`` in eV/Angstrom (one model call)."""
        return np.asarray(self.attach(atoms).get_forces(), dtype=float)

    def stress(self, atoms: Atoms) -> np.ndarray:
        """Stress tensor, shape ``(3, 3)`` in eV/Angstrom**3 (one model call).

        Raises :class:`NotImplementedError` if the model does not provide stress.
        """
        work = self.attach(atoms)
        try:
            voigt = np.asarray(work.get_stress(voigt=True), dtype=float)
        except Exception as exc:  # noqa: BLE001 - normalize to a clear contract
            raise NotImplementedError(f"Model does not provide stress: {exc}") from exc
        return voigt_to_tensor(voigt)

    def stress_voigt(self, atoms: Atoms) -> np.ndarray:
        """Stress as a 6-vector Voigt ``[xx, yy, zz, yz, xz, xy]`` (one model call)."""
        return tensor_to_voigt(self.stress(atoms))

    def energy_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Energy and forces from a *single* model call (one forward pass)."""
        work = self.attach(atoms)
        f = np.asarray(work.get_forces(), dtype=float)
        e = float(work.get_potential_energy())
        return e, f

    def has_stress(self, atoms: Atoms) -> bool:
        """Return whether the model can produce a stress for ``atoms``."""
        try:
            self.stress(atoms)
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- derived: matrix-free second derivatives ---------------------------
    def jvp(self, atoms: Atoms, v: np.ndarray, *, eps: float = 1e-3) -> np.ndarray:
        """Jacobian--vector product ``J @ v`` with ``J = dF/dR``.

        Central finite difference of forces along the *unit* direction ``v`` so the
        physical step is exactly ``eps`` Angstrom regardless of ``||v||`` (two model
        calls).  Note ``H = -J`` for a conservative model; see :meth:`hvp`.
        """
        v = np.asarray(v, dtype=float).reshape(-1)
        nrm = np.linalg.norm(v)
        if nrm == 0.0:
            return np.zeros_like(v)
        u = (v / nrm).reshape(len(atoms), 3)
        plus = atoms.copy()
        plus.set_positions(atoms.get_positions() + eps * u)
        minus = atoms.copy()
        minus.set_positions(atoms.get_positions() - eps * u)
        df = self.forces(plus) - self.forces(minus)
        return nrm * df.reshape(-1) / (2.0 * eps)

    def hvp(self, atoms: Atoms, v: np.ndarray, *, eps: float = 1e-3) -> np.ndarray:
        """Hessian--vector product ``H @ v`` with ``H = d2E/dR2 = -dF/dR``.

        Matrix-free via central differences of forces (two model calls).
        """
        return -self.jvp(atoms, v, eps=eps)

    def hessian(
        self,
        atoms: Atoms,
        *,
        eps: float = 1e-3,
        method: str = "fd_forces",
        symmetrize: bool = False,
    ) -> np.ndarray:
        """Dense Hessian ``H = d2E/dR2``, shape ``(3N, 3N)`` in eV/Angstrom**2.

        ``method="fd_forces"`` (default, recommended): column ``j`` is
        ``-(F(R + eps e_j) - F(R - eps e_j)) / (2 eps)`` flattened -> ``6N`` model
        calls.  ``method="fd_energy"`` is a slower fallback using second
        differences of the energy (``~2 (3N)^2`` calls).  ``symmetrize`` returns
        ``(H + H.T) / 2`` (the finite-difference Hessian is symmetric only up to
        discretization error).
        """
        n = len(atoms)
        ndof = 3 * n
        r0 = atoms.get_positions()
        if method == "fd_forces":
            h = np.empty((ndof, ndof))
            for j in range(ndof):
                disp = np.zeros(ndof)
                disp[j] = eps
                plus = atoms.copy()
                plus.set_positions(r0 + disp.reshape(n, 3))
                minus = atoms.copy()
                minus.set_positions(r0 - disp.reshape(n, 3))
                h[:, j] = -(self.forces(plus) - self.forces(minus)).reshape(-1) / (2.0 * eps)
        elif method == "fd_energy":
            h = _hessian_from_energy(self, atoms, eps)
        else:
            raise ValueError(f"unknown hessian method {method!r}")
        if symmetrize:
            h = 0.5 * (h + h.T)
        return h

    def relax(
        self,
        atoms: Atoms,
        *,
        fmax: float = 1e-3,
        steps: int = 300,
        optimizer: str = "BFGS",
    ) -> Atoms:
        """Relax to the *model's own* stationary point (no DFT, no reference).

        Returns a fresh :class:`ase.Atoms` at the relaxed positions; the converged
        residual ``fmax`` is stored on ``atoms.info['relax_fmax']``.
        """
        from ase.optimize import BFGS, FIRE, LBFGS

        opt_map = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}
        if optimizer not in opt_map:
            raise ValueError(f"unknown optimizer {optimizer!r}")
        work = self.attach(atoms)
        dyn = opt_map[optimizer](work, logfile=None)
        dyn.run(fmax=fmax, steps=steps)
        out = atoms.copy()
        out.set_positions(work.get_positions())
        out.info["relax_fmax"] = float(np.linalg.norm(work.get_forces(), axis=1).max())
        return out

    # -- counter -----------------------------------------------------------
    @property
    def n_calls(self) -> int:
        """Number of model single-point evaluations performed so far."""
        return self._counter[0]

    def reset_counter(self) -> None:
        """Reset the model-call counter to zero."""
        self._counter[0] = 0


def _hessian_from_energy(engine: ModelEngine, atoms: Atoms, eps: float) -> np.ndarray:
    n = len(atoms)
    ndof = 3 * n
    r0 = atoms.get_positions()
    h = np.empty((ndof, ndof))
    e0 = engine.energy(atoms)

    def e_at(disp):
        a = atoms.copy()
        a.set_positions(r0 + disp.reshape(n, 3))
        return engine.energy(a)

    diag = {}
    for i in range(ndof):
        d = np.zeros(ndof)
        d[i] = eps
        diag[i] = (e_at(d), e_at(-d))
    for i in range(ndof):
        ep, em = diag[i]
        h[i, i] = (ep - 2 * e0 + em) / eps**2
        for j in range(i + 1, ndof):
            dpp = np.zeros(ndof)
            dpp[i] = eps
            dpp[j] = eps
            dmm = np.zeros(ndof)
            dmm[i] = -eps
            dmm[j] = -eps
            epp = e_at(dpp)
            emm = e_at(dmm)
            ejp = diag[j][0]
            ejm = diag[j][1]
            eip = diag[i][0]
            eim = diag[i][1]
            h[i, j] = (epp - eip - ejp + 2 * e0 - eim - ejm + emm) / (2 * eps**2)
            h[j, i] = h[i, j]
    return h


# -- Voigt helpers ---------------------------------------------------------
def voigt_to_tensor(voigt: np.ndarray) -> np.ndarray:
    """Convert ASE Voigt stress ``[xx, yy, zz, yz, xz, xy]`` to a ``(3, 3)`` tensor."""
    xx, yy, zz, yz, xz, xy = voigt
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def tensor_to_voigt(tensor: np.ndarray) -> np.ndarray:
    """Convert a symmetric ``(3, 3)`` stress tensor to ASE Voigt order."""
    t = np.asarray(tensor)
    return np.array([t[0, 0], t[1, 1], t[2, 2], t[1, 2], t[0, 2], t[0, 1]])


def from_ase_calculator(calc: Calculator, **kw) -> ModelEngine:
    """Build a :class:`ModelEngine` from an ASE calculator."""
    return ModelEngine(calc, **kw)


def from_callable(
    energy_fn: Callable[[Atoms], float],
    forces_fn: Callable[[Atoms], np.ndarray],
    stress_fn: Callable[[Atoms], np.ndarray] | None = None,
    **kw,
) -> ModelEngine:
    """Build a :class:`ModelEngine` from raw energy/forces(/stress) callables.

    Each callable takes an :class:`ase.Atoms`.  ``energy_fn`` returns a float (eV),
    ``forces_fn`` an ``(N, 3)`` array (eV/Angstrom), and the optional ``stress_fn`` a
    ``(3, 3)`` tensor or 6-vector Voigt (eV/Angstrom**3).
    """
    return ModelEngine(_CallableCalculator(energy_fn, forces_fn, stress_fn), **kw)
