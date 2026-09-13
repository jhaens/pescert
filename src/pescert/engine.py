"""Model engine: the only place that touches the MLIP.

:class:`ModelEngine` wraps any ASE calculator (or raw energy/forces/stress callables)
and exposes *only* physical quantities -- ``energy``, ``forces``, ``stress`` -- plus
what finite differences derive from them (``hessian``, ``hvp``, ``jvp``) and a
model-driven ``relax``.

This indirection is what makes every proxy architecture-independent: no metric may
reach inside the model, and second derivatives are matrix-free finite differences of
forces rather than autodiff.  Every single-point evaluation is counted (one wrapped
``Calculator.calculate`` == one model call) so each proxy can budget its cost.

Units are ASE's throughout: eV, Angstrom, eV/Angstrom, eV/Angstrom**3, K.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

__all__ = ["ModelEngine", "from_ase_calculator", "from_callable"]

#: Smallest finite-difference step any probe uses, in Angstrom (or in strain).  Below
#: this a displacement stops being a useful probe of the physics -- it is not a numerical
#: limit but a physical one -- so it acts as a floor on top of the precision estimate.
FD_MIN_STEP = 1e-3

#: How much worse than one rounding the model's noise is assumed to be.  A network
#: accumulates round-off over many operations, so its effective noise sits well above
#: ``finfo(dtype).eps``; this factor keeps the step on the safe side of that.
FD_NOISE_AMPLIFICATION = 10.0


class _CountingCalculator(Calculator):
    """Wrap an inner ASE calculator and count every forward pass.

    A *model call* is one invocation of :meth:`calculate`, i.e. one geometry sent to
    the model; reading energy and forces from the same geometry counts once, as a real
    MLIP forward pass yields both together.  ASE optimizers and MD integrators drive
    the calculator through this path, so relaxation and dynamics are counted too.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, inner: Calculator, counter: list[int], count: bool = True):
        super().__init__()
        self.inner = inner
        self.counter = counter
        self._count = count
        #: dtype the wrapped model last returned, before this class widens it to float64.
        #: It is the only evidence of the model's working precision, and finite-difference
        #: steps have to respect it -- see :meth:`ModelEngine.fd_step`.
        self.observed_dtype: np.dtype | None = None

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
            raw = np.asarray(work.get_forces())
            self._observe(raw.dtype)
            results["forces"] = raw.astype(float)
        if "stress" in properties:
            raw = np.asarray(work.get_stress(voigt=True))
            self._observe(raw.dtype)
            results["stress"] = raw.astype(float)
        self.results = results

    def _observe(self, dtype) -> None:
        """Remember the narrowest float dtype seen; a mixed model is as noisy as its worst."""
        if not np.issubdtype(dtype, np.floating):
            return
        if self.observed_dtype is None or np.finfo(dtype).eps > np.finfo(self.observed_dtype).eps:
            self.observed_dtype = np.dtype(dtype)


class _CallableCalculator(Calculator):
    """Adapt raw ``energy_fn`` / ``forces_fn`` / ``stress_fn`` callables to ASE.

    The raw-checkpoint path: a model with no ASE calculator can still be certified.
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

    Passing the class ``LennardJones`` rather than ``LennardJones()`` is instantiated
    for you instead of failing deep inside ASE.
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
    precision:
        The model's working float precision -- ``"float32"``, ``"float64"``, or any numpy
        float dtype.  It sets the finite-difference step (see :meth:`fd_step`).  Left
        ``None`` it is inferred from the dtype the model returns, which is only known
        once a call has been made; declare it explicitly when you need the steps to be
        identical regardless of the order probes run in.
    """

    def __init__(self, calculator: Calculator, *, count: bool = True, precision=None):
        calculator = _coerce_calculator(calculator)
        self._counter: list[int] = [0]
        self._ccalc = _CountingCalculator(calculator, self._counter, count=count)
        self._inner = calculator
        self._declared_dtype = np.dtype(precision) if precision is not None else None

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
    def jvp(self, atoms: Atoms, v: np.ndarray, *, eps: float | None = None) -> np.ndarray:
        """Jacobian--vector product ``J @ v`` with ``J = dF/dR`` (two model calls).

        Central difference of forces along the *unit* direction ``v``, so the physical
        step is exactly ``eps`` Angstrom regardless of ``||v||``.  ``eps=None`` takes the
        precision-aware default from :meth:`fd_step`.
        """
        eps = self.fd_step(order=1, accuracy=2) if eps is None else eps
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

    def hvp(self, atoms: Atoms, v: np.ndarray, *, eps: float | None = None) -> np.ndarray:
        """Hessian--vector product ``H @ v`` with ``H = d2E/dR2 = -dF/dR``.

        Matrix-free via central differences of forces (two model calls).
        """
        return -self.jvp(atoms, v, eps=eps)

    def hessian(
        self,
        atoms: Atoms,
        *,
        eps: float | None = None,
        method: str = "fd_forces",
        symmetrize: bool = False,
    ) -> np.ndarray:
        """Dense Hessian ``H = d2E/dR2``, shape ``(3N, 3N)`` in eV/Angstrom**2.

        ``method="fd_forces"`` (default) differences the forces: ``6N`` model calls.
        ``method="fd_energy"`` is a slower fallback using second differences of the
        energy (``~2 (3N)^2`` calls).  ``symmetrize`` returns ``(H + H.T) / 2``; the
        finite-difference Hessian is symmetric only up to discretization error.
        """
        n = len(atoms)
        ndof = 3 * n
        r0 = atoms.get_positions()
        if eps is None:
            # fd_forces differences forces (a first derivative); fd_energy takes a second
            # difference of the energy, which tolerates round-off less well.
            eps = self.fd_step(order=1 if method == "fd_forces" else 2, accuracy=2)
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

        Returns a fresh :class:`ase.Atoms`; the converged residual is stored on
        ``info['relax_fmax']``.
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

    # -- working precision and finite-difference steps ---------------------
    @property
    def dtype(self) -> np.dtype:
        """The model's working float dtype: declared if given, else observed, else f64."""
        if self._declared_dtype is not None:
            return self._declared_dtype
        return self._ccalc.observed_dtype or np.dtype(np.float64)

    @property
    def precision(self) -> str:
        """``"float32"`` / ``"float64"`` -- the name behind :attr:`dtype`."""
        return str(self.dtype)

    @property
    def precision_is_known(self) -> bool:
        """Whether the precision is declared or observed rather than merely assumed."""
        return self._declared_dtype is not None or self._ccalc.observed_dtype is not None

    def detect_precision(self, atoms: Atoms) -> str:
        """Observe the model's precision by evaluating forces once (one model call)."""
        if self._declared_dtype is None and self._ccalc.observed_dtype is None:
            self.forces(atoms)
        return self.precision

    def fd_step(self, *, order: int = 1, accuracy: int = 2, scale: float = 1.0) -> float:
        """A finite-difference step the model's precision can actually support.

        A central difference trades truncation error, ``O(h**accuracy)``, against
        round-off, ``O(noise / h**order)``.  Balancing them puts the optimum at
        ``h ~ noise ** (1 / (order + accuracy))``, which for a float32 model is roughly
        ten times the float64 value -- differencing below it measures round-off rather
        than the potential energy surface.

        ``order`` is the derivative being taken (1 for a gradient or a Hessian from
        differenced forces, 2 for a curvature from second differences of the energy) and
        ``accuracy`` the order of the stencil.  The result is never smaller than
        :data:`FD_MIN_STEP`, so a float64 model keeps the well-tested default.
        """
        noise = FD_NOISE_AMPLIFICATION * float(np.finfo(self.dtype).eps)
        h = scale * noise ** (1.0 / (order + accuracy))
        return max(FD_MIN_STEP * scale, h)

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
