"""Analytic ASE calculators used as deterministic ground truth in the test suite.

No DFT, no downloads.  Clean references (smooth, conservative, symmetric) must pass
every "=0"/"=1" check; the deliberately broken wrappers must trip exactly the proxies
the spec's sensitivity matrix predicts.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones


def clean_lj(rc: float = 100.0) -> LennardJones:
    """A smooth, conservative LJ with a cutoff far beyond any interaction (no kink)."""
    return LennardJones(epsilon=1.0, sigma=1.0, rc=rc, smooth=False)


class AnchoredHarmonic(Calculator):
    """E = 1/2 k sum_i |R_i - a_i|^2.  Hessian == k I everywhere (engine ground truth)."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, k: float = 1.0, anchors: np.ndarray | None = None):
        super().__init__()
        self.k = k
        self.anchors = None if anchors is None else np.asarray(anchors, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions()
        anch = self.anchors if self.anchors is not None else np.zeros_like(pos)
        d = pos - anch
        self.results["energy"] = 0.5 * self.k * float(np.sum(d * d))
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -self.k * d


class PairHarmonic(Calculator):
    """E = 1/2 k sum_{i<j} (r_ij - r0)^2 over all pairs.

    Translation-, rotation- and permutation-invariant and conservative: a clean
    reference for the symmetry / zero-mode / equipartition proxies.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, k: float = 1.0, r0: float = 1.0):
        super().__init__()
        self.k = k
        self.r0 = r0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions()
        n = len(pos)
        e = 0.0
        f = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                rij = pos[i] - pos[j]
                d = np.linalg.norm(rij)
                if d < 1e-12:
                    continue
                e += 0.5 * self.k * (d - self.r0) ** 2
                g = self.k * (d - self.r0) * rij / d  # dE/dR_i
                f[i] -= g
                f[j] += g
        self.results["energy"] = float(e)
        self.results["free_energy"] = float(e)
        self.results["forces"] = f


class _Wrapper(Calculator):
    """Base for calculators that wrap an inner calculator."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, inner: Calculator):
        super().__init__()
        self.inner = inner

    def _inner_results(self, atoms, properties):
        work = atoms.copy()
        work.calc = self.inner
        e = float(work.get_potential_energy())
        f = np.asarray(work.get_forces(), dtype=float)
        s = None
        if "stress" in properties:
            s = np.asarray(work.get_stress(voigt=True), dtype=float)
        return e, f, s


class NonConservative(_Wrapper):
    """Add a curl-type force ``strength * omega x (R_i - R_com)`` not derivable from any E.

    Translation-invariant (depends only on positions relative to the centre of mass)
    and trace-free (does not change Tr H), so it isolates *non-conservativeness*:
    energy and stress are untouched, forces are not the gradient of any scalar.
    """

    def __init__(self, inner: Calculator, strength: float = 0.3, axis=(0.0, 0.0, 1.0)):
        super().__init__(inner)
        self.strength = strength
        self.axis = np.asarray(axis, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f, s = self._inner_results(atoms, properties)
        pos = atoms.get_positions()
        com = pos.mean(0)
        omega = self.strength * self.axis
        extra = np.cross(omega, pos - com)
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = f + extra
        if s is not None:
            self.results["stress"] = s


class Kinked(_Wrapper):
    """Add a conservative but non-smooth term A * sum_{i<j} |r_ij - d0|.

    Energy stays continuous and forces stay *consistent* with it (so conservativeness
    is unaffected), but the force has a step at ``r_ij = d0``: a roughness/discontinuity
    defect that should be caught by smoothness, reversibility and equipartition spread.
    """

    def __init__(self, inner: Calculator, amplitude: float = 0.5, d0: float = 1.25):
        super().__init__(inner)
        self.amplitude = amplitude
        self.d0 = d0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f, s = self._inner_results(atoms, properties)
        pos = atoms.get_positions()
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                rij = pos[i] - pos[j]
                d = np.linalg.norm(rij)
                if d < 1e-12:
                    continue
                e += self.amplitude * abs(d - self.d0)
                g = self.amplitude * np.sign(d - self.d0) * rij / d
                f[i] -= g
                f[j] += g
        self.results["energy"] = float(e)
        self.results["free_energy"] = float(e)
        self.results["forces"] = f
        if s is not None:
            self.results["stress"] = s


class SymmetryBreaking(_Wrapper):
    """Add a conservative term c * sum_i (R_i . n_hat)^2 that depends on absolute orientation.

    Breaks translational and rotational invariance (and reflection through planes not
    containing ``n_hat``) while staying conservative, so it should be caught by
    equivariance, NEW-1 (both zero-mode residuals) and NEW-3(i), but not by
    conservativeness.
    """

    def __init__(self, inner: Calculator, c: float = 0.2, n_hat=(1.0, 1.0, 1.0)):
        super().__init__(inner)
        self.c = c
        nh = np.asarray(n_hat, dtype=float)
        self.n_hat = nh / np.linalg.norm(nh)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f, s = self._inner_results(atoms, properties)
        pos = atoms.get_positions()
        proj = pos @ self.n_hat
        e += self.c * float(np.sum(proj**2))
        f -= 2.0 * self.c * np.outer(proj, self.n_hat)
        self.results["energy"] = float(e)
        self.results["free_energy"] = float(e)
        self.results["forces"] = f
        if s is not None:
            self.results["stress"] = s


class BadStress(_Wrapper):
    """Return a wrong stress (scaled by ``factor``) while energy and forces stay correct.

    Isolates a broken/approximate stress head: only OOB-2 should fire.
    """

    def __init__(self, inner: Calculator, factor: float = 1.4):
        super().__init__(inner)
        self.factor = factor

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f, s = self._inner_results(atoms, properties)
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = f
        if s is not None:  # only when stress was actually requested (periodic path)
            self.results["stress"] = self.factor * s
