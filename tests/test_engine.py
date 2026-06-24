"""Engine correctness: Hessian/hvp/jvp vs the analytic harmonic Hessian, counting, relax."""

from __future__ import annotations

import numpy as np
import pytest
from _potentials import AnchoredHarmonic, clean_lj
from ase import Atoms

from pescert import from_ase_calculator, from_callable
from pescert.engine import tensor_to_voigt, voigt_to_tensor


@pytest.fixture
def harmonic():
    rng = np.random.default_rng(0)
    r0 = rng.normal(size=(4, 3))
    k = 2.5
    atoms = Atoms("Ar4", positions=r0 + 0.05 * rng.normal(size=(4, 3)))
    engine = from_ase_calculator(AnchoredHarmonic(k=k, anchors=r0))
    return engine, atoms, k


def test_hessian_matches_analytic(harmonic):
    engine, atoms, k = harmonic
    h = engine.hessian(atoms, eps=1e-3)
    assert np.allclose(h, k * np.eye(12), atol=1e-5)


def test_hessian_fd_energy_fallback(harmonic):
    engine, atoms, k = harmonic
    h = engine.hessian(atoms, eps=1e-3, method="fd_energy")
    assert np.allclose(h, k * np.eye(12), atol=1e-2)


def test_hvp_and_jvp_signs(harmonic):
    engine, atoms, k = harmonic
    v = np.random.default_rng(1).normal(size=12)
    assert np.allclose(engine.hvp(atoms, v), k * v, atol=1e-5)  # H = k I
    assert np.allclose(engine.jvp(atoms, v), -k * v, atol=1e-5)  # J = -H


def test_call_counting(harmonic):
    engine, atoms, _ = harmonic
    engine.reset_counter()
    engine.energy(atoms)
    engine.forces(atoms)
    assert engine.n_calls == 2
    engine.reset_counter()
    engine.hessian(atoms, eps=1e-3)  # fd_forces => 6N calls
    assert engine.n_calls == 6 * len(atoms)
    engine.reset_counter()
    engine.hvp(atoms, np.ones(12))  # central diff => 2 calls
    assert engine.n_calls == 2


def test_energy_forces_single_call(harmonic):
    engine, atoms, _ = harmonic
    engine.reset_counter()
    e, f = engine.energy_forces(atoms)
    assert engine.n_calls == 1  # one forward pass yields both
    assert f.shape == (4, 3)


def test_relax_finds_minimum(harmonic):
    engine, atoms, _ = harmonic
    relaxed = engine.relax(atoms, fmax=1e-6)
    assert relaxed.info["relax_fmax"] < 1e-5
    # the anchored harmonic minimum is the anchor geometry
    assert np.linalg.norm(engine.forces(relaxed)) < 1e-4


def test_voigt_roundtrip():
    t = np.array([[1.0, 0.4, 0.5], [0.4, 2.0, 0.6], [0.5, 0.6, 3.0]])
    assert np.allclose(voigt_to_tensor(tensor_to_voigt(t)), t)


def test_stress_unsupported_raises():
    # Morse-like analytic with no stress -> NotImplementedError on a periodic cell
    engine = from_ase_calculator(AnchoredHarmonic(k=1.0))
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(NotImplementedError):
        engine.stress(atoms)
    assert engine.has_stress(atoms) is False


def test_from_callable_path():
    def efn(a):
        return float(0.5 * np.sum(a.get_positions() ** 2))

    def ffn(a):
        return -a.get_positions()

    engine = from_callable(efn, ffn)
    atoms = Atoms("Ar3", positions=np.eye(3))
    assert np.isclose(engine.energy(atoms), 1.5)
    assert np.allclose(engine.forces(atoms), -np.eye(3))


def test_lj_stress_is_symmetric():
    from ase.lattice.cubic import FaceCenteredCubic

    b = FaceCenteredCubic("Cu", size=(1, 1, 1))
    b.set_chemical_symbols(["Ar"] * len(b))
    b.set_cell(b.cell * (1.12 * np.sqrt(2) / b.cell[0, 0]), scale_atoms=True)
    engine = from_ase_calculator(clean_lj(rc=6.0))
    s = engine.stress(b)
    assert s.shape == (3, 3)
    assert np.allclose(s, s.T)
