"""Analytic reference potentials: the suite's own control.

Every identity the proxies test holds *by construction* for a closed-form pair potential
-- one analytic energy, forces as its exact gradient, invariance under E(3) because it
depends on interatomic distances alone.  Running one through the suite therefore does not
measure the potential: it measures the suite, and whatever score it fails to reach is the
floor set by finite differences and finite MD sampling rather than a defect of a model.

:class:`ElementLennardJones` is that control.  Plain
:class:`ase.calculators.lj.LennardJones` carries one ``sigma``/``epsilon`` pair for all
chemistry, which puts a fixed pair minimum against substrates whose spacing follows each
element's covalent radius -- carbon lands deep on the repulsive wall while sodium sits far
out in the tail, and the two are no longer measuring the same thing.  Here the parameters
follow the species instead, so every element is probed at the same point of the same
curve.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.lj import LennardJones

from .substrates import _nn_distance

__all__ = ["ElementLennardJones", "COHESIVE_ENERGY_EV", "lj_parameters"]

#: Rounded experimental cohesive energies in eV/atom (Kittel, *Introduction to Solid
#: State Physics*, table of cohesive energies).  They set only the *energy scale* of the
#: reference potential -- a Lennard-Jones fit to a covalent or metallic solid is not a
#: physical model of it, and nothing in the suite depends on the value being right.  What
#: they buy is an energy of the correct order for each element, so no substrate is probed
#: with forces near round-off while another is probed at electronvolt scale.
COHESIVE_ENERGY_EV = {
    "Li": 1.63, "Be": 3.32, "B": 5.81, "C": 7.37, "N": 4.92, "O": 2.60, "F": 0.84,
    "Ne": 0.020, "Na": 1.113, "Mg": 1.51, "Al": 3.39, "Si": 4.63, "P": 3.43, "S": 2.85,
    "Cl": 1.40, "Ar": 0.080, "K": 0.934, "Ca": 1.84, "Ti": 4.85, "Fe": 4.28, "Ni": 4.44,
    "Cu": 3.49, "Zn": 1.35, "Ge": 3.85, "Ag": 2.95, "Au": 3.81, "Pt": 5.84, "W": 8.90,
}

#: Lattice sum of the Lennard-Jones fcc crystal: at its own minimum the energy per atom
#: is ``-8.610 * epsilon`` (Kittel again).  Dividing a cohesive energy by it turns
#: "how deeply this element binds in a solid" into "how deep one pair well has to be".
LJ_FCC_LATTICE_SUM = 8.610

#: Well depth for an element with no tabulated cohesive energy: the middle of the range
#: spanned by the table, i.e. a typical solid.
DEFAULT_COHESIVE_EV = 3.0

#: ``r_min = 2**(1/6) * sigma`` for the Lennard-Jones pair potential.
_RMIN_OVER_SIGMA = 2.0 ** (1.0 / 6.0)


def lj_parameters(symbol: str, *, scale: float = 1.0) -> tuple[float, float]:
    """Return ``(epsilon, sigma)`` in (eV, Angstrom) for one element.

    ``sigma`` is fixed by putting the pair minimum exactly at the nearest-neighbour
    distance the substrate builders use -- :func:`pescert.substrates._nn_distance`, i.e.
    the element's experimental reference crystal -- so the cluster, trimer and crystal of
    every element start at the bottom of the well rather than somewhere accidental on it.
    ``epsilon`` follows from the cohesive energy through the fcc lattice sum.
    """
    sigma = _nn_distance(symbol, scale) / _RMIN_OVER_SIGMA
    epsilon = COHESIVE_ENERGY_EV.get(symbol, DEFAULT_COHESIVE_EV) / LJ_FCC_LATTICE_SUM
    return float(epsilon), float(sigma)


class ElementLennardJones(LennardJones):
    """Lennard-Jones whose ``sigma`` and ``epsilon`` follow the species being evaluated.

    The parameters depend on the *composition* only, never on the positions, so within
    any one evaluation this is an ordinary Lennard-Jones potential: exactly conservative,
    exactly E(3)-invariant, smooth to all orders below the cutoff.  Every proxy therefore
    still has its exact target here, which is the whole point of the control.

    The substrates are single-element by construction (:func:`pescert.substrates.trimer`
    even refuses a mixed one), but a mixed system is not rejected: its parameters are the
    Lorentz-Berthelot combination -- arithmetic mean of the ``sigma``, geometric mean of
    the ``epsilon`` -- applied as one effective pair, which is an approximation of a
    proper per-pair mixture and is documented as such rather than silently exact.

    Parameters
    ----------
    scale:
        Multiplies the covalent radius, exactly as the substrate builders' ``scale``
        does.  Keep the two equal to keep the minimum on the substrate spacing.
    rc_over_sigma, ro_over_rc:
        Cutoff and smoothing onset, in units of the element's own ``sigma`` and of the
        resulting cutoff.  ``smooth=True`` is the default here: a shifted-but-kinked
        cutoff would show up in the smoothness and MD probes as a defect of the cutoff
        rather than a floor of the suite.
    """

    def __init__(
        self,
        *,
        scale: float = 1.0,
        rc_over_sigma: float = 3.0,
        ro_over_rc: float = 0.66,
        smooth: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("smooth", smooth)
        # an explicitly given cutoff is honoured as an absolute distance and kept fixed
        # across species; otherwise the cutoff follows each element's own sigma
        self._fixed_rc = kwargs.get("rc")
        super().__init__(**kwargs)
        self.scale = float(scale)
        self.rc_over_sigma = float(rc_over_sigma)
        self.ro_over_rc = float(ro_over_rc)

    def parameters_for(self, atoms) -> tuple[float, float]:
        """``(epsilon, sigma)`` for the species in ``atoms`` (Lorentz-Berthelot if mixed)."""
        symbols = sorted(set(atoms.get_chemical_symbols()))
        pairs = [lj_parameters(s, scale=self.scale) for s in symbols]
        if len(pairs) == 1:
            return pairs[0]
        epsilon = float(np.prod([e for e, _ in pairs]) ** (1.0 / len(pairs)))
        sigma = float(np.mean([s for _, s in pairs]))
        return epsilon, sigma

    def calculate(self, atoms=None, properties=None, system_changes=None):
        target = atoms if atoms is not None else self.atoms
        epsilon, sigma = self.parameters_for(target)
        if (epsilon, sigma) != (self.parameters.epsilon, self.parameters.sigma):
            # assign in place rather than through set(): set() clears self.results, and
            # the neighbour list has to be rebuilt anyway because the cutoff moved
            self.parameters.epsilon = epsilon
            self.parameters.sigma = sigma
            if self._fixed_rc is None:
                self.parameters.rc = self.rc_over_sigma * sigma
                self.parameters.ro = self.ro_over_rc * self.parameters.rc
            self.nl = None
        kwargs = {} if system_changes is None else {"system_changes": system_changes}
        super().calculate(atoms, properties, **kwargs)
