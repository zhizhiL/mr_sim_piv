"""
constants.py — physical constants and the project unit convention.

Unit convention (from build_plan.md §2):
    * lengths    : millimetres  [mm]
    * velocities : mm / s
    * time       : seconds      [s]
    * fluid      : water at 20 degC
    * particle   : hydrogen gas bubbles

Physical formulae that are naturally written in SI (the particle response
time tau_p, the Froude number) are evaluated by converting the relevant
quantities to SI inside the helper functions; everything that touches the
PIV grid or the advection integrator stays in mm / s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Fluid: water at 20 degC
# --------------------------------------------------------------------------
RHO_F = 998.2          # density            [kg/m^3]
MU_F = 1.002e-3        # dynamic viscosity  [Pa.s]
NU_F_SI = MU_F / RHO_F  # kinematic visc.   [m^2/s]  ~= 1.004e-6
NU_F = NU_F_SI * 1.0e6  # kinematic visc.   [mm^2/s] ~= 1.004

# --------------------------------------------------------------------------
# Particle: hydrogen gas bubbles at 20 degC, ~1 atm
# --------------------------------------------------------------------------
RHO_P = 0.0838         # density            [kg/m^3]

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
G_SI = 9.81            # gravity            [m/s^2]
G_MM = G_SI * 1.0e3    # gravity            [mm/s^2]

DELTA_RHO = RHO_F - RHO_P   # density deficit of the bubble  [kg/m^3]

# Maxey-Riley density parameter beta = 3 rho_f / (rho_f + 2 rho_p).
# For a (near-massless) gas bubble this tends to 3.
BETA = 3.0 * RHO_F / (RHO_F + 2.0 * RHO_P)

# Ring reference radius, hard-coded by the build plan (§2 Stage E).
R0 = 20.0              # [mm]


@dataclass(frozen=True)
class Fluid:
    rho: float = RHO_F
    mu: float = MU_F
    nu_mm2s: float = NU_F


def particle_response_time(d_mm: float, rho_p: float = RHO_P, mu: float = MU_F) -> float:
    """tau_p = rho_p * d^2 / (18 mu)   (build_plan §2 Stage E), returned in seconds.

    ``d_mm`` is the bubble diameter in mm; it is converted to metres so the
    result is a physical time in seconds.
    """
    d_m = d_mm * 1.0e-3
    return rho_p * d_m * d_m / (18.0 * mu)


def stokes_number(d_mm, u_ring_mms: float, R0_mm: float = R0,
                  rho_p: float = RHO_P, mu: float = MU_F):
    """St = tau_p / tau_f with tau_f = R0 / U_ring  (build_plan §2 Stage E).

    Vectorised over ``d_mm``.  ``u_ring_mms`` is the ring velocity scale in
    mm/s; it is converted to m/s for the flow timescale so St is dimensionless.
    """
    import numpy as np
    d_mm = np.asarray(d_mm, dtype=float)
    tau_p = rho_p * (d_mm * 1.0e-3) ** 2 / (18.0 * mu)      # s
    tau_f = (R0_mm * 1.0e-3) / (u_ring_mms * 1.0e-3)         # s
    return tau_p / tau_f


def froude_number(a_eq_mm: float, u_ring_mms: float, d_mm: float,
                  rho: float = RHO_F, delta_rho: float = DELTA_RHO,
                  g: float = G_SI) -> float:
    """Fr = sqrt( rho * pi^2 * a_eq^2 * U_ring / (delta_rho * v_b * g) ).

    Implemented verbatim from build_plan.md §2 Stage F.  All quantities are
    converted to SI here.  ``v_b`` is the volume of a single spherical bubble
    of diameter ``d_mm``.

    NOTE: as written in the plan this group is not dimensionless (see the
    summary I gave you) — flagged here so the convention can be reconciled
    against the .tex writeup / the old solver before production use.
    """
    a_eq = a_eq_mm * 1.0e-3        # m
    U = u_ring_mms * 1.0e-3        # m/s
    d = d_mm * 1.0e-3             # m
    v_b = math.pi / 6.0 * d ** 3   # m^3
    return math.sqrt(rho * math.pi ** 2 * a_eq ** 2 * U / (delta_rho * v_b * g))
