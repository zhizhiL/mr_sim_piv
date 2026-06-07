"""
constants.py — physical constants + the dimensionless-scaling helpers.

Unit convention:
    * dimensional ingestion : mm, mm/s, s  (PIV grid, escape-CSV diameters)
    * fluid  : water at 20 degC
    * particle : hydrogen gas bubbles

Everything downstream of ``nondimensionalize()`` (see ``nondimensional.py``) is
DIMENSIONLESS, matching the reused solver ``advect_bubbles_3D_eval.py``.  The
St / Fr / R groups below follow ``dimension_conversion.md`` exactly:

    L_f = R0,  U_f = U_ring,  tau_f = R0 / U_ring
    R   = 2 rho_f / (2 rho_p + rho_f)            (-> 2 for a bubble)
    St  = d_b^2 * U_ring / (18 nu R0)            (density-free tau_p)
    Fr  = U_ring * sqrt(rho_f / (d_rho g R0))    (section 4a, per-ring)

The user's a_eq/v_b Froude form (section 4b) is NOT dimensionless and is kept
disabled in :func:`froude_number_proposed`.
"""

from __future__ import annotations

import math
import warnings

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
DELTA_RHO = RHO_F - RHO_P   # density deficit  [kg/m^3]

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
G_SI = 9.81            # gravity            [m/s^2]
R0 = 20.0              # ring radius L_f    [mm]  (hard-coded by build_plan §2E)

# Maxey-Riley density parameter used by the solver's R*(u-v)/St drag term.
# R = 2 rho_f / (2 rho_p + rho_f) -> 2 for a (near-massless) gas bubble.
R_BUBBLE = 2.0 * RHO_F / (2.0 * RHO_P + RHO_F)


def particle_response_time(d_mm, nu_si: float = NU_F_SI) -> float:
    """Density-free Stokes response time tau_p = d_b^2 / (18 nu)  [s].

    ``d_mm`` is the bubble diameter in mm.  Density-free because the density
    ratio is carried by ``R`` (see ``dimension_conversion.md`` §3)."""
    import numpy as np
    d = np.asarray(d_mm, float) * 1.0e-3
    return d * d / (18.0 * nu_si)


def stokes_number(d_mm, u_ring_mms: float, R0_mm: float = R0,
                  nu_si: float = NU_F_SI):
    """St = d_b^2 U_ring / (18 nu R0)  (density-free; vectorised over d_mm)."""
    import numpy as np
    d = np.asarray(d_mm, float) * 1.0e-3      # m
    U = u_ring_mms * 1.0e-3                    # m/s
    R0_m = R0_mm * 1.0e-3                      # m
    return d * d * U / (18.0 * nu_si * R0_m)


def froude_number(u_ring_mms: float, rho_f: float = RHO_F,
                  delta_rho: float = DELTA_RHO, R0_mm: float = R0,
                  g: float = G_SI) -> float:
    """Fr = U_ring sqrt(rho_f / (d_rho g R0))  (section 4a, per-ring, dimensionless).

    For an air/H2 bubble in water d_rho ~ rho_f, so Fr ~ U_ring / sqrt(g R0)."""
    U = u_ring_mms * 1.0e-3
    R0_m = R0_mm * 1.0e-3
    return U * math.sqrt(rho_f / (delta_rho * g * R0_m))


def froude_number_proposed(*args, **kwargs):
    """DISABLED — the a_eq/v_b form in build_plan §2F is not dimensionless.

    See ``dimension_conversion.md`` §4b: the radicand has units s/m, and with
    v_b ~ d^3 the rise speed would scale as d^5 instead of d^2.  Resolve the
    derivation with the author before enabling."""
    raise NotImplementedError(
        "froude_number_proposed (a_eq/v_b form) is non-dimensionless and "
        "disabled; use froude_number() (section 4a). See dimension_conversion.md §4.")


def stokes_terminal_velocity(d_mm) -> float:
    """Physical Stokes rise speed v_t = d_rho d_b^2 g / (18 mu)  [m/s]."""
    import numpy as np
    d = np.asarray(d_mm, float) * 1.0e-3
    return DELTA_RHO * d * d * G_SI / (18.0 * MU_F)


def bubble_reynolds(slip_mms, d_mm, nu_si: float = NU_F_SI):
    """Re_b = |u-v| d_b / nu  (Stokes-drag validity check, §7)."""
    import numpy as np
    slip = np.abs(np.asarray(slip_mms, float)) * 1.0e-3
    d = np.asarray(d_mm, float) * 1.0e-3
    return slip * d / nu_si


def check_terminal_velocity_consistency(d_mm, u_ring_mms, R0_mm: float = R0):
    """§6c self-check: St/Fr^2 must equal v_t/U_ring.  Returns (model, phys)."""
    St = stokes_number(d_mm, u_ring_mms, R0_mm)
    Fr = froude_number(u_ring_mms, R0_mm=R0_mm)
    w_model = St / Fr ** 2
    w_phys = stokes_terminal_velocity(d_mm) / (u_ring_mms * 1.0e-3)
    return w_model, w_phys
