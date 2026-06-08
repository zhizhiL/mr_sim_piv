"""
nondimensional.py — the ONE place dimensional PIV becomes dimensionless.

Per ``dimension_conversion.md`` §0: convert once here, keep everything
downstream (field build, seeding, advect) dimensionless and identical in form
to the reused solver ``advect_bubbles_3D_eval.py``.

Reference scales:  L_f = R0,  U_f = U_ring,  tau_f = R0 / U_ring.

    x* = x / R0
    r* = r / R0
    Ux* = (Ux - U_ring) / U_ring      # co-moving (axial) THEN normalise
    Ur* =  Ur / U_ring
"""

from __future__ import annotations

import numpy as np

from constants import (stokes_number, froude_number, stokes_terminal_velocity)


def nondimensionalize_field(x_mm, r_mm, Ux_mms, Ur_mms, U_ring, R0):
    """Convert a folded meridional field (mm, mm/s) to dimensionless, co-moving.

    ``U_ring`` may be SIGNED (negative for a ring propagating in -x, e.g. the
    coord_down stations): the co-moving subtraction uses the signed value while
    the velocity SCALE is its magnitude.  Returns ``(x_star, r_star, Ux_star,
    Ur_star)``; gradients are taken in the starred coordinates afterwards."""
    scale = abs(U_ring)
    if scale <= 0:
        raise ValueError("U_ring magnitude must be positive")
    x_star = np.asarray(x_mm, float) / R0
    r_star = np.asarray(r_mm, float) / R0
    Ux_star = (np.asarray(Ux_mms, float) - U_ring) / scale
    Ur_star = np.asarray(Ur_mms, float) / scale
    return x_star, r_star, Ux_star, Ur_star


def tau_f(U_ring_mms, R0_mm):
    """Flow timescale tau_f = R0 / U_ring  [s]."""
    return (R0_mm * 1.0e-3) / (U_ring_mms * 1.0e-3)


# --------------------------------------------------------------------------
# Validation assertions (dimension_conversion.md §6)
# --------------------------------------------------------------------------
def assert_field_O1(Ux_star, p=99, limit=8.0, warn=False):
    """§6a: the dimensionless co-moving axial velocity should be O(1).

    Strong rings legitimately have swirl a few x U_ring, so the limit is a
    loose sanity bound (default 8).  With ``warn=True`` a breach prints a
    warning and returns the value instead of raising — use that in the drivers
    so one bad station does not abort a batch; the hard assert is for tests."""
    val = float(np.nanpercentile(np.abs(Ux_star), p))
    if val >= limit:
        msg = (f"Ux* not O(1) (p{p}={val:.2f} >= {limit}); check U_f / units, "
               "the co-moving subtraction, or PIV outliers in this window")
        if warn:
            import warnings
            warnings.warn(msg)
        else:
            raise AssertionError(msg)
    return val


def assert_terminal_velocity_consistency(d_mm, U_ring_mms, R0_mm, rtol=1e-6):
    """§6c: St/Fr^2 must equal v_t/U_ring (ties §3 and §4a together)."""
    St = stokes_number(d_mm, U_ring_mms, R0_mm)
    Fr = froude_number(U_ring_mms, R0_mm=R0_mm)
    w_model = St / Fr ** 2
    w_phys = stokes_terminal_velocity(d_mm) / (U_ring_mms * 1.0e-3)
    assert np.allclose(w_model, w_phys, rtol=rtol), (
        f"St/Fr^2 ({np.max(w_model):.3e}) != v_t/U_ring ({np.max(w_phys):.3e})")
    return w_model, w_phys
