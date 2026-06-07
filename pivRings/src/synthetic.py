"""
synthetic.py — a synthetic axisymmetric vortex-ring PIV field (DIMENSIONAL).

Produces a :class:`digiflow_io.PIVFrames` indistinguishable (to the rest of the
pipeline) from real DigiFlow data, so the whole protocol can be exercised
end-to-end without the experimental ``.dfi`` files.

A vortex-ring cross-section is a pair of counter-rotating Lamb-Oseen vortices
straddling the ring axis ``y = y_axis`` (upper +Gamma at y_axis+R0, lower
-Gamma at y_axis-R0).  A uniform convection ``U_c`` is added so the ring
translates through the fixed FOV; the cores drift at ``U_c`` and Gamma decays
slowly, exercising the convection-speed estimator and the scale-separation
report.
"""

from __future__ import annotations

import numpy as np

from digiflow_io import PIVFrames
from constants import R0 as R0_DEFAULT


def _lamb_oseen(dx, dy, gamma, a, sign):
    rho2 = dx * dx + dy * dy
    rho = np.sqrt(rho2)
    rho_safe = np.maximum(rho, 1e-9)
    u_theta = sign * gamma / (2.0 * np.pi * rho_safe) * (1.0 - np.exp(-rho2 / a ** 2))
    u = -u_theta * (dy / rho_safe)
    v = u_theta * (dx / rho_safe)
    omega = sign * gamma / (np.pi * a ** 2) * np.exp(-rho2 / a ** 2)
    return u, v, omega


def make_ring_frames(n_frames=24, fps=200.0, R0=R0_DEFAULT, a_core=4.0,
                     gamma0=9000.0, U_c=200.0, gamma_decay=0.15, y_axis=0.0,
                     x_extent=(-40.0, 40.0), y_extent=(-40.0, 40.0),
                     spacing=0.5, noise=2.0, seed=0, station="synthetic") -> PIVFrames:
    """Translating Lamb-Oseen vortex-ring cross-section.  Lengths mm, velocity
    mm/s, circulation mm^2/s.  ``gamma_decay`` is the fractional Gamma drop over
    the window; ``noise`` is additive Gaussian velocity std (mm/s)."""
    rng = np.random.default_rng(seed)

    x = np.arange(x_extent[0], x_extent[1] + spacing, spacing)
    y = np.arange(y_extent[0], y_extent[1] + spacing, spacing)
    X, Y = np.meshgrid(x, y)

    t = np.arange(n_frames) / fps
    T = t[-1] if n_frames > 1 else 1.0
    x_c0 = x_extent[0] + 0.30 * (x_extent[1] - x_extent[0])

    u_st, v_st, w_st, core_positions = [], [], [], []
    for k in range(n_frames):
        x_c = x_c0 + U_c * t[k]
        gamma = gamma0 * (1.0 - gamma_decay * (t[k] / T))
        up_u, up_v, up_w = _lamb_oseen(X - x_c, Y - (y_axis + R0), gamma, a_core, +1)
        lo_u, lo_v, lo_w = _lamb_oseen(X - x_c, Y - (y_axis - R0), gamma, a_core, -1)
        u = up_u + lo_u + U_c
        v = up_v + lo_v
        w = up_w + lo_w
        if noise > 0:
            u = u + rng.normal(0.0, noise, u.shape)
            v = v + rng.normal(0.0, noise, v.shape)
        u_st.append(u); v_st.append(v); w_st.append(w)
        core_positions.append((x_c, gamma))

    return PIVFrames(
        X=X, Y=Y, u=np.stack(u_st), v=np.stack(v_st), omega=np.stack(w_st), t=t,
        meta={"source": "synthetic", "station": station, "dt": float(1.0 / fps),
              "truth": {"R0": R0, "a_core": a_core, "gamma0": gamma0, "U_c": U_c,
                        "gamma_decay": gamma_decay, "y_axis": y_axis,
                        "core_positions": core_positions}},
    )


def make_three_stations(**kwargs) -> dict:
    """Three synthetic stations (5D/10D/15D) with mild station-to-station drift
    in circulation and core size for the quasi-steady comparison (M4)."""
    stations = {}
    for i, (name, g_scale, a_scale) in enumerate(
        [("station_1", 1.00, 1.00), ("station_2", 0.92, 1.10), ("station_3", 0.85, 1.22)]
    ):
        params = dict(kwargs)
        params["gamma0"] = params.get("gamma0", 9000.0) * g_scale
        params["a_core"] = params.get("a_core", 4.0) * a_scale
        params["station"] = name
        params["seed"] = kwargs.get("seed", 0) + i
        stations[name] = make_ring_frames(**params)
    return stations
