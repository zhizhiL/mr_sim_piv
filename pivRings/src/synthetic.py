"""
synthetic.py — a synthetic axisymmetric vortex-ring PIV field.

Produces a :class:`digiflow_io.PIVFrames` object indistinguishable (to the rest
of the pipeline) from real DigiFlow data, so the whole protocol can be run and
reviewed end-to-end without the experimental ``.dfi`` files.

Model
-----
A vortex-ring cross-section in the meridional plane is a pair of
counter-rotating Lamb-Oseen vortices straddling the ring axis ``y = y_axis``:

    upper core at (x_c, y_axis + R0)  with circulation +Gamma
    lower core at (x_c, y_axis - R0)  with circulation -Gamma

Each core induces a Lamb-Oseen tangential velocity
``u_theta(rho) = Gamma/(2 pi rho) (1 - exp(-rho^2 / a^2))``.
A uniform convection ``U_c`` is added to the streamwise component so the ring
translates through the (fixed) field of view; the cores drift downstream at
``U_c`` and the circulation decays slowly, which exercises the convection-speed
estimator and the scale-separation report.
"""

from __future__ import annotations

import numpy as np

from digiflow_io import PIVFrames
from constants import R0 as R0_DEFAULT


def _lamb_oseen(dx, dy, gamma, a, sign):
    """(u, v, omega) of a single Lamb-Oseen vortex centred at the origin."""
    rho2 = dx * dx + dy * dy
    rho = np.sqrt(rho2)
    rho_safe = np.maximum(rho, 1e-9)
    u_theta = sign * gamma / (2.0 * np.pi * rho_safe) * (1.0 - np.exp(-rho2 / a ** 2))
    # tangential (CCW positive): (-sin, cos) with sin = dy/rho, cos = dx/rho
    u = -u_theta * (dy / rho_safe)
    v = u_theta * (dx / rho_safe)
    omega = sign * gamma / (np.pi * a ** 2) * np.exp(-rho2 / a ** 2)
    return u, v, omega


def make_ring_frames(
    n_frames: int = 24,
    fps: float = 200.0,
    R0: float = R0_DEFAULT,
    a_core: float = 4.0,
    gamma0: float = 9000.0,
    U_c: float = 100.0,
    gamma_decay: float = 0.15,
    y_axis: float = 0.0,
    x_extent=(-40.0, 40.0),
    y_extent=(-40.0, 40.0),
    spacing: float = 0.5,
    noise: float = 2.0,
    seed: int = 0,
    station: str = "synthetic",
) -> PIVFrames:
    """Generate a translating Lamb-Oseen vortex-ring cross-section.

    Parameters mirror the experimental scales: lengths in mm, velocities in
    mm/s, circulation in mm^2/s.  ``gamma_decay`` is the fractional drop in
    circulation over the whole window; ``noise`` is the std of additive
    Gaussian velocity noise (mm/s) emulating PIV measurement scatter.
    """
    rng = np.random.default_rng(seed)

    x = np.arange(x_extent[0], x_extent[1] + spacing, spacing)
    y = np.arange(y_extent[0], y_extent[1] + spacing, spacing)
    X, Y = np.meshgrid(x, y)

    t = np.arange(n_frames) / fps
    T = t[-1] if n_frames > 1 else 1.0

    # Start the ring slightly upstream so it stays inside the FOV across the
    # window (keeps the time-average meaningful after co-moving subtraction).
    x_c0 = x_extent[0] + 0.30 * (x_extent[1] - x_extent[0])

    u_st, v_st, w_st = [], [], []
    core_positions = []  # bookkeeping (truth) for validation
    for k in range(n_frames):
        x_c = x_c0 + U_c * t[k]
        gamma = gamma0 * (1.0 - gamma_decay * (t[k] / T))

        # upper (+) and lower (-) cores
        up_u, up_v, up_w = _lamb_oseen(X - x_c, Y - (y_axis + R0), gamma, a_core, +1)
        lo_u, lo_v, lo_w = _lamb_oseen(X - x_c, Y - (y_axis - R0), gamma, a_core, -1)

        u = up_u + lo_u + U_c          # lab frame: add ring convection
        v = up_v + lo_v
        w = up_w + lo_w

        if noise > 0:
            u = u + rng.normal(0.0, noise, u.shape)
            v = v + rng.normal(0.0, noise, v.shape)

        u_st.append(u)
        v_st.append(v)
        w_st.append(w)
        core_positions.append((x_c, gamma))

    return PIVFrames(
        X=X, Y=Y,
        u=np.stack(u_st), v=np.stack(v_st), omega=np.stack(w_st),
        t=t,
        meta={
            "source": "synthetic",
            "station": station,
            "truth": {
                "R0": R0, "a_core": a_core, "gamma0": gamma0, "U_c": U_c,
                "gamma_decay": gamma_decay, "y_axis": y_axis,
                "core_positions": core_positions,
            },
            "dt": float(1.0 / fps),
        },
    )


def make_three_stations(**kwargs) -> dict:
    """Three synthetic stations (5D / 10D / 15D downstream) with mild
    station-to-station drift in circulation and core size, to exercise the
    quasi-steady station-comparison diagnostic (milestone M4)."""
    stations = {}
    for i, (name, g_scale, a_scale) in enumerate(
        [("station_1", 1.00, 1.00), ("station_2", 0.92, 1.10), ("station_3", 0.85, 1.22)]
    ):
        params = dict(kwargs)
        params.setdefault("gamma0", 9000.0)
        params.setdefault("a_core", 4.0)
        params["gamma0"] = params["gamma0"] * g_scale
        params["a_core"] = params["a_core"] * a_scale
        params["station"] = name
        params["seed"] = kwargs.get("seed", 0) + i
        stations[name] = make_ring_frames(**params)
    return stations
