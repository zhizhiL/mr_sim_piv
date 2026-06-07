"""
averaging.py — Stage C: window time-average + spatial smoothing.

Average the co-moving frames over a window ``T_win``, low-pass the result
*before* differentiation (PIV gradient noise), and emit the scale-separation
report that feeds the quasi-steady justification (build_plan §2 Stage C).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from digiflow_io import PIVFrames


@dataclass
class MeanField:
    """Time-averaged, smoothed meridional field on the world grid."""
    X: np.ndarray
    Y: np.ndarray
    Ux: np.ndarray      # mean streamwise velocity [mm/s]
    Uy: np.ndarray      # mean transverse velocity [mm/s]
    omega: np.ndarray   # mean vorticity [1/s]
    meta: dict = field(default_factory=dict)

    @property
    def dx(self):
        return float(self.X[0, 1] - self.X[0, 0])

    @property
    def dy(self):
        return float(self.Y[1, 0] - self.Y[0, 0])


def time_average(frames: PIVFrames, t_window=None) -> MeanField:
    """Average ``u, v, omega`` over the frames whose timestamps fall in
    ``t_window = (t0, t1)`` (default: all frames)."""
    t = frames.t
    if t_window is None:
        sel = np.ones(frames.nframes, dtype=bool)
    else:
        sel = (t >= t_window[0]) & (t <= t_window[1])
    if sel.sum() == 0:
        raise ValueError("No frames inside the averaging window")

    return MeanField(
        X=frames.X, Y=frames.Y,
        Ux=frames.u[sel].mean(axis=0),
        Uy=frames.v[sel].mean(axis=0),
        omega=frames.omega[sel].mean(axis=0),
        meta={"n_avg": int(sel.sum()),
              "t_window": (float(t[sel][0]), float(t[sel][-1]))},
    )


def smooth(field_2d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass in *grid-point* units (apply before differentiation)."""
    if sigma <= 0:
        return field_2d
    return gaussian_filter(field_2d, sigma=sigma, mode="nearest")


def smooth_field(mean: MeanField, sigma: float) -> MeanField:
    """Smooth Ux, Uy, omega of a :class:`MeanField` in place-safe fashion."""
    out = MeanField(
        X=mean.X, Y=mean.Y,
        Ux=smooth(mean.Ux, sigma),
        Uy=smooth(mean.Uy, sigma),
        omega=smooth(mean.omega, sigma),
        meta=dict(mean.meta),
    )
    out.meta["smooth_sigma"] = sigma
    return out


def scale_separation_report(frames: PIVFrames, U_c: float, R0: float,
                            per_frame_gamma=None) -> dict:
    """Quasi-steady scale-separation metrics over the window.

    Reports turnovers-per-window (``T_win / tau_turnover``) and the fractional
    change of Gamma, R0-scale and U_c across the window.  ``per_frame_gamma``
    (optional) is a sequence of circulation estimates used for the Gamma drift.
    """
    T_win = float(frames.t[-1] - frames.t[0]) if frames.nframes > 1 else 0.0

    # turnover time ~ 2 pi R0 / U_theta_scale; use U_c as a velocity proxy.
    u_scale = max(abs(U_c), 1e-9)
    tau_turn = 2.0 * np.pi * R0 / u_scale
    turnovers = T_win / tau_turn if tau_turn > 0 else np.nan

    report = {
        "T_win": T_win,
        "tau_turnover": tau_turn,
        "turnovers_per_window": turnovers,
    }
    if per_frame_gamma is not None and len(per_frame_gamma) >= 2:
        g = np.asarray(per_frame_gamma, dtype=float)
        report["gamma_fractional_change"] = float((g[0] - g[-1]) / (abs(g[0]) + 1e-12))
    return report
