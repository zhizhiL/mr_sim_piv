"""
averaging.py — Stage C: window time-average + spatial smoothing (DIMENSIONAL).

Average the frames over a window ``T_win`` and low-pass the result *before*
differentiation (PIV gradient noise is amplified in grad u).  Also emits the
scale-separation report feeding the quasi-steady justification (build_plan §2C).

The averaging is done in the LAB frame here; the co-moving subtraction and
non-dimensionalisation happen once in ``nondimensional.py`` (see §0 of
``dimension_conversion.md``).  The omega field is kept for the core/ellipse fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from digiflow_io import PIVFrames


@dataclass
class MeanField:
    """Time-averaged, smoothed meridional field on the world grid (mm, mm/s)."""
    X: np.ndarray
    Y: np.ndarray
    Ux: np.ndarray      # mean streamwise velocity [mm/s], lab frame
    Uy: np.ndarray      # mean transverse velocity [mm/s]
    omega: np.ndarray   # mean vorticity [1/s]
    meta: dict = field(default_factory=dict)

    @property
    def dx(self):
        return float(self.X[0, 1] - self.X[0, 0])

    @property
    def dy(self):
        return float(self.Y[1, 0] - self.Y[0, 0])


def reject_outliers(stack: np.ndarray, n_mad: float = 8.0) -> np.ndarray:
    """Mask gross PIV outliers (spurious vectors) as NaN via a global MAD test.

    Returns a copy with ``|x - median| > n_mad * 1.4826 * MAD`` set to NaN, so a
    subsequent ``np.nanmean`` ignores them.  No-op if ``n_mad`` is None or MAD=0."""
    if n_mad is None:
        return stack
    med = np.nanmedian(stack)
    mad = np.nanmedian(np.abs(stack - med)) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return stack
    out = stack.astype(float, copy=True)
    out[np.abs(stack - med) > n_mad * mad] = np.nan
    return out


def time_average(frames: PIVFrames, t_window=None, reject_mad: float = 8.0) -> MeanField:
    """Average u, v, omega over frames in ``t_window`` (default all), rejecting
    gross velocity outliers first (``reject_mad`` MADs; None to disable)."""
    t = frames.t
    sel = (np.ones(frames.nframes, bool) if t_window is None
           else (t >= t_window[0]) & (t <= t_window[1]))
    if sel.sum() == 0:
        raise ValueError("No frames inside the averaging window")

    u = reject_outliers(frames.u[sel], reject_mad)
    v = reject_outliers(frames.v[sel], reject_mad)
    n_rejected = int(np.isnan(u).sum() + np.isnan(v).sum())
    Ux = np.nanmean(u, axis=0)
    Uy = np.nanmean(v, axis=0)
    # fill any all-NaN pixels (rare) so gradients stay finite
    Ux = np.nan_to_num(Ux, nan=float(np.nanmedian(Ux)))
    Uy = np.nan_to_num(Uy, nan=float(np.nanmedian(Uy)))
    return MeanField(
        X=frames.X, Y=frames.Y, Ux=Ux, Uy=Uy,
        omega=frames.omega[sel].mean(axis=0),
        meta={"n_avg": int(sel.sum()), "n_outliers_rejected": n_rejected,
              "t_window": (float(t[sel][0]), float(t[sel][-1]))})


def smooth(field_2d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass in grid-point units (apply before differentiation)."""
    if sigma <= 0:
        return field_2d
    return gaussian_filter(field_2d, sigma=sigma, mode="nearest")


def smooth_field(mean: MeanField, sigma: float) -> MeanField:
    out = MeanField(X=mean.X, Y=mean.Y,
                    Ux=smooth(mean.Ux, sigma), Uy=smooth(mean.Uy, sigma),
                    omega=smooth(mean.omega, sigma), meta=dict(mean.meta))
    out.meta["smooth_sigma"] = sigma
    return out


def scale_separation_report(frames: PIVFrames, U_ring: float, R0: float,
                            per_frame_gamma=None) -> dict:
    """Turnovers-per-window and fractional drift of Gamma / U_ring across the
    window (quasi-steady metrics, build_plan §2C)."""
    T_win = float(frames.t[-1] - frames.t[0]) if frames.nframes > 1 else 0.0
    u_scale = max(abs(U_ring), 1e-9)
    tau_turn = 2.0 * np.pi * R0 / u_scale
    report = {"T_win": T_win, "tau_turnover": tau_turn,
              "turnovers_per_window": T_win / tau_turn if tau_turn > 0 else np.nan}
    if per_frame_gamma is not None and len(per_frame_gamma) >= 2:
        g = np.asarray(per_frame_gamma, float)
        report["gamma_fractional_change"] = float((g[0] - g[-1]) / (abs(g[0]) + 1e-12))
    return report
