"""
frame_transform.py — Stage B: reference frame (DIMENSIONAL).

Estimate the ring convection velocity ``U_ring`` ( = U_f, the velocity scale).
Per ``dimension_conversion.md`` §1 the recommended tracked point is the
**vorticity-weighted axial centroid** (most robust); a two-core linear fit is
also provided.  The same ``U_ring`` is reused for the co-moving subtraction
inside :func:`nondimensional.nondimensionalize_field` — do not re-estimate it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from digiflow_io import PIVFrames
from vortex_core import find_vortex_cores_iterative


@dataclass
class UcEstimate:
    U_c: float                 # convection velocity (= U_ring) [mm/s]
    intercept: float           # fitted x at t=0 [mm]
    x_track: np.ndarray        # per-frame tracked axial position [mm]
    t: np.ndarray              # timestamps [s]
    residual_rms: float        # rms of the linear-fit residual [mm]
    method: str = ""


def _vorticity_centroid_x(frames: PIVFrames) -> np.ndarray:
    """Per-frame axial position of the |omega|-weighted centroid [mm]."""
    x = np.full(frames.nframes, np.nan)
    X = frames.X
    for k in range(frames.nframes):
        w = np.abs(frames.omega[k])
        s = w.sum()
        if s > 0:
            x[k] = float((X * w).sum() / s)
    return x


def _core_mean_x(frames: PIVFrames, fit_radius=8.0) -> np.ndarray:
    """Per-frame mean axial position of the two vortex cores [mm]."""
    x = np.full(frames.nframes, np.nan)
    gp = gn = None
    for k in range(frames.nframes):
        cores, _, _ = find_vortex_cores_iterative(
            frames.X, frames.Y, frames.omega[k],
            initial_guess_pos=gp, initial_guess_neg=gn,
            fit_radius=fit_radius, verbose=False)
        (xp, yp, _), (xn, yn, _) = cores
        gp, gn = (xp, yp), (xn, yn)
        x[k] = 0.5 * (xp + xn)
    return x


def estimate_Uc(frames: PIVFrames, method="vorticity_centroid",
                fit_radius=8.0) -> UcEstimate:
    """Estimate U_ring by linearly fitting the tracked axial position vs time.

    ``method`` in {"vorticity_centroid" (default, recommended), "core_linear_fit"}.
    """
    if method == "vorticity_centroid":
        x_track = _vorticity_centroid_x(frames)
    elif method == "core_linear_fit":
        x_track = _core_mean_x(frames, fit_radius=fit_radius)
    else:
        raise NotImplementedError(f"Unknown U_c method {method!r}")

    t = frames.t
    good = np.isfinite(x_track)
    if good.sum() < 2:
        raise RuntimeError("Need >= 2 frames with a valid tracked point to fit U_ring")
    slope, intercept = np.polyfit(t[good], x_track[good], 1)
    resid = x_track[good] - (slope * t[good] + intercept)
    return UcEstimate(U_c=float(abs(slope)), intercept=float(intercept),
                      x_track=x_track, t=t,
                      residual_rms=float(np.sqrt(np.mean(resid ** 2))),
                      method=method)


def to_comoving(frames: PIVFrames, U_c: float) -> PIVFrames:
    """Return a copy with the uniform convection ``U_c e_x`` subtracted
    (dimensional; used for the residual-unsteadiness QC metric)."""
    out = copy.copy(frames)
    out.u = frames.u - U_c
    out.v = frames.v.copy()
    out.omega = frames.omega.copy()    # vorticity is Galilean invariant
    out.meta = dict(frames.meta)
    out.meta["U_c_subtracted"] = float(U_c)
    return out


def estimate_uring_thinring(mean, sign=1.0, R0_fallback_a=4.0):
    """Vortex-ring self-propagation speed from the thin-core formula

        U = Gamma / (4 pi R) * (ln(8 R / a) - 1/4)

    using the measured circulation Gamma, ring radius R (half the core
    spacing) and core radius a (a_eq).  This is the physically-principled,
    time-independent U_ring estimator; it agrees with the vorticity-centroid
    fit where the latter is reliable and recovers sensible values where the
    centroid fit fails.  ``sign`` carries the propagation direction (+/-x).
    Returns ``(U_ring_signed, info)``."""
    cores, _, gammas = find_vortex_cores_iterative(mean.X, mean.Y, mean.omega,
                                                   fit_radius=8.0, verbose=False)
    (xp, yp, _), (xn, yn, _) = cores
    R = 0.5 * abs(yp - yn)
    G = 0.5 * (abs(gammas[0]) + abs(gammas[1]))
    from seeding import fit_core_ellipse
    try:
        ell = fit_core_ellipse(mean, y_axis=0.5 * (yp + yn), R0=20.0)
        a = ell.a_eq if np.isfinite(ell.a_eq) else R0_fallback_a
    except Exception:
        a = R0_fallback_a
    U = G / (4.0 * np.pi * R) * (np.log(8.0 * R / a) - 0.25)
    return float(np.sign(sign) * U), {"Gamma": float(G), "R": float(R), "a": float(a)}


def register_frames(frames: PIVFrames, x_track: np.ndarray, x_ref=None) -> PIVFrames:
    """Spatially de-translate the ring: shift each frame in x so its tracked
    axial position ``x_track[k]`` aligns to a common ``x_ref`` (default the
    window mean).  Needed when the ring moves more than ~half its core spacing
    across the window, otherwise the lab-frame time-average smears the cores.

    Velocity magnitudes are unchanged (a rigid spatial shift); the residual ring
    translation is still removed later by the U_ring co-moving subtraction."""
    from scipy.ndimage import shift as ndshift
    if x_ref is None:
        x_ref = float(np.nanmean(x_track))
    dx = frames.dx
    u, v, w = [], [], []
    for k in range(frames.nframes):
        px = -(x_track[k] - x_ref) / dx           # pixels along axis=1 (x)
        u.append(ndshift(frames.u[k], (0.0, px), order=1, mode="nearest"))
        v.append(ndshift(frames.v[k], (0.0, px), order=1, mode="nearest"))
        w.append(ndshift(frames.omega[k], (0.0, px), order=1, mode="nearest"))
    out = copy.copy(frames)
    out.u, out.v, out.omega = np.stack(u), np.stack(v), np.stack(w)
    out.meta = dict(frames.meta)
    out.meta["registered_x_ref"] = x_ref
    return out


def residual_unsteadiness(frames_co: PIVFrames) -> dict:
    """QC: temporal std of the co-moving speed relative to its spatial rms.
    Small values support the quasi-steady assumption."""
    speed = np.sqrt(frames_co.u ** 2 + frames_co.v ** 2)
    mean_field = speed.mean(axis=0)
    temporal_std = speed.std(axis=0)
    ref = np.sqrt(np.mean(mean_field ** 2)) + 1e-12
    return {"temporal_std_mean": float(temporal_std.mean()),
            "temporal_std_max": float(temporal_std.max()),
            "relative_unsteadiness": float(temporal_std.mean() / ref)}
