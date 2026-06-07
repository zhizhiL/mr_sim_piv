"""
frame_transform.py — Stage B: reference frame.

Estimate the ring convection velocity ``U_c`` and move the PIV stack into the
co-moving frame.  ``U_c`` is obtained by linearly fitting the streamwise core
position (mean of the upper and lower cores) against time, as specified in
build_plan.md §2 Stage B.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from digiflow_io import PIVFrames
from vortex_core import find_vortex_cores_iterative  # repo-root helper


@dataclass
class UcEstimate:
    U_c: float                 # convection velocity [mm/s]
    intercept: float           # fitted x_core at t=0 [mm]
    x_core: np.ndarray         # per-frame mean core x [mm]
    t: np.ndarray              # timestamps [s]
    residual_rms: float        # rms of the linear-fit residual [mm]
    per_core: list             # per-frame [(x_pos, y_pos), (x_neg, y_neg)]


def _track_cores(frames: PIVFrames, fit_radius: float = 8.0):
    """Locate the two vortex cores in every frame (m/s independent)."""
    x_mean = np.full(frames.nframes, np.nan)
    per_core = []
    guess_pos = guess_neg = None
    for k in range(frames.nframes):
        cores, _, _ = find_vortex_cores_iterative(
            frames.X, frames.Y, frames.omega[k],
            initial_guess_pos=guess_pos, initial_guess_neg=guess_neg,
            fit_radius=fit_radius, verbose=False,
        )
        (xp, yp, _), (xn, yn, _) = cores
        # warm-start the next frame from this frame's result
        guess_pos, guess_neg = (xp, yp), (xn, yn)
        x_mean[k] = 0.5 * (xp + xn)
        per_core.append([(xp, yp), (xn, yn)])
    return x_mean, per_core


def estimate_Uc(frames: PIVFrames, method: str = "core_linear_fit",
                fit_radius: float = 8.0) -> UcEstimate:
    """Estimate the ring convection velocity.

    Currently implements ``core_linear_fit``: track both cores per frame,
    average their streamwise positions, and fit ``x_core(t) = U_c t + b``.
    """
    if method != "core_linear_fit":
        raise NotImplementedError(f"Unknown U_c method {method!r}")

    x_core, per_core = _track_cores(frames, fit_radius=fit_radius)
    t = frames.t
    good = np.isfinite(x_core)
    if good.sum() < 2:
        raise RuntimeError("Need >= 2 frames with detected cores to fit U_c")

    slope, intercept = np.polyfit(t[good], x_core[good], 1)
    resid = x_core[good] - (slope * t[good] + intercept)
    return UcEstimate(
        U_c=float(slope), intercept=float(intercept),
        x_core=x_core, t=t,
        residual_rms=float(np.sqrt(np.mean(resid ** 2))),
        per_core=per_core,
    )


def to_comoving(frames: PIVFrames, U_c: float, spatially_varying: bool = False) -> PIVFrames:
    """Return a copy of ``frames`` with the convection subtracted.

    Default is a uniform ``U_c e_x`` subtraction (build_plan §2 Stage B
    default).  ``spatially_varying=True`` is reserved for a per-location mean
    flow and currently raises (the flag is exposed for parity with the plan).
    """
    if spatially_varying:
        raise NotImplementedError("Spatially-varying U_c not implemented; use uniform")
    out = copy.copy(frames)
    out.u = frames.u - U_c
    out.v = frames.v.copy()
    out.omega = frames.omega.copy()  # vorticity is Galilean invariant
    out.meta = dict(frames.meta)
    out.meta["U_c_subtracted"] = float(U_c)
    return out


def residual_unsteadiness(frames_co: PIVFrames) -> dict:
    """QC metric: how steady is the co-moving field across the window?

    Reports the temporal std of the velocity magnitude (averaged over space)
    relative to the spatial-rms velocity — small values support the
    quasi-steady assumption.
    """
    speed = np.sqrt(frames_co.u ** 2 + frames_co.v ** 2)       # (nf, ny, nx)
    mean_field = speed.mean(axis=0)
    temporal_std = speed.std(axis=0)
    ref = np.sqrt(np.mean(mean_field ** 2)) + 1e-12
    return {
        "temporal_std_mean": float(temporal_std.mean()),
        "temporal_std_max": float(temporal_std.max()),
        "relative_unsteadiness": float(temporal_std.mean() / ref),
    }
