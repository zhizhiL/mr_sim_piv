"""
build_field.py — Stage D: build & cache the quasi-steady 3D field (DIMENSIONLESS).

From a smoothed meridional mean field (dimensional, lab frame) this module:

    1. locates the ring axis (r = 0),
    2. folds the two half-planes onto a common ``(x, r>=0)`` grid,
    3. non-dimensionalises ONCE (co-moving + L_f=R0, U_f=U_ring),
    4. computes the four gradients in the starred coordinates,
    5. revolves to a 3D axisymmetric Cartesian field with the CORRECTED
       chain-rule (the old solver's 1/cosθ, 1/sinθ form is the known bug;
       here r->0 is guarded with the analytic on-axis limit), and
    6. caches under ``fields/<station>/`` in the exact on-disk format the reused
       solver expects (so ``advect_bubbles_3D_eval.py`` can load it directly).

Splines are functions of ``(x, r)`` (x first) to match the solver's
``interp_Ux(xp, r_planar)`` call convention.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline

from averaging import MeanField
from nondimensional import nondimensionalize_field

_AXIS_EPS = 1.0e-6   # dimensionless r below which we use the on-axis limit


def locate_axis(mean: MeanField, y_axis=None) -> float:
    """World-y of the ring axis (mm).  If not given, use the |omega|-weighted
    centre of mass in y (stable reflection-symmetry proxy for a ring)."""
    if y_axis is not None:
        return float(y_axis)
    yvals = mean.Y[:, 0]
    col_profile = np.abs(mean.omega).mean(axis=1)
    return float(np.sum(yvals * col_profile) / (np.sum(col_profile) + 1e-12))


def _fold_halfplanes(mean: MeanField, y_axis, n_r=None):
    """Fold upper/lower halves onto a common ``(r>=0, x)`` grid (mm, mm/s).
    Radial velocity Ur is positive OUTWARD from the axis."""
    x_axis = mean.X[0, :].astype(float)
    yvals = mean.Y[:, 0].astype(float)
    r_signed = yvals - y_axis
    upper, lower = r_signed >= 0, r_signed <= 0

    r_up, r_lo = r_signed[upper], -r_signed[lower]
    o_up, o_lo = np.argsort(r_up), np.argsort(r_lo)
    Ux_up, Uy_up = mean.Ux[upper][o_up], mean.Uy[upper][o_up]
    Ux_lo, Uy_lo = mean.Ux[lower][o_lo], mean.Uy[lower][o_lo]
    r_up, r_lo = r_up[o_up], r_lo[o_lo]

    r_max = min(r_up.max(), r_lo.max())
    if n_r is None:
        n_r = max(len(r_up), len(r_lo))
    r_axis = np.linspace(0.0, r_max, n_r)

    Ux = np.zeros((n_r, len(x_axis)))
    Ur = np.zeros((n_r, len(x_axis)))
    for j in range(len(x_axis)):
        ux = 0.5 * (np.interp(r_axis, r_up, Ux_up[:, j]) + np.interp(r_axis, r_lo, Ux_lo[:, j]))
        # outward radial velocity: +v upper half, -v lower half
        ur = 0.5 * (np.interp(r_axis, r_up, Uy_up[:, j]) + np.interp(r_axis, r_lo, -Uy_lo[:, j]))
        Ux[:, j] = ux
        Ur[:, j] = ur
    Ur[0, :] = 0.0   # radial velocity vanishes on the axis
    return x_axis, r_axis, Ux, Ur


@dataclass
class Field3D:
    """Revolved 3D axisymmetric DIMENSIONLESS field backed by (x, r) splines.

    Ring axis is along x; radial plane is (y, z) with r = sqrt(y^2 + z^2).
    velocity_and_gradient returns (u*, du_i*/dx_j*)."""
    x_axis: np.ndarray         # dimensionless x grid
    r_axis: np.ndarray         # dimensionless r grid
    sp_Ux: RectBivariateSpline
    sp_Ur: RectBivariateSpline
    sp_dUxdx: RectBivariateSpline
    sp_dUxdr: RectBivariateSpline
    sp_dUrdx: RectBivariateSpline
    sp_dUrdr: RectBivariateSpline
    y_axis_mm: float = 0.0
    U_ring: float = 1.0        # mm/s, the velocity scale used to nondimensionalise
    R0: float = 20.0           # mm, the length scale
    meta: dict = None

    @property
    def bounds(self):
        return (float(self.x_axis[0]), float(self.x_axis[-1]),
                float(self.r_axis[0]), float(self.r_axis[-1]))

    def in_domain(self, P):
        P = np.atleast_2d(P)
        x = P[:, 0]
        r = np.sqrt(P[:, 1] ** 2 + P[:, 2] ** 2)
        xmin, xmax, rmin, rmax = self.bounds
        return (x >= xmin) & (x <= xmax) & (r >= rmin) & (r <= rmax)

    def velocity(self, P):
        P = np.atleast_2d(P)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        r = np.sqrt(y * y + z * z)
        rs = np.maximum(r, _AXIS_EPS)
        Ux = self.sp_Ux.ev(x, r)
        Ur = self.sp_Ur.ev(x, r)
        u = np.empty_like(P, dtype=float)
        u[:, 0] = Ux
        u[:, 1] = Ur * (y / rs)
        u[:, 2] = Ur * (z / rs)
        return u

    def velocity_and_gradient(self, P):
        """Return ``(u, gradu)``: u (N,3), gradu (N,3,3) with gradu[n,i,j]=du_i/dx_j.
        On-axis (r<eps) uses the analytic limit (the guard the build plan asks for)."""
        P = np.atleast_2d(P)
        N = P.shape[0]
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        r = np.sqrt(y * y + z * z)
        on_axis = r < _AXIS_EPS
        rs = np.where(on_axis, 1.0, r)

        Ux = self.sp_Ux.ev(x, r); Ur = self.sp_Ur.ev(x, r)
        dUxdx = self.sp_dUxdx.ev(x, r); dUxdr = self.sp_dUxdr.ev(x, r)
        dUrdx = self.sp_dUrdx.ev(x, r); dUrdr = self.sp_dUrdr.ev(x, r)
        ey, ez = y / rs, z / rs

        u = np.empty((N, 3))
        u[:, 0] = Ux
        u[:, 1] = np.where(on_axis, 0.0, Ur * ey)
        u[:, 2] = np.where(on_axis, 0.0, Ur * ez)

        g = np.zeros((N, 3, 3))
        g[:, 0, 0] = dUxdx
        g[:, 0, 1] = np.where(on_axis, 0.0, dUxdr * ey)
        g[:, 0, 2] = np.where(on_axis, 0.0, dUxdr * ez)
        g[:, 1, 0] = np.where(on_axis, 0.0, dUrdx * ey)
        g[:, 1, 1] = np.where(on_axis, dUrdr, dUrdr * ey * ey + Ur * (z * z) / rs ** 3)
        g[:, 1, 2] = np.where(on_axis, 0.0, dUrdr * ey * ez - Ur * (y * z) / rs ** 3)
        g[:, 2, 0] = np.where(on_axis, 0.0, dUrdx * ez)
        g[:, 2, 1] = np.where(on_axis, 0.0, dUrdr * ez * ey - Ur * (z * y) / rs ** 3)
        g[:, 2, 2] = np.where(on_axis, dUrdr, dUrdr * ez * ez + Ur * (y * y) / rs ** 3)
        return u, g


def build_field(mean: MeanField, U_ring: float, R0: float = 20.0,
                y_axis=None, n_r=None, spline_k=3) -> Field3D:
    """Fold (dimensional) -> nondimensionalise (co-moving) -> differentiate ->
    revolve into a dimensionless :class:`Field3D`.

    ``U_ring`` may be signed (negative -> ring propagates in -x); the stored
    ``field.U_ring`` is the magnitude (the velocity scale used for St/Fr)."""
    y0 = locate_axis(mean, y_axis)
    x_mm, r_mm, Ux_mms, Ur_mms = _fold_halfplanes(mean, y0, n_r=n_r)

    # ---- the single conversion boundary ----
    x_axis, r_axis, Ux, Ur = nondimensionalize_field(x_mm, r_mm, Ux_mms, Ur_mms, U_ring, R0)

    # gradients in the starred (dimensionless) coordinates: axis0=r, axis1=x
    dUxdr, dUxdx = np.gradient(Ux, r_axis, x_axis)
    dUrdr, dUrdx = np.gradient(Ur, r_axis, x_axis)

    k = min(spline_k, len(r_axis) - 1, len(x_axis) - 1)

    def _sp(Z):   # spline as a function of (x, r) -> Z.T has shape (n_x, n_r)
        return RectBivariateSpline(x_axis, r_axis, Z.T, kx=k, ky=k)

    field = Field3D(
        x_axis=x_axis, r_axis=r_axis,
        sp_Ux=_sp(Ux), sp_Ur=_sp(Ur),
        sp_dUxdx=_sp(dUxdx), sp_dUxdr=_sp(dUxdr),
        sp_dUrdx=_sp(dUrdx), sp_dUrdr=_sp(dUrdr),
        y_axis_mm=y0, U_ring=abs(float(U_ring)), R0=float(R0),
        meta={"n_r": len(r_axis), "n_x": len(x_axis),
              "U_ring_signed": float(U_ring), **(mean.meta or {})},
    )
    # stash the gridded dimensionless arrays for the solver-format dump
    field.meta["_grids"] = dict(Ux=Ux, Ur=Ur, dUxdx=dUxdx, dUxdr=dUxdr,
                                dUrdx=dUrdx, dUrdr=dUrdr)
    return field


def _core_geometry(mean: MeanField, y_axis, R0):
    """(x_core, y_core, y_core_lower, x_ring, y_ring) dimensionless, for the
    solver's geometry.npy (plotting/donut only)."""
    from vortex_core import find_vortex_cores_iterative
    cores, _, _ = find_vortex_cores_iterative(mean.X, mean.Y, mean.omega,
                                              fit_radius=8.0, verbose=False)
    (xp, yp, sp), (xn, yn, sn) = cores
    up = (xp, yp) if yp >= yn else (xn, yn)
    lo = (xn, yn) if yn <= yp else (xp, yp)
    x_core = up[0] / R0
    y_core = (up[1] - y_axis) / R0
    y_core_lower = (lo[1] - y_axis) / R0
    x_ring = 0.5 * (xp + xn) / R0
    y_ring = 0.0
    return np.array([[x_core, y_core, y_core_lower, x_ring, y_ring]])


def save_field(field: Field3D, out_dir: str, mean: MeanField = None):
    """Cache the dimensionless field.

    Writes the solver-compatible artifacts (``x.npy, y.npy, Ux.npy, Uy.npy,
    dUxdx.npy, dUxdy.npy, dUydx.npy, dUydy.npy, geometry.npy,
    interp_functions.pkl``) plus this pipeline's ``field.pkl`` (the
    :class:`Field3D`) and, if given, ``mean_field.pkl`` (for refitting the core
    ellipse) and ``meta.json``."""
    os.makedirs(out_dir, exist_ok=True)
    G = field.meta.get("_grids", {})

    # --- solver on-disk format (interp called as interp(x, r)) ---
    np.save(os.path.join(out_dir, "x.npy"), field.x_axis)
    np.save(os.path.join(out_dir, "y.npy"), field.r_axis)
    name_map = {"Ux": "Ux", "Ur": "Uy", "dUxdx": "dUxdx", "dUxdr": "dUxdy",
                "dUrdx": "dUydx", "dUrdr": "dUydy"}
    for key, fname in name_map.items():
        if key in G:
            np.save(os.path.join(out_dir, f"{fname}.npy"), G[key])

    if mean is not None:
        geom = _core_geometry(mean, field.y_axis_mm, field.R0)
    else:
        geom = np.array([[np.nan] * 5])
    np.save(os.path.join(out_dir, "geometry.npy"), geom)

    interp_tuple = (field.sp_Ux, field.sp_Ur, field.sp_dUxdx, field.sp_dUxdr,
                    field.sp_dUrdx, field.sp_dUrdr)
    with open(os.path.join(out_dir, "interp_functions.pkl"), "wb") as fh:
        pickle.dump(interp_tuple, fh)

    # --- this pipeline's objects ---
    field_to_pickle = Field3D(**{k: getattr(field, k) for k in
                                 ("x_axis", "r_axis", "sp_Ux", "sp_Ur", "sp_dUxdx",
                                  "sp_dUxdr", "sp_dUrdx", "sp_dUrdr", "y_axis_mm",
                                  "U_ring", "R0")},
                              meta={k: v for k, v in field.meta.items() if k != "_grids"})
    with open(os.path.join(out_dir, "field.pkl"), "wb") as fh:
        pickle.dump(field_to_pickle, fh)
    if mean is not None:
        with open(os.path.join(out_dir, "mean_field.pkl"), "wb") as fh:
            pickle.dump(mean, fh)

    meta = dict(field_to_pickle.meta)
    meta.update(y_axis_mm=field.y_axis_mm, U_ring=field.U_ring, R0=field.R0,
                bounds=field.bounds)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(_jsonable(meta), fh, indent=2)


def load_field(out_dir: str) -> Field3D:
    with open(os.path.join(out_dir, "field.pkl"), "rb") as fh:
        return pickle.load(fh)


def load_mean_field(out_dir: str):
    path = os.path.join(out_dir, "mean_field.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
