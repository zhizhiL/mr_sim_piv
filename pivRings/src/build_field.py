"""
build_field.py — Stage D: build & cache the quasi-steady 3D field.

From a smoothed meridional mean field this module:

    1. locates the ring axis (r = 0),
    2. folds the two half-planes onto a common ``(x, r>=0)`` grid,
    3. computes the four meridional gradients,
    4. revolves to a 3D axisymmetric Cartesian field with a chain-rule
       conversion that **guards the r -> 0 axis singularity**, and
    5. caches everything under ``fields/<station>/`` with one path convention.

The revolved field is exposed as :class:`Field3D`, which returns the fluid
velocity and its full 3x3 gradient tensor at arbitrary Cartesian points — this
is exactly what the Maxey-Riley RHS needs for ``Du/Dt = (u . grad) u``.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline

from averaging import MeanField

# radial distance below which we switch to the on-axis analytic limit
_AXIS_EPS = 1.0e-6


def locate_axis(mean: MeanField, y_axis: float = None) -> float:
    """Return the world-y of the ring axis.

    If ``y_axis`` is given it is used directly; otherwise it is taken from the
    transverse location where the streamwise velocity is extremal along the
    centre column (a robust proxy for the reflection-symmetry line of an
    axisymmetric ring)."""
    if y_axis is not None:
        return float(y_axis)
    # symmetry estimate: row minimising the up/down asymmetry of |omega|
    yvals = mean.Y[:, 0]
    absw = np.abs(mean.omega)
    col_profile = absw.mean(axis=1)
    # centre of mass of |omega| in y is a stable axis estimate for a ring
    return float(np.sum(yvals * col_profile) / (np.sum(col_profile) + 1e-12))


def _fold_halfplanes(mean: MeanField, y_axis: float, n_r: int = None):
    """Fold upper/lower halves onto a common ``(r>=0, x)`` grid.

    Returns ``(x_axis, r_axis, Ux, Ur)`` with arrays shaped ``(n_r, n_x)``.
    Radial velocity ``Ur`` is positive **outward** from the axis.
    """
    x_axis = mean.X[0, :].astype(float)
    yvals = mean.Y[:, 0].astype(float)

    r_signed = yvals - y_axis
    upper = r_signed >= 0
    lower = r_signed <= 0

    r_up = r_signed[upper]
    r_lo = -r_signed[lower]
    order_up = np.argsort(r_up)
    order_lo = np.argsort(r_lo)

    Ux_up, Uy_up = mean.Ux[upper][order_up], mean.Uy[upper][order_up]
    Ux_lo, Uy_lo = mean.Ux[lower][order_lo], mean.Uy[lower][order_lo]
    r_up, r_lo = r_up[order_up], r_lo[order_lo]

    r_max = min(r_up.max(), r_lo.max())
    if n_r is None:
        n_r = max(len(r_up), len(r_lo))
    r_axis = np.linspace(0.0, r_max, n_r)

    Ux = np.zeros((n_r, len(x_axis)))
    Ur = np.zeros((n_r, len(x_axis)))
    for j in range(len(x_axis)):
        ux_u = np.interp(r_axis, r_up, Ux_up[:, j])
        ux_l = np.interp(r_axis, r_lo, Ux_lo[:, j])
        # outward radial velocity: +v in upper half, -v in lower half
        ur_u = np.interp(r_axis, r_up, Uy_up[:, j])
        ur_l = np.interp(r_axis, r_lo, -Uy_lo[:, j])
        Ux[:, j] = 0.5 * (ux_u + ux_l)
        Ur[:, j] = 0.5 * (ur_u + ur_l)

    # radial velocity must vanish on the axis
    Ur[0, :] = 0.0
    return x_axis, r_axis, Ux, Ur


@dataclass
class Field3D:
    """Revolved 3D axisymmetric field backed by meridional splines.

    The ring axis is the world-x direction; the radial plane is ``(y, z)`` with
    ``r = sqrt(y^2 + z^2)``.  All splines are functions of ``(r, x)``.
    """
    x_axis: np.ndarray
    r_axis: np.ndarray
    sp_Ux: RectBivariateSpline
    sp_Ur: RectBivariateSpline
    sp_dUxdx: RectBivariateSpline
    sp_dUxdr: RectBivariateSpline
    sp_dUrdx: RectBivariateSpline
    sp_dUrdr: RectBivariateSpline
    y_axis: float = 0.0
    meta: dict = None

    @property
    def bounds(self):
        """(x_min, x_max, r_min, r_max) of the valid meridional domain."""
        return (float(self.x_axis[0]), float(self.x_axis[-1]),
                float(self.r_axis[0]), float(self.r_axis[-1]))

    def in_domain(self, P: np.ndarray) -> np.ndarray:
        """Boolean mask: which Cartesian points lie inside the PIV FOV."""
        P = np.atleast_2d(P)
        x = P[:, 0]
        r = np.sqrt(P[:, 1] ** 2 + P[:, 2] ** 2)
        xmin, xmax, rmin, rmax = self.bounds
        return (x >= xmin) & (x <= xmax) & (r >= rmin) & (r <= rmax)

    def velocity(self, P: np.ndarray) -> np.ndarray:
        """Fluid velocity (N,3) in Cartesian at points ``P`` (N,3)."""
        P = np.atleast_2d(P)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        r = np.sqrt(y * y + z * z)
        rs = np.maximum(r, _AXIS_EPS)
        Ux = self.sp_Ux.ev(r, x)
        Ur = self.sp_Ur.ev(r, x)
        ey, ez = y / rs, z / rs
        u = np.empty_like(P)
        u[:, 0] = Ux
        u[:, 1] = Ur * ey
        u[:, 2] = Ur * ez
        return u

    def velocity_and_gradient(self, P: np.ndarray):
        """Return ``(u, gradu)`` with ``u`` shape (N,3) and ``gradu`` (N,3,3),
        where ``gradu[n,i,j] = d u_i / d x_j``.

        On the axis (``r < _AXIS_EPS``) the analytic limit is used in place of
        the 1/r, 1/r^3 terms (the guard the build plan asks for)."""
        P = np.atleast_2d(P)
        N = P.shape[0]
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        r = np.sqrt(y * y + z * z)
        on_axis = r < _AXIS_EPS
        rs = np.where(on_axis, 1.0, r)  # safe denominator

        Ux = self.sp_Ux.ev(r, x)
        Ur = self.sp_Ur.ev(r, x)
        dUxdx = self.sp_dUxdx.ev(r, x)
        dUxdr = self.sp_dUxdr.ev(r, x)
        dUrdx = self.sp_dUrdx.ev(r, x)
        dUrdr = self.sp_dUrdr.ev(r, x)

        ey, ez = y / rs, z / rs

        u = np.empty((N, 3))
        u[:, 0] = Ux
        u[:, 1] = np.where(on_axis, 0.0, Ur * ey)
        u[:, 2] = np.where(on_axis, 0.0, Ur * ez)

        g = np.zeros((N, 3, 3))
        # d u_x / d{x,y,z}
        g[:, 0, 0] = dUxdx
        g[:, 0, 1] = np.where(on_axis, 0.0, dUxdr * ey)
        g[:, 0, 2] = np.where(on_axis, 0.0, dUxdr * ez)
        # d u_y / d{x,y,z}
        g[:, 1, 0] = np.where(on_axis, 0.0, dUrdx * ey)
        g[:, 1, 1] = np.where(on_axis, dUrdr,
                              dUrdr * ey * ey + Ur * (z * z) / rs ** 3)
        g[:, 1, 2] = np.where(on_axis, 0.0,
                              dUrdr * ey * ez - Ur * (y * z) / rs ** 3)
        # d u_z / d{x,y,z}
        g[:, 2, 0] = np.where(on_axis, 0.0, dUrdx * ez)
        g[:, 2, 1] = np.where(on_axis, 0.0,
                              dUrdr * ez * ey - Ur * (z * y) / rs ** 3)
        g[:, 2, 2] = np.where(on_axis, dUrdr,
                              dUrdr * ez * ez + Ur * (y * y) / rs ** 3)
        return u, g


def build_field(mean: MeanField, y_axis: float = None, n_r: int = None,
                spline_k: int = 3) -> Field3D:
    """Fold, differentiate and revolve a :class:`MeanField` into a
    :class:`Field3D`."""
    y0 = locate_axis(mean, y_axis)
    x_axis, r_axis, Ux, Ur = _fold_halfplanes(mean, y0, n_r=n_r)

    # gradients on the (r, x) grid: axis 0 = r, axis 1 = x
    dUxdr, dUxdx = np.gradient(Ux, r_axis, x_axis)
    dUrdr, dUrdx = np.gradient(Ur, r_axis, x_axis)

    k = min(spline_k, len(r_axis) - 1, len(x_axis) - 1)

    def _sp(Z):
        return RectBivariateSpline(r_axis, x_axis, Z, kx=k, ky=k)

    return Field3D(
        x_axis=x_axis, r_axis=r_axis,
        sp_Ux=_sp(Ux), sp_Ur=_sp(Ur),
        sp_dUxdx=_sp(dUxdx), sp_dUxdr=_sp(dUxdr),
        sp_dUrdx=_sp(dUrdx), sp_dUrdr=_sp(dUrdr),
        y_axis=y0,
        meta={"n_r": len(r_axis), "n_x": len(x_axis),
              **(mean.meta or {})},
    )


def save_field(field: Field3D, out_dir: str, arrays: dict = None, mean=None):
    """Cache a :class:`Field3D` under ``out_dir`` (one path convention).

    Writes the raw arrays as ``.npy``, the meta as ``meta.json`` and the
    pickled spline tuple as ``interpolators.pkl`` so a worker process can
    reload the field without recomputation.  If ``mean`` (the meridional
    :class:`averaging.MeanField`) is given it is pickled too, so the seeding
    stage can refit the core ellipse from a cached field alone."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "x_axis.npy"), field.x_axis)
    np.save(os.path.join(out_dir, "r_axis.npy"), field.r_axis)

    # also persist the gridded fields for inspection / plotting
    if arrays:
        for name, arr in arrays.items():
            np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    with open(os.path.join(out_dir, "interpolators.pkl"), "wb") as fh:
        pickle.dump(field, fh)

    if mean is not None:
        with open(os.path.join(out_dir, "mean_field.pkl"), "wb") as fh:
            pickle.dump(mean, fh)

    meta = dict(field.meta or {})
    meta["y_axis"] = field.y_axis
    meta["bounds"] = field.bounds
    # truth dicts may contain tuples; json handles lists only
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(_jsonable(meta), fh, indent=2)


def load_field(out_dir: str) -> Field3D:
    """Reload a cached :class:`Field3D`."""
    with open(os.path.join(out_dir, "interpolators.pkl"), "rb") as fh:
        return pickle.load(fh)


def load_mean_field(out_dir: str):
    """Reload the cached meridional :class:`averaging.MeanField` (or None)."""
    path = os.path.join(out_dir, "mean_field.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _jsonable(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (_np.floating, _np.integer)):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj
