"""
separatrix.py — co-moving Stokes streamfunction + atmosphere separatrix polygon.

The FOV-rectangle escape test (``advect._escape_event``) is FOV- and
``t_max``-dependent (REPORT §7d).  This module replaces it with an escape
definition tied to the *vortex structure*: the **co-moving Stokes streamfunction
separatrix** (decision REPORT §7e).  Physically it IS the atmosphere boundary —
the dividing streamline of the recirculating bubble in the ring's frame — so a
detraining bubble literally crosses it.

Axisymmetric Stokes streamfunction (x along the ring axis, r the meridional
radius), defined so it vanishes on the axis::

    u_x = (1/r) dψ/dr ,    u_r = -(1/r) dψ/dx          (ψ = 0 on r = 0)

The naive radial integral ``ψ(x,r) = ∫₀ʳ r' u_x dr'`` is *exact in u_x* but
ignores u_r, and PIV mean fields are only approximately solenoidal, so that ψ is
slightly path-dependent (REPORT §7e gotcha).  We instead solve the
**least-squares streamfunction**: the ψ whose gradient best matches *both*
components,

    minimise  Σ |∂ψ/∂x + r u_r|² + |∂ψ/∂r - r u_x|² ,   ψ|_{r=0} = 0,

which is the discrete Helmholtz/solenoidal projection of the field onto a pure
streamfunction (the curl-free residual is discarded).  ``method='integral'``
keeps the naive form for comparison.

The separatrix is the ``ψ = ψ_sep`` contour through the rear on-axis stagnation
saddle (ψ_sep ≈ 0 since the saddle sits on the axis).  Because ψ = 0 is *both*
the axis and the separatrix, we never sign-test ψ directly — we extract the
off-axis closed loop enclosing the core as a polygon and do point-in-polygon
along an already-integrated trajectory (REPORT §7e step 4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr


# --------------------------------------------------------------------------
# Streamfunction
# --------------------------------------------------------------------------
def _grid_velocity(field, xg, rg):
    """(Ux, Ur) on the meridional (x, r) grid, shapes (nr, nx)."""
    Xg, Rg = np.meshgrid(xg, rg)            # (nr, nx)
    Ux = field.sp_Ux.ev(Xg.ravel(), Rg.ravel()).reshape(Xg.shape)
    Ur = field.sp_Ur.ev(Xg.ravel(), Rg.ravel()).reshape(Xg.shape)
    return Xg, Rg, Ux, Ur


def stream_function(field, nx=160, nr=140, method="leastsq", pad=0.0):
    """Co-moving Stokes streamfunction ψ(x, r) on a meridional grid.

    ``method='leastsq'`` (default) solves the gradient-matching (solenoidal-
    projection) least-squares problem; ``method='integral'`` uses the naive
    ψ = ∫₀ʳ r' u_x dr'.  Returns ``(xg, rg, psi)`` with ``psi`` shape (nr, nx),
    ψ = 0 on the axis row (r = rg[0])."""
    xmin, xmax, rmin, rmax = field.bounds
    xg = np.linspace(xmin + pad, xmax - pad, nx)
    rg = np.linspace(max(rmin, 0.0) + pad, rmax - pad, nr)
    Xg, Rg, Ux, Ur = _grid_velocity(field, xg, rg)

    if method == "integral":
        # cumulative trapezoid of r' u_x in r, per x-column (axis row -> 0)
        integrand = Rg * Ux                          # (nr, nx)
        psi = np.zeros_like(integrand)
        dr = np.diff(rg)
        psi[1:, :] = np.cumsum(0.5 * (integrand[1:, :] + integrand[:-1, :])
                               * dr[:, None], axis=0)
        return xg, rg, psi

    if method != "leastsq":
        raise ValueError(f"unknown method {method!r}")

    # Least-squares ψ: match ∂ψ/∂x = -r u_r and ∂ψ/∂r = r u_x, with ψ=0 on axis.
    # Unknowns ψ[i, j], i over r (nr), j over x (nx), flat index k = i*nx + j.
    dx = xg[1] - xg[0]
    dr = rg[1] - rg[0]
    Gx = -Rg * Ur            # target ∂ψ/∂x
    Gr = Rg * Ux             # target ∂ψ/∂r

    def k(i, j):
        return i * nx + j

    n_unknown = nr * nx
    # rows: x-edges (nr*(nx-1)) + r-edges ((nr-1)*nx) + axis pin (nx)
    n_eq = nr * (nx - 1) + (nr - 1) * nx + nx
    A = lil_matrix((n_eq, n_unknown))
    b = np.zeros(n_eq)
    row = 0
    # x-gradient equations on cell edges (midpoint target)
    for i in range(nr):
        for j in range(nx - 1):
            A[row, k(i, j + 1)] = 1.0 / dx
            A[row, k(i, j)] = -1.0 / dx
            b[row] = 0.5 * (Gx[i, j] + Gx[i, j + 1])
            row += 1
    # r-gradient equations on cell edges
    for i in range(nr - 1):
        for j in range(nx):
            A[row, k(i + 1, j)] = 1.0 / dr
            A[row, k(i, j)] = -1.0 / dr
            b[row] = 0.5 * (Gr[i, j] + Gr[i + 1, j])
            row += 1
    # axis pin ψ[0, j] = 0  (heavily weighted -> effectively hard)
    w_pin = 1.0e3
    for j in range(nx):
        A[row, k(0, j)] = w_pin
        b[row] = 0.0
        row += 1

    sol = lsqr(A.tocsr(), b, atol=1e-10, btol=1e-10, iter_lim=20000)
    psi = sol[0].reshape(nr, nx)
    psi -= psi[0, :].mean()                  # re-zero the axis exactly
    return xg, rg, psi


# --------------------------------------------------------------------------
# Stagnation / separatrix extraction
# --------------------------------------------------------------------------
def axis_stagnation_x(field, n=600):
    """On-axis (r->0) stagnation x-locations: sign changes of co-moving Ux*(x,0).

    Returns the x* values (front + rear for a closed atmosphere).  Uses a small
    off-axis r to stay clear of the on-axis spline limit, matching
    ``run_ftle._axis_stagnation_count``."""
    xg = np.linspace(field.x_axis[0], field.x_axis[-1], n)
    P = np.column_stack([xg, np.zeros_like(xg), np.full_like(xg, 1e-4)])
    ux = field.velocity(P)[:, 0]
    sgn = np.sign(ux)
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    xs = []
    for i in idx:                            # linear root between samples
        x0, x1, u0, u1 = xg[i], xg[i + 1], ux[i], ux[i + 1]
        xs.append(x0 + (0.0 - u0) * (x1 - x0) / (u1 - u0 - 1e-30))
    return np.array(xs)


def _polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _contour_segments(xg, rg, psi, level):
    """Level-set polylines via contourpy (no axes / pyplot state)."""
    from contourpy import contour_generator
    Xg, Rg = np.meshgrid(xg, rg)
    cg = contour_generator(Xg, Rg, psi)
    segs = cg.lines(level)                    # list of (M, 2) [x, r] arrays
    return [np.asarray(s) for s in segs if len(s) >= 4]


def _upper_branch(loop):
    """Upper (larger-r) arc of a closed loop, ordered by ascending x.

    A vortex atmosphere loop is double-valued in x (an upper dividing-streamline
    arc and a lower return).  Split at the x-extreme vertices and keep the arc
    with the larger mean r — the dividing streamline over the core."""
    n = len(loop)
    i_lo = int(np.argmin(loop[:, 0]))
    rot = loop[(np.arange(i_lo, i_lo + n)) % n]
    j_hi = int(np.argmax(rot[:, 0]))
    a, b = rot[: j_hi + 1], rot[j_hi:]
    upper = a if a[:, 1].mean() >= b[:, 1].mean() else b
    if upper[0, 0] > upper[-1, 0]:
        upper = upper[::-1]
    return upper


def _close_to_axis(loop):
    """Atmosphere polygon = dividing-streamline arc closed along the axis (r=0).

    The recirculating bubble is bounded below by the axis and above by the
    dividing streamline between the front/rear stagnation saddles, so the
    physically-correct interior includes the near-axis strip (without this, a
    captured bubble that dips toward the axis is spuriously flagged as escaped)."""
    up = _upper_branch(loop)
    return np.vstack([up, [up[-1, 0], 0.0], [up[0, 0], 0.0], up[0]])


def extract_separatrix(xg, rg, psi, core_xr, level_eps=2e-3):
    """Atmosphere separatrix polygon enclosing ``core_xr=(x_c, r_c)``.

    The separatrix is ψ = ψ_sep ≈ 0, but ψ = 0 is degenerate with the axis, so
    we step the contour a hair into the atmosphere — ``level = s·eps·|ψ|_core``
    with ``s = sign(ψ_core)`` — take the largest closed loop enclosing the core,
    then close its dividing-streamline arc along the axis (:func:`_close_to_axis`).
    Returns ``(poly, level)`` with ``poly`` an (M, 2) closed (x, r) array, or
    ``(None, level)`` if no enclosing loop exists."""
    # ψ at the core center (nearest grid node) sets the interior sign + scale
    j = int(np.clip(np.searchsorted(xg, core_xr[0]), 1, len(xg) - 1))
    i = int(np.clip(np.searchsorted(rg, core_xr[1]), 1, len(rg) - 1))
    psi_core = psi[i, j]
    s = np.sign(psi_core) or 1.0
    level = float(s * level_eps * abs(psi_core))

    segs = _contour_segments(xg, rg, psi, level)
    core_pt = np.array(core_xr)
    best, best_area = None, 0.0
    for seg in segs:
        loop = seg if np.allclose(seg[0], seg[-1]) else np.vstack([seg, seg[0]])
        if len(loop) < 5 or not MplPath(loop).contains_point(core_pt):
            continue
        area = _polygon_area(loop)
        if area > best_area:
            best, best_area = loop, area
    if best is None:
        return None, level
    return _close_to_axis(best), level


# --------------------------------------------------------------------------
# Separatrix object (cacheable, point-in-polygon test)
# --------------------------------------------------------------------------
@dataclass
class Separatrix:
    """Cached atmosphere boundary: the (x, r) separatrix polygon + metadata."""
    poly: np.ndarray                 # (M, 2) closed (x, r) loop
    level: float                     # ψ contour level used
    xg: np.ndarray
    rg: np.ndarray
    psi: np.ndarray
    method: str = "leastsq"
    fov_truncated: bool = False      # loop touches the FOV bound (atmosphere>FOV)
    meta: dict = None

    @property
    def path(self) -> MplPath:
        if getattr(self, "_path", None) is None:
            self._path = MplPath(self.poly)
        return self._path

    def contains_xr(self, xr):
        """Point-in-polygon for meridional points ``xr`` of shape (N, 2) [x, r]."""
        return self.path.contains_points(np.atleast_2d(xr))

    def contains_xyz(self, P):
        """Point-in-polygon for 3-D points P (N,3); maps to (x, r=hypot(y,z))."""
        P = np.atleast_2d(P)
        xr = np.column_stack([P[:, 0], np.hypot(P[:, 1], P[:, 2])])
        return self.contains_xr(xr)

    def area(self):
        return _polygon_area(self.poly)

    def save(self, path):
        np.savez(path, poly=self.poly, level=self.level, xg=self.xg, rg=self.rg,
                 psi=self.psi, method=self.method, fov_truncated=self.fov_truncated)

    @staticmethod
    def load(path):
        if not path.endswith(".npz"):
            path = path + ".npz"
        d = np.load(path, allow_pickle=True)
        return Separatrix(poly=d["poly"], level=float(d["level"]), xg=d["xg"],
                          rg=d["rg"], psi=d["psi"], method=str(d["method"]),
                          fov_truncated=bool(d["fov_truncated"]))


def compute_separatrix(field, core_xr=None, mean=None, core="upper",
                       nx=160, nr=140, method="leastsq", level_eps=5e-3,
                       fov_tol=0.02):
    """Build the :class:`Separatrix` for ``field``.

    ``core_xr`` (the dimensionless core center used to pick the enclosing loop)
    may be given directly; otherwise it is fit from ``mean`` via
    :func:`seeding.fit_core_ellipse`.  Flags ``fov_truncated`` when the loop comes
    within ``fov_tol`` of a FOV wall (atmosphere > FOV -> REPORT §7e fallback)."""
    if core_xr is None:
        if mean is None:
            raise ValueError("need core_xr or mean to locate the core")
        import seeding as sd
        ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0,
                                  core=core)
        core_xr = (ell.x_c / field.R0, ell.r_c / field.R0)

    xg, rg, psi = stream_function(field, nx=nx, nr=nr, method=method)
    poly, level = extract_separatrix(xg, rg, psi, core_xr, level_eps=level_eps)
    if poly is None:
        return None

    xmin, xmax, rmin, rmax = field.bounds
    near = (poly[:, 0].min() - xmin < fov_tol or xmax - poly[:, 0].max() < fov_tol
            or rmax - poly[:, 1].max() < fov_tol)
    xs = axis_stagnation_x(field)
    return Separatrix(poly=poly, level=level, xg=xg, rg=rg, psi=psi,
                      method=method, fov_truncated=bool(near),
                      meta={"core_xr": tuple(core_xr), "axis_stagnation_x": xs.tolist(),
                            "n_axis_stagnation": int(len(xs))})


def load_or_compute(field_dir, field, **kw):
    """Load ``<field_dir>/separatrix.npz`` if present, else compute & cache it."""
    path = os.path.join(field_dir, "separatrix.npz")
    if os.path.exists(path):
        return Separatrix.load(path)
    mean = None
    try:
        from build_field import load_mean_field
        mean = load_mean_field(field_dir)
    except Exception:
        pass
    sep = compute_separatrix(field, mean=mean, **kw)
    if sep is not None:
        sep.save(path)
    return sep
