"""
seeding.py — Stage E: seed placement and size -> Stokes sampling.

    * :func:`fit_core_ellipse`  — quadratic vorticity fit -> elliptical core
      geometry (ported from ``ellipseFit_example.ipynb``).
    * :func:`seed_positions`    — place n seeds around the core and revolve the
      ring in azimuth; ``measure in {arclength, area, annulus}``.
    * :func:`sample_stokes`     — fit a log-normal to the measured escape
      diameters, draw sizes in ``[eps_plus, d_max]`` and map ``d -> St``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from averaging import MeanField
from digiflow_io import load_escape_csv
from vortex_core import find_vortex_cores_iterative
from constants import stokes_number, R0 as R0_DEFAULT


@dataclass
class CoreEllipse:
    """Elliptical core geometry in the meridional (x, r) plane."""
    x_c: float          # streamwise centre [mm]
    r_c: float          # radial centre (distance from axis) [mm]
    a_xi: float         # semi-axis along principal dir 1 [mm]
    a_eta: float        # semi-axis along principal dir 2 [mm]
    a_eq: float         # equivalent radius sqrt(a_xi a_eta) [mm]
    tilt: float         # major-axis angle wrt x [rad]
    omega_c: float      # peak vorticity [1/s]


def _quad(xy, a0, a1, a2, a3, a4, a5):
    xp, yp = xy
    return a0 + a1 * xp + a2 * xp ** 2 + a3 * xp * yp + a4 * yp + a5 * yp ** 2


def fit_core_ellipse(mean: MeanField, y_axis: float, core="upper",
                     fit_radius: float = 8.0, mask_radius: float = 7.0) -> CoreEllipse:
    """Fit a 2D quadratic to the vorticity near a core and extract the ellipse.

    ``core`` selects the upper (positive, r>0) or lower core.  The returned
    centre is expressed in meridional ``(x, r)`` with ``r`` measured from the
    ring axis.
    """
    cores, _, _ = find_vortex_cores_iterative(
        mean.X, mean.Y, mean.omega, fit_radius=fit_radius, verbose=False)
    (xp, yp, sp), (xn, yn, sn) = cores
    if core == "upper":
        xc, yc, sign = (xp, yp, sp) if yp >= yn else (xn, yn, sn)
    else:
        xc, yc, sign = (xn, yn, sn) if yn <= yp else (xp, yp, sp)

    dX = mean.X - xc
    dY = mean.Y - yc
    rho = np.sqrt(dX ** 2 + dY ** 2)
    m = rho <= mask_radius
    om = mean.omega[m]

    omega_c_guess = sign * np.abs(om).max()
    curv = -omega_c_guess / mask_radius ** 2
    p0 = [omega_c_guess, 0.0, curv, 0.0, 0.0, curv]
    popt, _ = curve_fit(_quad, (dX[m], dY[m]), om, p0=p0, maxfev=10000)

    a0, a1, a2, a3, a4, a5 = popt
    H = np.array([[2 * a2, a3], [a3, 2 * a5]])
    grad = np.array([a1, a4])
    x0, y0 = np.linalg.solve(H, -grad)
    omega_c = _quad((x0, y0), *popt)

    lam, ev = np.linalg.eig(H)
    a_xi = np.sqrt(-2 * omega_c / lam[0]) if omega_c * lam[0] < 0 else np.nan
    a_eta = np.sqrt(-2 * omega_c / lam[1]) if omega_c * lam[1] < 0 else np.nan
    a_eq = float(np.sqrt(a_xi * a_eta)) if np.isfinite(a_xi * a_eta) else np.nan
    # major-axis orientation = eigenvector of the smaller |curvature|
    major = ev[:, int(np.argmin(np.abs(lam)))]
    tilt = float(np.arctan2(major[1], major[0]))

    return CoreEllipse(
        x_c=float(xc + x0), r_c=float(abs((yc + y0) - y_axis)),
        a_xi=float(a_xi), a_eta=float(a_eta), a_eq=a_eq,
        tilt=tilt, omega_c=float(omega_c),
    )


# --------------------------------------------------------------------------
# Seed placement
# --------------------------------------------------------------------------
def _ellipse_boundary(ell: CoreEllipse, n: int, measure: str):
    """Return ``(x, r)`` meridional seed points for the chosen ``measure``."""
    c, s = np.cos(ell.tilt), np.sin(ell.tilt)
    R = np.array([[c, -s], [s, c]])

    if measure == "arclength":
        tt = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        pts = R @ np.vstack([ell.a_xi * np.cos(tt), ell.a_eta * np.sin(tt)])
        seg = np.hypot(np.diff(pts[0], append=pts[0, 0]),
                       np.diff(pts[1], append=pts[1, 0]))
        cum = np.concatenate([[0], np.cumsum(seg)[:-1]])
        targets = np.linspace(0, cum[-1], n, endpoint=False)
        idx = np.searchsorted(cum, targets)
        sel = pts[:, np.clip(idx, 0, pts.shape[1] - 1)]
        return ell.x_c + sel[0], ell.r_c + sel[1]

    if measure in ("area", "annulus"):
        rng = np.random.default_rng(0)
        lo = 0.0 if measure == "area" else 0.8
        # sqrt for uniform area density in the (lo,1) radial band
        rad = np.sqrt(rng.uniform(lo ** 2, 1.0, n))
        ang = rng.uniform(0, 2 * np.pi, n)
        local = R @ np.vstack([ell.a_xi * rad * np.cos(ang),
                               ell.a_eta * rad * np.sin(ang)])
        return ell.x_c + local[0], ell.r_c + local[1]

    raise ValueError(f"Unknown measure {measure!r}")


def seed_positions(ell: CoreEllipse, n: int = 24, n_phi: int = 16,
                   measure: str = "arclength") -> np.ndarray:
    """Place ``n`` meridional seeds and revolve them over ``n_phi`` azimuths.

    Returns Cartesian seed positions ``(n * n_phi, 3)`` with the ring axis
    along x and the radial plane ``(y, z)``."""
    xm, rm = _ellipse_boundary(ell, n, measure)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    P = []
    for x, r in zip(xm, rm):
        r = max(r, 0.0)
        P.append(np.column_stack([np.full_like(phi, x), r * np.cos(phi), r * np.sin(phi)]))
    return np.vstack(P)


# --------------------------------------------------------------------------
# Size -> Stokes
# --------------------------------------------------------------------------
@dataclass
class StokesSample:
    d: np.ndarray        # diameters [mm]
    St: np.ndarray       # Stokes numbers
    mu_log: float        # log-normal location (of ln d)
    sigma_log: float     # log-normal scale
    d_max: float         # largest observed escape diameter [mm]


def fit_lognormal(diameters: np.ndarray):
    """ML estimate of a log-normal for positive diameters: returns (mu, sigma)
    of ``ln d``."""
    d = np.asarray(diameters, dtype=float)
    d = d[d > 0]
    logs = np.log(d)
    return float(logs.mean()), float(logs.std(ddof=1))


def sample_stokes(n: int, csv_path: str, u_ring_mms: float,
                  eps_plus: float = None, R0_mm: float = R0_DEFAULT,
                  seed: int = 0) -> StokesSample:
    """Fit a log-normal to escape diameters and draw ``n`` sizes -> St.

    Sizes are drawn from a log-normal truncated to ``[eps_plus, d_max]`` where
    ``d_max`` is the largest observed escape diameter and ``eps_plus`` is the
    small positive lower cut (default: the smallest observed diameter), per
    build_plan §2 Stage E.
    """
    diameters = load_escape_csv(csv_path)
    mu, sigma = fit_lognormal(diameters)
    d_max = float(diameters.max())
    if eps_plus is None:
        eps_plus = float(diameters.min())

    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        draw = rng.lognormal(mu, sigma, n * 4)
        draw = draw[(draw >= eps_plus) & (draw <= d_max)]
        out.extend(draw.tolist())
    d = np.array(out[:n])

    St = stokes_number(d, u_ring_mms, R0_mm=R0_mm)
    return StokesSample(d=d, St=np.asarray(St), mu_log=mu, sigma_log=sigma, d_max=d_max)
