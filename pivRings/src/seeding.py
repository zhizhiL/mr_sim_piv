"""
seeding.py — Stage E: seed placement and size -> Stokes sampling.

    * :func:`fit_core_ellipse` — quadratic vorticity fit -> elliptical core
      geometry (ported from ``ellipseFit_example.ipynb``).  Fit is done on the
      DIMENSIONAL mean field; the centre/axes are also exposed in dimensionless
      (``/R0``) form for placement in the dimensionless field.
    * :func:`seed_positions`   — place n seeds around the core, revolve in
      azimuth; ``measure in {arclength, area, annulus}``.  Output is
      DIMENSIONLESS Cartesian (ring axis = x, radial plane = y,z).
    * :func:`sample_stokes`    — fit a log-normal to escape diameters, draw
      sizes in ``[eps_plus, d_max]`` and map ``d -> St`` (density-free, §3).
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
    """Elliptical core geometry.  Centre/axes stored in mm; ``R0`` lets the
    seeder convert to the dimensionless field frame."""
    x_c: float          # streamwise centre [mm]
    r_c: float          # radial centre (from axis) [mm]
    a_xi: float         # semi-axis 1 [mm]
    a_eta: float        # semi-axis 2 [mm]
    a_eq: float         # equivalent radius sqrt(a_xi a_eta) [mm]
    tilt: float         # major-axis angle wrt x [rad]
    omega_c: float      # peak vorticity [1/s]
    R0: float = R0_DEFAULT


def _quad(xy, a0, a1, a2, a3, a4, a5):
    xp, yp = xy
    return a0 + a1 * xp + a2 * xp ** 2 + a3 * xp * yp + a4 * yp + a5 * yp ** 2


def fit_core_ellipse(mean: MeanField, y_axis: float, core="upper",
                     fit_radius=8.0, mask_radius=7.0, R0=R0_DEFAULT) -> CoreEllipse:
    """Fit a 2D quadratic to the vorticity near a core and extract the ellipse."""
    cores, _, _ = find_vortex_cores_iterative(mean.X, mean.Y, mean.omega,
                                              fit_radius=fit_radius, verbose=False)
    (xp, yp, sp), (xn, yn, sn) = cores
    if core == "upper":
        xc, yc, sign = (xp, yp, sp) if yp >= yn else (xn, yn, sn)
    else:
        xc, yc, sign = (xn, yn, sn) if yn <= yp else (xp, yp, sp)

    dX, dY = mean.X - xc, mean.Y - yc
    rho = np.sqrt(dX ** 2 + dY ** 2)
    m = rho <= mask_radius
    om = mean.omega[m]
    p0 = [sign * np.abs(om).max(), 0.0, -sign * np.abs(om).max() / mask_radius ** 2,
          0.0, 0.0, -sign * np.abs(om).max() / mask_radius ** 2]
    popt, _ = curve_fit(_quad, (dX[m], dY[m]), om, p0=p0, maxfev=10000)

    a0, a1, a2, a3, a4, a5 = popt
    H = np.array([[2 * a2, a3], [a3, 2 * a5]])
    x0, y0 = np.linalg.solve(H, -np.array([a1, a4]))
    omega_c = _quad((x0, y0), *popt)
    lam, ev = np.linalg.eig(H)
    a_xi = np.sqrt(-2 * omega_c / lam[0]) if omega_c * lam[0] < 0 else np.nan
    a_eta = np.sqrt(-2 * omega_c / lam[1]) if omega_c * lam[1] < 0 else np.nan
    a_eq = float(np.sqrt(a_xi * a_eta)) if np.isfinite(a_xi * a_eta) else np.nan
    major = ev[:, int(np.argmin(np.abs(lam)))]
    tilt = float(np.arctan2(major[1], major[0]))

    return CoreEllipse(x_c=float(xc + x0), r_c=float(abs((yc + y0) - y_axis)),
                       a_xi=float(a_xi), a_eta=float(a_eta), a_eq=a_eq,
                       tilt=tilt, omega_c=float(omega_c), R0=float(R0))


def _ellipse_boundary(ell: CoreEllipse, n: int, measure: str):
    """Meridional seed points (x, r) in DIMENSIONLESS units for ``measure``."""
    c, s = np.cos(ell.tilt), np.sin(ell.tilt)
    R = np.array([[c, -s], [s, c]])
    xc, rc = ell.x_c / ell.R0, ell.r_c / ell.R0
    axi, aeta = ell.a_xi / ell.R0, ell.a_eta / ell.R0

    if measure == "arclength":
        tt = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        pts = R @ np.vstack([axi * np.cos(tt), aeta * np.sin(tt)])
        seg = np.hypot(np.diff(pts[0], append=pts[0, 0]), np.diff(pts[1], append=pts[1, 0]))
        cum = np.concatenate([[0], np.cumsum(seg)[:-1]])
        idx = np.searchsorted(cum, np.linspace(0, cum[-1], n, endpoint=False))
        sel = pts[:, np.clip(idx, 0, pts.shape[1] - 1)]
        return xc + sel[0], rc + sel[1]

    if measure in ("area", "annulus"):
        rng = np.random.default_rng(0)
        lo = 0.0 if measure == "area" else 0.8
        rad = np.sqrt(rng.uniform(lo ** 2, 1.0, n))
        ang = rng.uniform(0, 2 * np.pi, n)
        local = R @ np.vstack([axi * rad * np.cos(ang), aeta * rad * np.sin(ang)])
        return xc + local[0], rc + local[1]

    raise ValueError(f"Unknown measure {measure!r}")


def seed_positions(ell: CoreEllipse, n=24, n_phi=16, measure="arclength",
                   seed=None) -> np.ndarray:
    """Place ``n`` meridional seeds and revolve over ``n_phi`` azimuths.
    Returns DIMENSIONLESS Cartesian positions (n*n_phi, 3).

    ``seed`` (int) randomises the azimuthal angles and the meridional start
    phase so an ensemble of seeds samples the ring differently — used for
    multi-seed statistics."""
    xm, rm = _ellipse_boundary(ell, n, measure)
    if seed is None:
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        rng = None
    else:
        rng = np.random.default_rng(seed)
        phi = rng.uniform(0, 2 * np.pi, n_phi)        # random azimuths
        roll = rng.integers(0, n)                       # rotate meridional start
        xm, rm = np.roll(xm, roll), np.roll(rm, roll)
    P = []
    for x, r in zip(xm, rm):
        r = max(r, 0.0)
        ph = phi if rng is None else rng.uniform(0, 2 * np.pi, n_phi)
        P.append(np.column_stack([np.full_like(ph, x), r * np.cos(ph), r * np.sin(ph)]))
    return np.vstack(P)


# ==========================================================================
# Build-plan change #1 — seed on the CORE SURFACE, azimuth uniformly random
# ==========================================================================
def seed_core_surface(ell: CoreEllipse, n: int, rng) -> np.ndarray:
    """Place ``n`` seeds on the elliptical core SURFACE (build-plan change #1).

    The core is the vortex-ring tube; its surface is the meridional core ellipse
    revolved about the ring axis (a torus).  Each seed is drawn with BOTH angles
    uniformly random:

        * the poloidal angle ``theta`` around the core cross-section (the
          ellipse) — "the azimuth of the core";
        * the ring azimuth ``phi`` around the ring axis.

    so the population uniformly covers the whole torus surface.  Returns
    DIMENSIONLESS Cartesian positions ``(n, 3)`` (ring axis = x, radial plane =
    y,z), matching :func:`seed_positions`."""
    c, s = np.cos(ell.tilt), np.sin(ell.tilt)
    Rm = np.array([[c, -s], [s, c]])
    xc, rc = ell.x_c / ell.R0, ell.r_c / ell.R0
    axi, aeta = ell.a_xi / ell.R0, ell.a_eta / ell.R0

    theta = rng.uniform(0.0, 2 * np.pi, n)         # poloidal angle on the core ellipse
    local = Rm @ np.vstack([axi * np.cos(theta), aeta * np.sin(theta)])
    xm = xc + local[0]
    rm = np.maximum(rc + local[1], 0.0)            # radial distance from ring axis
    phi = rng.uniform(0.0, 2 * np.pi, n)           # ring azimuth
    return np.column_stack([xm, rm * np.cos(phi), rm * np.sin(phi)])


# ==========================================================================
# Build-plan change #2 — uniform-in-count discrete size distribution
# ==========================================================================
def make_radii_grid(r_min=0.06, r_max=0.9, step=0.04) -> np.ndarray:
    """Discrete bubble RADII (mm): ``r_min`` .. ``r_max`` inclusive, spacing
    ``step`` (build-plan change #2: 0.06 .. 0.90 mm by 0.04 -> 22 radii)."""
    n = int(round((r_max - r_min) / step)) + 1
    return r_min + step * np.arange(n)


@dataclass
class UniformSizeDist:
    """Uniform-in-count size distribution (build-plan change #2).

    Equal physical bubble count at EVERY radius on the grid; the common count is
    fixed by requiring the total injected bubble volume to equal ``loading_uL``
    (l1 = 20 uL, l3 = 40 uL).  Bubbles are spheres: ``v_b = 4/3 pi r^3`` [uL]."""
    radii: np.ndarray            # bubble radii [mm]
    diameters: np.ndarray        # 2*radii [mm] (St is a function of diameter)
    vol_each: np.ndarray         # per-bubble volume [uL = mm^3]
    count_per_radius: float      # equal physical count per radius bin (may be < 1)
    loading_uL: float

    @property
    def n_radii(self) -> int:
        return len(self.radii)

    @property
    def total_volume(self) -> float:
        return float(self.count_per_radius * self.vol_each.sum())


def uniform_size_distribution(loading_uL, r_min=0.06, r_max=0.9,
                              step=0.04) -> UniformSizeDist:
    """Build the uniform-in-count size distribution (build-plan change #2).

    ``count_per_radius = loading_uL / sum_i v_i`` so every radius is equally
    represented in COUNT while the total volume integrates to the experimental
    loading.  Pass ``loading_uL`` directly or use :func:`loading_from_csv_name`
    / :data:`LOADING_UL` to map the l1/l3 tag."""
    radii = make_radii_grid(r_min, r_max, step)
    vol_each = 4.0 / 3.0 * np.pi * radii ** 3
    count = float(loading_uL) / float(vol_each.sum())
    return UniformSizeDist(radii=radii, diameters=2.0 * radii, vol_each=vol_each,
                           count_per_radius=count, loading_uL=float(loading_uL))


@dataclass
class StokesSample:
    d: np.ndarray        # diameters [mm]
    St: np.ndarray       # Stokes numbers (density-free)
    mu_log: float
    sigma_log: float
    d_max: float
    V_initial: float = np.nan     # initial loading [mm^3 = uL]
    V_escaped_upstream: float = np.nan
    V_remaining: float = np.nan   # bubble volume still in the ring at the station
    n_physical: float = np.nan    # implied physical bubble count at the station


# initial bubble loading by CSV label: l1 = 20 uL, l3 = 40 uL  (1 uL = 1 mm^3)
LOADING_UL = {"l1": 20.0, "l3": 40.0}


def loading_from_csv_name(path) -> float:
    """Initial loading (uL) inferred from the ``_l1``/``_l3`` filename tag."""
    name = str(path).lower()
    for tag, vol in LOADING_UL.items():
        if f"_{tag}" in name:
            return vol
    raise ValueError(f"Cannot infer loading (l1/l3) from {path!r}; pass initial_loading_uL")


def fit_lognormal(diameters):
    d = np.asarray(diameters, float)
    d = d[d > 0]
    logs = np.log(d)
    return float(logs.mean()), float(logs.std(ddof=1))


def _vol(d):
    """Sphere volume(s) for diameter(s) d [mm] -> mm^3."""
    return np.pi / 6.0 * np.asarray(d, float) ** 3


def first_escape_after(csv_path, station_D):
    """(d, Y_world) of the nearest escape downstream of ``station_D`` — the
    largest bubble still present at the station (big bubbles escape first)."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    yw = np.asarray(data["Y_world"], float)
    d = 2.0 * np.asarray(data["R_world"], float)
    after = yw >= station_D
    if not after.any():
        return None, None
    k = np.argmin(np.where(after, yw, np.inf))
    return float(d[k]), float(yw[k])


def sample_stokes(n, csv_path, u_ring_mms, eps_plus=None, R0_mm=R0_DEFAULT,
                  seed=0, station_D=None, initial_loading_uL=None,
                  y_center=None, y_halfwidth=100.0) -> StokesSample:
    """Fit a log-normal to escape diameters and draw ``n`` sizes -> St.

    St is density-free: ``d^2 U_ring / (18 nu R0)`` (dimension_conversion.md §3).

    Volume-conservation mode (pass ``station_D``, the downstream Y_world of the
    PIV station):
      * upper truncation ``d_max`` = the **first escape downstream of D** (the
        largest bubble still in the ring; big bubbles escape first);
      * ``V_remaining = V_initial - sum(escaped volume with Y_world < D)`` with
        ``V_initial`` = ``initial_loading_uL`` (or inferred l1=20/l3=40 uL);
      * the implied physical bubble count ``n_physical = V_remaining / E[vol]``
        is reported (the size distribution integrates to V_remaining).
    Without ``station_D`` it falls back to the plain fit over ``[eps_plus,
    max d]`` (optionally Y_world-banded via ``y_center``)."""
    # the lognormal shape is fit to the observed escape sizes
    diameters = load_escape_csv(csv_path, y_center=y_center, y_halfwidth=y_halfwidth)
    mu, sigma = fit_lognormal(diameters)
    if eps_plus is None:
        eps_plus = float(diameters.min())

    V_initial = V_esc = V_rem = n_phys = np.nan
    if station_D is not None:
        if initial_loading_uL is None:
            initial_loading_uL = loading_from_csv_name(csv_path)
        V_initial = float(initial_loading_uL)
        all_d = load_escape_csv(csv_path)
        yw = np.asarray(np.genfromtxt(csv_path, delimiter=",", names=True)["Y_world"], float)
        V_esc = float(_vol(all_d[yw < station_D]).sum())
        V_rem = max(V_initial - V_esc, 0.0)
        d_after, _ = first_escape_after(csv_path, station_D)
        d_max = d_after if d_after is not None else float(diameters.max())
    else:
        d_max = float(diameters.max())

    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        draw = rng.lognormal(mu, sigma, n * 4)
        draw = draw[(draw >= eps_plus) & (draw <= d_max)]
        out.extend(draw.tolist())
    d = np.array(out[:n])

    if station_D is not None and np.isfinite(V_rem):
        n_phys = V_rem / float(_vol(d).mean())   # distribution integrates to V_remaining

    return StokesSample(d=d, St=np.asarray(stokes_number(d, u_ring_mms, R0_mm)),
                        mu_log=mu, sigma_log=sigma, d_max=d_max,
                        V_initial=V_initial, V_escaped_upstream=V_esc,
                        V_remaining=V_rem, n_physical=n_phys)
