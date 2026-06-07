"""
digiflow_io.py — Stage A: PIV ingestion (DIMENSIONAL: mm, mm/s, 1/s).

Defines the common in-memory contract (:class:`PIVFrames`) and the loaders:

    * :func:`load_piv`        — read a folder of DigiFlow ``.dfi`` frames and
                                map pixels -> world (mm) with a coordinate csv.
    * :func:`load_escape_csv` — read an escape-size ``.csv`` -> bubble diameters.

The synthetic generator (``synthetic.py``) produces the same ``PIVFrames``
object, so the rest of the pipeline is source-agnostic.  All conversion to
dimensionless variables happens later, in ``nondimensional.py``.

Grid convention mirrors ``ellipseFit_example.ipynb``: ``X`` is streamwise (the
ring axis direction), ``Y`` is the in-plane transverse coordinate, and the ring
axis (r = 0) is a horizontal line ``Y = y_axis``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PIVFrames:
    """A stack of meridional PIV frames on a common world grid (mm, mm/s)."""

    X: np.ndarray            # (ny, nx) world x [mm]
    Y: np.ndarray            # (ny, nx) world y [mm]
    u: np.ndarray            # (nf, ny, nx) streamwise velocity [mm/s]
    v: np.ndarray            # (nf, ny, nx) transverse velocity [mm/s]
    omega: np.ndarray        # (nf, ny, nx) vorticity [1/s]
    t: np.ndarray            # (nf,) timestamps [s]
    meta: dict = field(default_factory=dict)

    @property
    def nframes(self) -> int:
        return self.u.shape[0]

    @property
    def dx(self) -> float:
        return float(self.X[0, 1] - self.X[0, 0])

    @property
    def dy(self) -> float:
        return float(self.Y[1, 0] - self.Y[0, 0])

    def __post_init__(self):
        for name in ("u", "v", "omega"):
            arr = getattr(self, name)
            if arr.ndim != 3:
                raise ValueError(f"{name} must be (nframes, ny, nx), got {arr.shape}")
        if not (self.u.shape == self.v.shape == self.omega.shape):
            raise ValueError("u, v, omega must share the same shape")


def _build_world_grid(ny, nx, mapping_x, mapping_y):
    """Apply the 8-term quadratic pixel->world map used by DigiFlow exports."""
    xp = np.arange(nx)
    yp = np.arange(ny)
    XP, YP = np.meshgrid(xp, yp)

    def _apply(c):
        return (c[0] + c[1] * XP + c[2] * XP ** 2 + c[3] * YP + c[4] * YP ** 2
                + c[5] * XP * YP + c[6] * XP ** 2 * YP + c[7] * XP * YP ** 2)

    return _apply(mapping_x), _apply(mapping_y)


def load_piv(station_dir, coord_file, pattern="*.dfi", orientation="none",
             dt=None, fps=None, max_frames=None, start=0, stop=None) -> PIVFrames:
    """Read ``.dfi`` frames under ``station_dir`` into a :class:`PIVFrames`.

    ``coord_file`` is the ``*_mapping.csv`` with two rows of 8 quadratic
    coefficients (row 0 -> world x, row 1 -> world y).  Frames are sorted
    lexicographically (zero-padded world index == temporal order).  Provide
    ``dt`` or ``fps`` for the timebase.  ``start:stop`` selects a frame window
    (the averaging window T_win); ``max_frames`` further caps the count.
    """
    import digiflowio as dfi   # project reader at repo root

    files = sorted(glob.glob(os.path.join(station_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {station_dir}")
    files = files[start:stop]
    if max_frames:
        files = files[:max_frames]
    if not files:
        raise ValueError(f"Empty frame window start={start} stop={stop}")

    mapping_x = np.loadtxt(coord_file, delimiter=",", skiprows=0, max_rows=1)
    mapping_y = np.loadtxt(coord_file, delimiter=",", skiprows=1, max_rows=1)

    u_list, v_list, w_list = [], [], []
    X = Y = None
    for f in files:
        img = dfi.read(f, orientation=orientation)
        u, v, omega = img.data
        if X is None:
            X, Y = _build_world_grid(u.shape[0], u.shape[1], mapping_x, mapping_y)
        u_list.append(u)
        v_list.append(v)
        w_list.append(omega)

    n = len(files)
    if dt is None:
        dt = 1.0 / fps if fps else 1.0
    t = np.arange(n) * dt

    return PIVFrames(
        X=X, Y=Y,
        u=np.stack(u_list), v=np.stack(v_list), omega=np.stack(w_list), t=t,
        meta={"source": station_dir, "coord_file": coord_file,
              "n_files": n, "dt": dt, "unit_dt": dt == 1.0 and fps is None},
    )


def autodetect_window(station_dir, coord_file, pattern="*.dfi", orientation="none",
                      ens_frac=0.50, margin_mm=20.0, drift_mm=10.0, stride=2,
                      max_frames=240, verbose=False):
    """Find a quasi-steady frame window with the ring centred in the FOV.

    A coherent ring is in view for much of a run but *decelerates* as it
    crosses, so U_ring is not constant over the whole traverse.  This scans
    every ``stride``-th frame (|omega|-weighted centroid + enstrophy), picks the
    frame where the centroid is closest to the FOV centre among coherent,
    well-inside frames, then expands while the centroid stays within
    ``drift_mm`` of that point — a short, near-constant-U_ring window.
    Returns ``(start, stop)`` file indices for :func:`load_piv`.
    """
    import digiflowio as dfi

    files = sorted(glob.glob(os.path.join(station_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {station_dir}")

    idx = list(range(0, len(files), stride))
    cx = np.full(len(idx), np.nan)
    ens = np.zeros(len(idx))
    inside = np.zeros(len(idx), bool)
    X = Y = bounds = xc_fov = None
    for j, i in enumerate(idx):
        _, _, omega = dfi.read(files[i], orientation=orientation).data
        if X is None:
            X, Y = _build_world_grid(omega.shape[0], omega.shape[1],
                                     np.loadtxt(coord_file, delimiter=",", max_rows=1),
                                     np.loadtxt(coord_file, delimiter=",", skiprows=1, max_rows=1))
            bounds = (X.min() + margin_mm, X.max() - margin_mm,
                      Y.min() + margin_mm, Y.max() - margin_mm)
            xc_fov = 0.5 * (X.min() + X.max())
        w = np.abs(omega)
        s = w.sum()
        ens[j] = float((omega ** 2).sum())
        if s > 0:
            cx[j] = (X * w).sum() / s
            cy = (Y * w).sum() / s
            inside[j] = (bounds[0] <= cx[j] <= bounds[1]) and (bounds[2] <= cy <= bounds[3])

    coherent = inside & (ens >= ens_frac * ens.max())
    if not coherent.any():
        raise RuntimeError("No coherent in-FOV frame; relax ens_frac/margin_mm")

    # frame whose centroid is nearest the FOV centre (ring best centred)
    dist = np.where(coherent, np.abs(cx - xc_fov), np.inf)
    k = int(np.argmin(dist))
    cx0 = cx[k]

    lo = k
    while lo - 1 >= 0 and coherent[lo - 1] and abs(cx[lo - 1] - cx0) <= drift_mm:
        lo -= 1
    hi = k
    while hi + 1 < len(idx) and coherent[hi + 1] and abs(cx[hi + 1] - cx0) <= drift_mm:
        hi += 1

    start = idx[lo]
    stop = min(idx[hi] + stride, len(files))
    clamped = False
    if stop - start > max_frames:   # slow ring -> huge window; cap around centre
        half = max_frames // 2
        start = max(idx[k] - half, 0)
        stop = min(idx[k] + half, len(files))
        clamped = True
    if verbose:
        print(f"[autodetect] {len(files)} frames; ring centred near frame {idx[k]} "
              f"(cx={cx0:.1f}mm, FOV centre {xc_fov:.1f}); window [{start}:{stop}] "
              f"({stop - start} frames, drift<= {drift_mm}mm"
              f"{', capped' if clamped else ''})")
    return start, stop


def load_escape_csv(path, y_center=None, y_halfwidth=100.0) -> np.ndarray:
    """Load an escape-size csv (cols ``X_world, Y_world, R_world``) and return
    bubble **diameters** in mm (``d = 2 * R_world``; the csv column is radius).

    The csv is in the GLOBAL bubble frame where ``Y_world`` is the downstream
    distance (5D/10D/15D = 200/400/600 mm, D=40 mm).  A single file spans many
    stations, so pass ``y_center`` (e.g. ``downstream_mm(10)``) to keep only
    bubbles with ``|Y_world - y_center| <= y_halfwidth`` — that station's
    escape-size distribution."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    r = np.asarray(data["R_world"], dtype=float)
    if y_center is not None:
        yw = np.asarray(data["Y_world"], dtype=float)
        r = r[np.abs(yw - y_center) <= y_halfwidth]
        if r.size == 0:
            raise ValueError(f"No bubbles within {y_halfwidth} mm of Y_world={y_center}")
    return 2.0 * r


def downstream_mm(n_D, D=40.0) -> float:
    """Downstream distance for an n-D station: ``Y_world = n_D * D`` (D=40 mm)."""
    return float(n_D) * D
