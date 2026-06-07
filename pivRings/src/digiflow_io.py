"""
digiflow_io.py — Stage A: PIV ingestion.

Defines the common in-memory contract used by every later stage
(:class:`PIVFrames`) and the loaders that fill it:

    * :func:`load_piv`        — read a folder of DigiFlow ``.dfi`` frames and
                                map pixels -> world (mm) with a coordinate csv.
    * :func:`load_escape_csv` — read an escape-size ``.csv`` -> bubble diameters.

The synthetic generator (``synthetic.py``) produces the *same* ``PIVFrames``
object, so the rest of the pipeline is agnostic to where the field came from.

All world coordinates are in mm, velocities in mm/s, vorticity in 1/s.
The grid convention mirrors ``ellipseFit_example.ipynb``: ``X`` is streamwise
(the ring axis direction) and ``Y`` is the in-plane transverse coordinate;
the ring axis (r = 0) is a horizontal line ``Y = y_axis``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PIVFrames:
    """A stack of meridional PIV frames on a common world grid.

    Attributes
    ----------
    X, Y : (ny, nx) ndarray
        World coordinate grids in mm (``np.meshgrid`` ``xy`` indexing).
    u, v : (nframes, ny, nx) ndarray
        Streamwise (x) and transverse (y) velocity in mm/s.
    omega : (nframes, ny, nx) ndarray
        Out-of-plane vorticity in 1/s.
    t : (nframes,) ndarray
        Per-frame timestamps in s.
    meta : dict
        Free-form provenance (station label, source paths, ...).
    """

    X: np.ndarray
    Y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    t: np.ndarray
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


# --------------------------------------------------------------------------
# Pixel -> world coordinate mapping (quadratic, from the example notebook)
# --------------------------------------------------------------------------
def _build_world_grid(ny: int, nx: int, mapping_x: np.ndarray, mapping_y: np.ndarray):
    """Apply the 8-term quadratic pixel->world map used by DigiFlow exports."""
    xp = np.arange(nx)
    yp = np.arange(ny)
    XP, YP = np.meshgrid(xp, yp)

    def _apply(c):
        return (c[0] + c[1] * XP + c[2] * XP ** 2 + c[3] * YP + c[4] * YP ** 2
                + c[5] * XP * YP + c[6] * XP ** 2 * YP + c[7] * XP * YP ** 2)

    return _apply(mapping_x), _apply(mapping_y)


def load_piv(station_dir: str,
             coord_file: str,
             pattern: str = "*.dfi",
             orientation: str = "none",
             dt: Optional[float] = None,
             fps: Optional[float] = None) -> PIVFrames:
    """Read every ``.dfi`` frame under ``station_dir`` into a :class:`PIVFrames`.

    Parameters
    ----------
    station_dir : str
        Directory containing the per-frame ``.dfi`` files (e.g. the
        ``Camera_1`` folder for a station).
    coord_file : str
        Path to the ``*_mapping.csv`` with two rows of 8 quadratic
        coefficients (row 0 -> world x, row 1 -> world y).
    pattern : str
        Glob for the frame files, sorted lexicographically (the world index
        ``_0001`` ... is zero padded, so lexicographic == temporal order).
    dt, fps : float, optional
        Frame spacing.  Provide one; if neither is given a unit dt = 1 s is
        used and a warning meta flag is set.

    Notes
    -----
    Relies on ``digiflowio`` (the project's ``.dfi`` reader) being importable.
    This path is for *local* runs against the real data on ``/mnt/d/...``;
    it is never exercised by the synthetic demo.
    """
    import digiflowio as dfi  # project reader at repo root

    files = sorted(glob.glob(os.path.join(station_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {station_dir}")

    mapping_x = np.loadtxt(coord_file, delimiter=",", skiprows=0, max_rows=1)
    mapping_y = np.loadtxt(coord_file, delimiter=",", skiprows=1, max_rows=1)

    u_list, v_list, w_list = [], [], []
    X = Y = None
    for f in files:
        img = dfi.read(f, orientation=orientation)
        u, v, omega = img.data  # three planes
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
        u=np.stack(u_list), v=np.stack(v_list), omega=np.stack(w_list),
        t=t,
        meta={"source": station_dir, "coord_file": coord_file,
              "n_files": n, "dt": dt, "unit_dt": dt == 1.0 and fps is None},
    )


def load_escape_csv(path: str) -> np.ndarray:
    """Load an escape-size csv (columns ``X_world, Y_world, R_world``) and
    return bubble **diameters** in mm (``d = 2 * R_world``)."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    r = np.asarray(data["R_world"], dtype=float)
    return 2.0 * r
