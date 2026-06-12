#!/usr/bin/env python3
"""
diag_15D_core.py — diagnose the 15D core faults (120_15D R_ring too small,
200_15D a_eq too large): compare the TIME-AVERAGE core geometry against a
representative SINGLE FRAME, to test whether the fault is an averaging smear of
the meandering far-downstream vortex.  Also evaluates an alternative dataset.

Usage: diag_15D_core.py <folder> <coord> [--window a b]
"""
from __future__ import annotations
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(ROOT))
import numpy as np
import digiflow_io as dio, frame_transform as ft, averaging as avg, seeding as sd
from averaging import MeanField
from build_field import locate_axis

BASE = "/mnt/d/Users/zl483/highspeedcamera"


def geom(mean):
    """Return (R_ring_mm, a_eq_mm) from the thin-ring fit + core ellipse."""
    try:
        _, ring = ft.estimate_uring_thinring(mean, sign=1.0)
        R = float(ring["R"])
    except Exception:
        R = float("nan")
    try:
        ell = sd.fit_core_ellipse(mean, y_axis=locate_axis(mean), R0=20.0)
        a = float(ell.a_eq)
    except Exception:
        a = float("nan")
    return R, a


def _stat(v):
    v = [x for x in v if np.isfinite(x)]
    if not v:
        return "n/a"
    return f"median={np.median(v):5.1f} (IQR {np.percentile(v,25):.1f}-{np.percentile(v,75):.1f}) mm"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder"); p.add_argument("coord")
    p.add_argument("--window", type=int, nargs=2, default=None)
    args = p.parse_args()
    sdir = os.path.join(BASE, args.folder, "Camera_1")
    cf = os.path.join(BASE, f"{args.coord}_mapping.csv")
    if args.window:
        start, stop = args.window
    else:
        start, stop = dio.autodetect_window(sdir, cf, stride=4)
    frames = dio.load_piv(sdir, cf, fps=500, start=start, stop=stop)
    uc = ft.estimate_Uc(frames, method="vorticity_centroid")
    frames = ft.register_frames(frames, uc.x_track)
    print(f"{args.folder} ({args.coord}) window=[{start},{stop}] n={frames.nframes}", flush=True)

    Ravg, aavg = geom(avg.smooth_field(avg.time_average(frames), sigma=1.5))
    print(f"  time-average : R_ring={Ravg:6.1f} mm   a_eq={aavg:6.1f} mm", flush=True)

    Rs, as_ = [], []
    for k in np.linspace(0, frames.nframes - 1, 15).astype(int):
        m = avg.smooth_field(MeanField(X=frames.X, Y=frames.Y, Ux=frames.u[k],
                                       Uy=frames.v[k], omega=frames.omega[k]), sigma=1.5)
        R, a = geom(m)
        Rs.append(R); as_.append(a)
    print(f"  single-frame : R_ring {_stat(Rs)}", flush=True)
    print(f"  single-frame : a_eq   {_stat(as_)}", flush=True)


if __name__ == "__main__":
    main()
