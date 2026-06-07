#!/usr/bin/env python3
"""
preprocess_station.py — raw PIV -> cached dimensionless 3D field, per station.

    load_piv -> estimate_Uc -> time_average -> smooth
             -> nondimensionalize + gradients + revolve(3D) -> cache

Reads the .dfi frames IN PLACE from /mnt/d (never copied into the repo); only
the small processed field cache is written under fields/<station>/.

Real data (build_plan.md §2A), e.g. the 10D / Up=200 station:

  .venv/bin/python pivRings/drivers/preprocess_station.py \
      --station-dir /mnt/d/Users/zl483/highspeedcamera/bonus_test_11/Camera_1 \
      --coord-file  /mnt/d/Users/zl483/highspeedcamera/april_bonus_mapping.csv \
      --fps 200 --station station_10D

Use --synthetic to run the same path on a generated field.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(ROOT))

import numpy as np

import frame_transform as ft
import averaging as avg
import build_field as bf
import nondimensional as nd
import diagnostics as dg
from constants import R0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station-dir")
    p.add_argument("--coord-file")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--station", default="station_1")
    p.add_argument("--out", default=None)
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--uc-method", default="vorticity_centroid",
                   choices=["vorticity_centroid", "core_linear_fit"])
    p.add_argument("--u-ring", type=float, default=None,
                   help="override U_ring (mm/s), e.g. interpolate a bad station "
                        "from neighbours; used for co-moving + St/Fr scaling")
    p.add_argument("--y-axis", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start", type=int, default=0, help="first frame of the window")
    p.add_argument("--stop", type=int, default=None, help="last frame (exclusive)")
    p.add_argument("--autodetect", action="store_true",
                   help="auto-pick the in-FOV frame window")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    out = args.out or os.path.join(ROOT, "fields", args.station)

    if args.synthetic:
        import synthetic
        frames = synthetic.make_ring_frames(station=args.station)
    else:
        if not (args.station_dir and args.coord_file):
            p.error("--station-dir and --coord-file required without --synthetic")
        import digiflow_io as dio
        start, stop = args.start, args.stop
        if args.autodetect:
            start, stop = dio.autodetect_window(args.station_dir, args.coord_file,
                                                verbose=True)
        frames = dio.load_piv(args.station_dir, args.coord_file,
                              fps=args.fps, dt=args.dt, max_frames=args.max_frames,
                              start=start, stop=stop)

    uc = ft.estimate_Uc(frames, method=args.uc_method)
    U_ring = args.u_ring if args.u_ring is not None else uc.U_c
    qc = ft.residual_unsteadiness(ft.to_comoving(frames, U_ring))
    mean = avg.smooth_field(avg.time_average(frames), sigma=args.smooth_sigma)
    field = bf.build_field(mean, U_ring=U_ring, R0=R0, y_axis=args.y_axis)
    bf.save_field(field, out, mean=mean)

    Xg, Rg = np.meshgrid(field.x_axis, field.r_axis)
    o1 = nd.assert_field_O1(field.sp_Ux.ev(Xg.ravel(), Rg.ravel()))
    dg.plot_field_sanity(field, os.path.join(out, "streamlines.png"))

    note = "" if args.u_ring is None else f" [overridden; measured {uc.U_c:.2f}]"
    print(f"[{args.station}] frames={frames.nframes}  U_ring={U_ring:.2f} mm/s{note} "
          f"({uc.method}, residual {uc.residual_rms:.3f} mm)")
    print(f"[{args.station}] residual unsteadiness={qc['relative_unsteadiness']:.3f}  "
          f"|Ux*| p99={o1:.2f}")
    print(f"[{args.station}] axis y={field.y_axis_mm:.2f} mm  cached -> {out}")


if __name__ == "__main__":
    main()
