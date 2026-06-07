#!/usr/bin/env python3
"""
preprocess_station.py — raw PIV -> cached 3D field, per station.

    load_piv -> estimate_Uc -> to_comoving -> time_average -> smooth
             -> gradients -> revolve(3D) -> cache interpolators

For the real experimental data, point ``--station-dir`` / ``--coord-file`` at
the DigiFlow folders documented in build_plan.md §2 Stage A, e.g.

  python drivers/preprocess_station.py \
      --station-dir /mnt/d/Users/zl483/highspeedcamera/bonus_test_11/Camera_1 \
      --coord-file  /mnt/d/Users/zl483/highspeedcamera/april_bonus_mapping.csv \
      --fps 200 --station station_10D --out fields/station_10D

Use ``--synthetic`` to run against a generated field instead of real data.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(ROOT))   # repo root: digiflowio, vortex_core

import frame_transform as ft
import averaging as avg
import build_field as bf
import diagnostics as dg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station-dir", help="folder of .dfi frames")
    p.add_argument("--coord-file", help="pixel->world mapping csv")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--station", default="station_1")
    p.add_argument("--out", default=None, help="cache dir (default fields/<station>)")
    p.add_argument("--smooth-sigma", type=float, default=1.5)
    p.add_argument("--y-axis", type=float, default=None, help="force ring axis (mm)")
    p.add_argument("--synthetic", action="store_true",
                   help="use a synthetic field instead of reading .dfi")
    args = p.parse_args()

    out = args.out or os.path.join(ROOT, "fields", args.station)

    if args.synthetic:
        import synthetic
        frames = synthetic.make_ring_frames(station=args.station)
    else:
        if not (args.station_dir and args.coord_file):
            p.error("--station-dir and --coord-file are required without --synthetic")
        import digiflow_io as dio
        frames = dio.load_piv(args.station_dir, args.coord_file,
                              fps=args.fps, dt=args.dt)

    uc = ft.estimate_Uc(frames)
    co = ft.to_comoving(frames, uc.U_c)
    qc = ft.residual_unsteadiness(co)
    mean = avg.smooth_field(avg.time_average(co), sigma=args.smooth_sigma)
    field = bf.build_field(mean, y_axis=args.y_axis)
    bf.save_field(field, out, mean=mean)

    dg.plot_field_sanity(field, os.path.join(out, "streamlines.png"))

    print(f"[{args.station}] U_c = {uc.U_c:.2f} mm/s "
          f"(residual {uc.residual_rms:.3f} mm)")
    print(f"[{args.station}] residual unsteadiness = {qc['relative_unsteadiness']:.3f}")
    print(f"[{args.station}] cached field -> {out}")


if __name__ == "__main__":
    main()
