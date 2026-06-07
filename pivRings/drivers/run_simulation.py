#!/usr/bin/env python3
"""
run_simulation.py — seed -> advect -> record, for one cached station.

    load cached field -> fit_core_ellipse -> seed_positions + sample_stokes
                      -> advect (pool) -> record escape -> diagnostics

Example:
  python drivers/run_simulation.py --field fields/station_10D \
      --escape-csv data/escape/bubbles_120_l1.csv \
      --n-seed 24 --n-phi 16 --workers 4 --gravity
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(ROOT))

import numpy as np

import build_field as bf
import seeding as sd
import advect as ad
import diagnostics as dg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", required=True, help="cached field dir")
    p.add_argument("--escape-csv", required=True)
    p.add_argument("--measure", default="arclength",
                   choices=["arclength", "area", "annulus"])
    p.add_argument("--n-seed", type=int, default=24)
    p.add_argument("--n-phi", type=int, default=16)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--t-max", type=float, default=2.0)
    p.add_argument("--gravity", action="store_true")
    p.add_argument("--out", default=None, help="json results path")
    args = p.parse_args()

    field = bf.load_field(args.field)
    mean = bf.load_mean_field(args.field)
    if mean is None:
        raise SystemExit(
            f"No mean_field.pkl in {args.field}; re-run preprocess_station.py "
            "(it now caches the meridional mean needed for the ellipse fit).")

    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis)
    positions = sd.seed_positions(ell, n=args.n_seed, n_phi=args.n_phi,
                                  measure=args.measure)

    Xg, Rg = np.meshgrid(field.x_axis, field.r_axis)
    u_ring = float(np.nanmax(np.hypot(field.sp_Ux.ev(Rg.ravel(), Xg.ravel()),
                                      field.sp_Ur.ev(Rg.ravel(), Xg.ravel()))))
    stk = sd.sample_stokes(positions.shape[0], args.escape_csv, u_ring_mms=u_ring)

    results = ad.advect_bubbles(args.field, positions, stk.d, stokes=stk.St,
                                n_workers=args.workers, t_max=args.t_max,
                                gravity=args.gravity)
    stats = dg.escape_statistics(results)
    print(json.dumps(stats, indent=2))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
