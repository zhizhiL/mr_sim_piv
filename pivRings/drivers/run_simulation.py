#!/usr/bin/env python3
"""
run_simulation.py — seed -> advect -> record, for one cached station.

    load cached field -> fit_core_ellipse -> seed_positions + sample_stokes
                      -> advect (pool) -> escape statistics + plots

Example:
  .venv/bin/python pivRings/drivers/run_simulation.py \
      --field pivRings/fields/station_10D \
      --escape-csv pivRings/data/escape/bubbles_200_l1.csv \
      --n-seed 24 --n-phi 16 --workers 8 --gravity
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
from constants import R0, froude_number


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", required=True)
    p.add_argument("--escape-csv", required=True)
    p.add_argument("--measure", default="arclength",
                   choices=["arclength", "area", "annulus"])
    p.add_argument("--n-seed", type=int, default=24)
    p.add_argument("--n-phi", type=int, default=16)
    p.add_argument("--station-D", type=float, default=None,
                   help="station downstream Y_world (mm), e.g. 400 for 10D; "
                        "enables volume-conserved sizing")
    p.add_argument("--loading-uL", type=float, default=None,
                   help="initial bubble loading; default inferred l1=20/l3=40")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--t-max", type=float, default=20.0)
    p.add_argument("--gravity", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    field = bf.load_field(args.field)
    mean = bf.load_mean_field(args.field)
    if mean is None:
        raise SystemExit(f"No mean_field.pkl in {args.field}; re-run preprocess_station.py")

    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0)
    positions = sd.seed_positions(ell, n=args.n_seed, n_phi=args.n_phi, measure=args.measure)
    stk = sd.sample_stokes(positions.shape[0], args.escape_csv,
                           u_ring_mms=field.U_ring, R0_mm=field.R0,
                           station_D=args.station_D, initial_loading_uL=args.loading_uL)
    Fr = froude_number(field.U_ring, R0_mm=field.R0)

    results = ad.advect_bubbles(args.field, positions, stk.d, stk.St, Fr,
                                n_workers=args.workers, gravity=args.gravity,
                                t_max=args.t_max)
    stats = dg.escape_statistics(results)
    stats.update(U_ring=field.U_ring, Fr=Fr, a_eq_mm=ell.a_eq,
                 St_min=float(stk.St.min()), St_max=float(stk.St.max()),
                 d_max_mm=stk.d_max, V_initial_uL=stk.V_initial,
                 V_escaped_upstream_uL=stk.V_escaped_upstream,
                 V_remaining_uL=stk.V_remaining, n_physical=stk.n_physical)
    print(json.dumps(stats, indent=2))

    base = args.out or os.path.join(ROOT, "outputs", os.path.basename(args.field))
    os.makedirs(os.path.dirname(base) if os.path.dirname(base) else ".", exist_ok=True)
    dg.plot_escape_distribution(results, args.escape_csv, base + "_escape_sizes.png")
    with open(base + "_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"-> {base}_escape_sizes.png , {base}_stats.json")


if __name__ == "__main__":
    main()
