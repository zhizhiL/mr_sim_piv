#!/usr/bin/env python3
"""
scan_dcrit_uring.py — capture threshold d_crit as a function of U_ring.

For a marginal station the chosen U_ring (within the closed band, §7c/§7d)
controls how robust the trapping atmosphere is, and hence d_crit.  Too high
(near closure) → marginal atmosphere, d_crit collapses; too low → buoyancy
W*∝1/U dominates, d_crit shrinks.  This scans U_ring, rebuilds the field, seeds
the core surface across a diameter grid, advects, and reports the 50%-capture
diameter d_crit(U) (upper core) so a physically-reasonable U_ring can be chosen.

  .venv/bin/python pivRings/drivers/scan_dcrit_uring.py 200_5D 50,60,70,80,85
"""
from __future__ import annotations

import argparse, json, os, pickle, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(ROOT))

import numpy as np
import build_field as bf, seeding as sd, advect as ad
from constants import R0, froude_number, stokes_number

FIELDS = os.path.join(ROOT, "fields")


def dcrit_at_U(label, U, core, diam, n_sim, workers, t_max):
    d = os.path.join(FIELDS, label)
    mp = os.path.join(d, "mean_reg.pkl")
    mp = mp if os.path.exists(mp) else os.path.join(d, "mean_field.pkl")
    mean = pickle.load(open(mp, "rb"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    sign = float(np.sign(meta.get("U_ring_signed", 1.0))) or 1.0
    field = bf.build_field(mean, U_ring=sign * U, R0=R0)
    tmp = tempfile.mkdtemp(prefix=f"{label}_scan_")
    try:
        bf.save_field(field, tmp, mean=mean)
        ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core=core)
        Fr = froude_number(field.U_ring, R0_mm=field.R0)
        rng = np.random.default_rng(0)
        pos, dia = [], []
        for dd in diam:
            P = sd.seed_core_surface(ell, n_sim, rng); pos.append(P); dia.append(np.full(n_sim, dd))
        pos = np.vstack(pos); dia = np.concatenate(dia)
        St = stokes_number(dia, field.U_ring, field.R0)
        res = ad.advect_bubbles(tmp, pos, dia, St, Fr, n_workers=workers, gravity=True, t_max=t_max)
        esc = np.array([r.escaped for r in res])
        cap = np.array([1 - esc[(dia == dd)].mean() for dd in diam])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 50% crossing
    dc = np.nan
    if cap[0] >= 0.5:
        below = np.where(cap < 0.5)[0]
        if len(below):
            i = below[0]; x0, x1, y0, y1 = diam[i-1], diam[i], cap[i-1], cap[i]
            dc = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0 - 1e-12)
        else:
            dc = diam[-1]
    return field.U_ring, Fr, cap, dc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label")
    p.add_argument("uvals", help="comma list of U_ring magnitudes (mm/s)")
    p.add_argument("--core", default="upper")
    p.add_argument("--diam", default="0.12,0.24,0.36,0.5,0.7,0.9,1.1,1.3,1.6")
    p.add_argument("--n-sim", type=int, default=48)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--t-max", type=float, default=20.0)
    args = p.parse_args()
    Us = [float(x) for x in args.uvals.split(",")]
    diam = np.array([float(x) for x in args.diam.split(",")])
    print(f"{args.label} [{args.core}]  diam grid {diam}")
    print(f"{'U_ring':>7} {'Fr':>6} {'d_crit':>7}   capture-by-diameter")
    for U in Us:
        Um, Fr, cap, dc = dcrit_at_U(args.label, U, args.core, diam, args.n_sim, args.workers, args.t_max)
        capstr = " ".join(f"{100*c:3.0f}" for c in cap)
        print(f"{Um:7.1f} {Fr:6.3f} {dc:7.2f}   {capstr}", flush=True)


if __name__ == "__main__":
    main()
