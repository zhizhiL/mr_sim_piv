#!/usr/bin/env python3
"""
scan_strength.py — TRUE total-strength scaling of one station (independent-run
variability, §7f).  Scaling the ring's circulation by s scales the lab field AND
U_ring by s, which leaves the dimensionless co-moving field (and the core
geometry) UNCHANGED and only lowers the U_ring that sets St/Fr (so W*=St/Fr^2
rises by 1/s).  Implemented by keeping the built dimensionless field and passing
St/Fr evaluated at the scaled U_ring.  Reports d_crit(s) to pick the factor that
puts the station on its series' trend.

  .venv/bin/python pivRings/drivers/scan_strength.py 200_15D_sf 1.0,0.7,0.5,0.35,0.25,0.15
"""
from __future__ import annotations
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(ROOT))
import numpy as np
import build_field as bf, seeding as sd, advect as ad
from constants import R0, froude_number, stokes_number


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label"); p.add_argument("scales")
    p.add_argument("--diam", default="0.24,0.36,0.5,0.7,0.9,1.1,1.3")
    p.add_argument("--n-sim", type=int, default=48)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    fdir = os.path.join(ROOT, "fields", args.label)
    field = bf.load_field(fdir); mean = bf.load_mean_field(fdir)
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0)
    U0 = field.U_ring
    diam = np.array([float(x) for x in args.diam.split(",")])
    scales = [float(x) for x in args.scales.split(",")]
    print(f"{args.label}: U0={U0:.1f} mm/s  a_eq={ell.a_eq:.1f} mm (fixed under strength scaling)")
    print(f"{'s':>5} {'U_eff':>6} {'Fr':>6} {'d_crit':>7}   capture-by-diameter")
    rng = np.random.default_rng(0)
    pos, dia = [], []
    for d in diam:
        pos.append(sd.seed_core_surface(ell, args.n_sim, rng)); dia.append(np.full(args.n_sim, d))
    pos = np.vstack(pos); dia = np.concatenate(dia)
    for s in scales:
        U = s * U0; Fr = froude_number(U)
        St = stokes_number(dia, U, R0)
        res = ad.advect_bubbles(fdir, pos, dia, St, Fr, n_workers=args.workers, gravity=True, t_max=20.0)
        esc = np.array([r.escaped for r in res])
        cap = np.array([1 - esc[dia == d].mean() for d in diam])
        dc = np.nan
        if cap[0] >= 0.5:
            b = np.where(cap < 0.5)[0]
            if len(b):
                i = b[0]; dc = diam[i-1] + (0.5-cap[i-1])*(diam[i]-diam[i-1])/(cap[i]-cap[i-1]-1e-12)
            else:
                dc = diam[-1]
        print(f"{s:5.2f} {U:6.1f} {Fr:6.3f} {dc:7.2f}   " + " ".join(f"{100*c:3.0f}" for c in cap), flush=True)


if __name__ == "__main__":
    main()
