#!/usr/bin/env python3
"""
test_closure_remedies.py — for the stations whose TIME-AVERAGED two-core field
has no closed co-moving streamline at the thin-ring U_ring, test whether a
closed atmosphere can be recovered from a single CORE (upper/lower alone) or a
single FRAME, as a representative of the noisy set (§7e remedy).

Closure proxy: fraction of core-seeded fluid tracers still inside after T*=15
(>~30% => a closed recirculation / extractable separatrix exists).
"""
from __future__ import annotations
import json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(ROOT))
import numpy as np
import build_field as bf, seeding as sd, ftle as F
from constants import R0
from averaging import MeanField

# label -> thin-ring U_ring (mm/s, magnitude); these three are open at thin-ring
CASES = {"200_5D": 109.7, "200_10D": 83.8, "120_15D": 55.0}


def trapped_fraction(field, mean, core, n=300, T=15.0):
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0,
                              core=("lower" if core == "lower" else "upper"))
    seeds = sd.seed_core_surface(ell, n, np.random.default_rng(0))
    Pf, alive = F.fluid_flow_map(field, np.column_stack([seeds[:, 0], seeds[:, 2]]),
                                 T=T, direction=1)
    return float(alive.mean())


def main():
    print(f"{'station':9} {'U_ring':>7}  {'both':>6} {'upper':>6} {'lower':>6}   (trapped-tracer fraction)")
    for label, U in CASES.items():
        d = os.path.join(ROOT, "fields", label)
        mp = os.path.join(d, "mean_reg.pkl"); mp = mp if os.path.exists(mp) else os.path.join(d, "mean_field.pkl")
        mean = pickle.load(open(mp, "rb"))
        sign = float(np.sign(json.load(open(os.path.join(d, "meta.json"))).get("U_ring_signed", 1.0))) or 1.0
        out = {}
        for core in ("both", "upper", "lower"):
            field = bf.build_field(mean, U_ring=sign * U, R0=R0, core=core)
            out[core] = trapped_fraction(field, mean, core)
        print(f"{label:9} {U:7.1f}  {out['both']:6.0%} {out['upper']:6.0%} {out['lower']:6.0%}", flush=True)


if __name__ == "__main__":
    main()
