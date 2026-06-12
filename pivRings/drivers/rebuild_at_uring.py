#!/usr/bin/env python3
"""
rebuild_at_uring.py — rebuild a station's dimensionless field at a CHOSEN U_ring.

For the marginal stations the thin-ring U_ring overestimates the propagation
speed and the co-moving atmosphere looks open (REPORT §3, §7c).  The FTLE/
topological closure sweep gives a principled U_ring (the speed at which the ring
atmosphere closes).  This rebuilds the field at that closure speed and caches it
under fields/<label>_<suffix>/ so the downstream sims (run_uniform_size,
investigate_attractor) run on a trapping field.

  .venv/bin/python pivRings/drivers/rebuild_at_uring.py 200_10D 65 --suffix uc
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(ROOT))

import numpy as np

import build_field as bf
from constants import R0, froude_number

FIELDS = os.path.join(ROOT, "fields")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label")
    p.add_argument("u_ring", type=float, help="closure U_ring magnitude (mm/s)")
    p.add_argument("--suffix", default="uc", help="output dir suffix (fields/<label>_<suffix>)")
    args = p.parse_args()

    d = os.path.join(FIELDS, args.label)
    mp = os.path.join(d, "mean_reg.pkl")
    if not os.path.exists(mp):
        mp = os.path.join(d, "mean_field.pkl")
    mean = pickle.load(open(mp, "rb"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    sign = float(np.sign(meta.get("U_ring_signed", 1.0))) or 1.0

    field = bf.build_field(mean, U_ring=sign * args.u_ring, R0=R0)
    out = os.path.join(FIELDS, f"{args.label}_{args.suffix}")
    bf.save_field(field, out, mean=mean)
    Fr = froude_number(field.U_ring, R0_mm=R0)
    print(f"{args.label}: rebuilt at U_ring={field.U_ring:.1f} mm/s (was thin-ring "
          f"{meta['U_ring']:.1f})  Fr={Fr:.3f}  -> {out}")


if __name__ == "__main__":
    main()
