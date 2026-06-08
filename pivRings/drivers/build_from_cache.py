#!/usr/bin/env python3
"""
build_from_cache.py — build dimensionless fields from cached registered means.

Reads fields/<label>/mean_reg.pkl + station.json (from cache_means.py), sets
U_ring = the signed lab-frame axial velocity at the vortex core (Ux@core ~ ring
propagation speed), builds the field (sign-aware), validates that the core
traps tracers, and saves the field cache.  Fast: no .dfi reloads.
"""

from __future__ import annotations

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
import seeding as sd
import advect as ad
import frame_transform as ft
from constants import R0, froude_number

FIELDS = os.path.join(ROOT, "fields")
LABELS = ["120_5D", "120_10D", "120_15D", "200_5D", "200_10D", "200_15D"]


def core_trapped(field, mean, n=8, t_max=12.0):
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0)
    xc, rc = ell.x_c / field.R0, ell.r_c / field.R0
    Fr = froude_number(field.U_ring, R0_mm=field.R0)
    c = 0
    for phi in np.linspace(0, 2 * np.pi, n, endpoint=False):
        p = np.array([xc, rc * np.cos(phi), rc * np.sin(phi)])
        r = ad.advect_one(field, p, 0.3, St=1e-3, Fr=Fr, gravity=False, t_max=t_max)
        c += (not r.escaped)
    return c / n, ell.a_eq


def main():
    summary = {}
    for label in LABELS:
        d = os.path.join(FIELDS, label)
        mp = os.path.join(d, "mean_reg.pkl")
        sj = os.path.join(d, "station.json")
        if not (os.path.exists(mp) and os.path.exists(sj)):
            print(f"{label}: no cached mean; run cache_means.py"); continue
        mean = pickle.load(open(mp, "rb"))
        info = json.load(open(sj))
        # physically-principled propagation speed (thin-ring), signed by the
        # tracked centroid direction
        U_signed, ring = ft.estimate_uring_thinring(
            mean, sign=info["centroid_signed"])
        field = bf.build_field(mean, U_ring=U_signed, R0=R0)
        bf.save_field(field, d, mean=mean)
        tf, a_eq = core_trapped(field, mean)
        Fr = froude_number(field.U_ring, R0_mm=R0)
        summary[label] = {"U_ring": field.U_ring, "U_ring_signed": U_signed,
                          "centroid": info["centroid_signed"], "Fr": Fr,
                          "Gamma": ring["Gamma"], "R_ring_mm": ring["R"],
                          "a_eq_mm": a_eq, "core_trapped": tf,
                          "piston": info["piston"], "D": info["D"]}
        print(f"{label:9s} U_ring={field.U_ring:6.1f} (thin-ring; centroid "
              f"{info['centroid_signed']:+.1f})  Fr={Fr:.3f}  Gamma={ring['Gamma']:.0f}  "
              f"R={ring['R']:.1f}  a_eq={a_eq:5.1f}  core_trapped={tf:.2f}", flush=True)
    with open(os.path.join(ROOT, "outputs", "field_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("-> outputs/field_summary.json")


if __name__ == "__main__":
    main()
