#!/usr/bin/env python3
"""
run_ensembles.py — multi-seed ensemble simulations for statistics.

For every condition (station x loading) it runs, across several random seeds:
  * a diameter SWEEP (uniform in log d) -> escape probability vs size
    -> critical escape size d_crit (P_esc = 0.5);
  * a volume-conserved POPULATION (lognormal from the escape CSV) -> per-bubble
    escape times -> time development of remaining count / size / volume.

Saves per-bubble records to outputs/ensembles.npz and per-condition metadata to
outputs/ensembles_meta.json.  Plot with plot_ensembles.py.
"""

from __future__ import annotations

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
from constants import R0, froude_number, stokes_number

FIELDS = os.path.join(ROOT, "fields")
ESC = os.path.join(ROOT, "data", "escape")
OUT = os.path.join(ROOT, "outputs")

# (label, piston, downstream D)
STATIONS = [("120_5D", 120, 5), ("120_10D", 120, 10), ("120_15D", 120, 15),
            ("200_5D", 200, 5), ("200_10D", 200, 10), ("200_15D", 200, 15)]
LOADINGS = [("l1", 20.0), ("l3", 40.0)]

N_SEEDS = 3
D_GRID = np.logspace(np.log10(0.1), np.log10(2.5), 11)
SWEEP_NMERID, SWEEP_NPHI = 3, 8      # 24 positions / (seed, d)
POP_NMERID, POP_NPHI = 6, 12         # 72 bubbles / seed
WORKERS, T_MAX = 16, 20.0


def main():
    rec = {k: [] for k in ("kind", "piston", "D", "loading", "seed", "d", "St",
                           "escaped", "t_escape", "exit_face")}
    meta = {}

    for label, piston, nD in STATIONS:
        fdir = os.path.join(FIELDS, label)
        if not os.path.exists(os.path.join(fdir, "field.pkl")):
            print(f"  skip {label}: no cached field"); continue
        field = bf.load_field(fdir)
        mean = bf.load_mean_field(fdir)
        ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0)
        Fr = froude_number(field.U_ring, R0_mm=field.R0)
        D_mm = nD * 40.0

        for tag, vol in LOADINGS:
            csv = os.path.join(ESC, f"bubbles_{piston}_{tag}.csv")
            cond = f"{label}_{tag}"
            meta[cond] = {"piston": piston, "D": nD, "loading": vol,
                          "U_ring": field.U_ring, "Fr": Fr, "a_eq_mm": ell.a_eq}

            pos_list, d_list, St_list, kind_list, seed_list = [], [], [], [], []
            # --- diameter sweep ---
            for s in range(N_SEEDS):
                P = sd.seed_positions(ell, n=SWEEP_NMERID, n_phi=SWEEP_NPHI, seed=s)
                for d in D_GRID:
                    St = float(stokes_number(d, field.U_ring, field.R0))
                    for p in P:
                        pos_list.append(p); d_list.append(d); St_list.append(St)
                        kind_list.append(0); seed_list.append(s)
            # --- volume-conserved population ---
            for s in range(N_SEEDS):
                stk = sd.sample_stokes(POP_NMERID * POP_NPHI, csv, u_ring_mms=field.U_ring,
                                       R0_mm=field.R0, station_D=D_mm, seed=100 + s)
                P = sd.seed_positions(ell, n=POP_NMERID, n_phi=POP_NPHI, seed=200 + s)
                for i in range(len(stk.d)):
                    pos_list.append(P[i]); d_list.append(float(stk.d[i]))
                    St_list.append(float(stk.St[i])); kind_list.append(1); seed_list.append(s)

            positions = np.array(pos_list)
            res = ad.advect_bubbles(fdir, positions, np.array(d_list), np.array(St_list),
                                    Fr, n_workers=WORKERS, gravity=True, t_max=T_MAX)
            for i, r in enumerate(res):
                rec["kind"].append(kind_list[i]); rec["piston"].append(piston)
                rec["D"].append(nD); rec["loading"].append(vol); rec["seed"].append(seed_list[i])
                rec["d"].append(d_list[i]); rec["St"].append(St_list[i])
                rec["escaped"].append(bool(r.escaped))
                rec["t_escape"].append(r.t_escape)
                rec["exit_face"].append(r.exit_face)
            ne = sum(1 for r in res if r.escaped)
            print(f"  {cond}: {len(res)} bubbles, {ne} escaped  (U_ring={field.U_ring:.0f}, Fr={Fr:.3f})")

    np.savez(os.path.join(OUT, "ensembles.npz"),
             **{k: np.array(v) for k, v in rec.items()})
    with open(os.path.join(OUT, "ensembles_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n-> outputs/ensembles.npz ({len(rec['d'])} records), outputs/ensembles_meta.json")


if __name__ == "__main__":
    main()
