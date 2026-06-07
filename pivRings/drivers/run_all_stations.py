#!/usr/bin/env python3
"""
run_all_stations.py — preprocess + simulate all six PIV movies.

For each (piston speed, downstream station) movie from build_plan.md §2A:
    auto-detect the in-FOV window -> cache the dimensionless field
    -> for each loading (l1=20uL, l3=40uL): volume-conserved seeding -> advect
       -> escape statistics.

Writes a summary table (outputs/all_stations_summary.json) and an
escape-fraction-vs-downstream plot per piston speed.  Reads .dfi in place from
/mnt/d; only the small field caches + plots are written under pivRings/.

Run:  .venv/bin/python pivRings/drivers/run_all_stations.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import digiflow_io as dio
import frame_transform as ft
import averaging as avg
import build_field as bf
import seeding as sd
import advect as ad
import diagnostics as dg
from constants import R0, froude_number

BASE = "/mnt/d/Users/zl483/highspeedcamera"
FIELDS = os.path.join(ROOT, "fields")
OUT = os.path.join(ROOT, "outputs")
ESC = os.path.join(ROOT, "data", "escape")
os.makedirs(OUT, exist_ok=True)

# (label, folder, coord_name, piston speed, downstream D index)
STATIONS = [
    ("120_5D",  "bonus_test_17", "coord_up",        120,  5),
    ("120_10D", "bonus_test_01", "april_vorticity", 120, 10),
    ("120_15D", "bonus_test_27", "coord_down",      120, 15),
    ("200_5D",  "bonus_test_15", "coord_up",        200,  5),
    ("200_10D", "bonus_test_11", "april_bonus",     200, 10),
    ("200_15D", "bonus_test_25", "coord_down",      200, 15),
]
LOADINGS = [("l1", 20.0), ("l3", 40.0)]
N_SEED, N_PHI, WORKERS, FPS = 16, 8, 8, 500.0


def preprocess(label, folder, coord):
    station_dir = os.path.join(BASE, folder, "Camera_1")
    coord_file = os.path.join(BASE, f"{coord}_mapping.csv")
    start, stop = dio.autodetect_window(station_dir, coord_file, stride=4, verbose=True)
    frames = dio.load_piv(station_dir, coord_file, fps=FPS, start=start, stop=stop)
    uc = ft.estimate_Uc(frames, method="vorticity_centroid")
    mean = avg.smooth_field(avg.time_average(frames), sigma=1.5)
    field = bf.build_field(mean, U_ring=uc.U_c, R0=R0)
    fdir = os.path.join(FIELDS, label)
    bf.save_field(field, fdir, mean=mean)
    return fdir, field, mean, uc, (start, stop)


def main():
    summary = {}
    for label, folder, coord, piston, nD in STATIONS:
        print("\n" + "=" * 70 + f"\n{label}  ({folder}, {coord}, Up={piston} mm/s, {nD}D)\n" + "=" * 70)
        try:
            fdir, field, mean, uc, win = preprocess(label, folder, coord)
        except Exception as e:
            print(f"  PREPROCESS FAILED: {e}")
            summary[label] = {"error": str(e)}
            continue
        Fr = froude_number(field.U_ring, R0_mm=R0)
        ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=R0)
        positions = sd.seed_positions(ell, n=N_SEED, n_phi=N_PHI)
        print(f"  window {win}  U_ring={field.U_ring:.1f} mm/s  Fr={Fr:.3f}  "
              f"a_eq={ell.a_eq:.1f} mm  axis_y={field.y_axis_mm:.1f} mm")

        rec = {"U_ring": field.U_ring, "Fr": Fr, "a_eq_mm": ell.a_eq,
               "window": list(win), "downstream_mm": nD * 40.0, "loadings": {}}
        for tag, vol in LOADINGS:
            csv = os.path.join(ESC, f"bubbles_{piston}_{tag}.csv")
            try:
                stk = sd.sample_stokes(positions.shape[0], csv, u_ring_mms=field.U_ring,
                                       R0_mm=R0, station_D=nD * 40.0, initial_loading_uL=vol)
                res = ad.advect_bubbles(fdir, positions, stk.d, stk.St, Fr,
                                        n_workers=WORKERS, gravity=True, t_max=20.0)
                st = dg.escape_statistics(res)
                rec["loadings"][tag] = {
                    "loading_uL": vol, "escape_fraction": st["escape_fraction"],
                    "d_max_mm": stk.d_max, "V_remaining_uL": stk.V_remaining,
                    "V_escaped_upstream_uL": stk.V_escaped_upstream,
                    "n_physical": stk.n_physical,
                    "St_min": float(stk.St.min()), "St_max": float(stk.St.max())}
                print(f"  {tag} ({vol:.0f}uL): escape={st['escape_fraction']:.2f}  "
                      f"d_max={stk.d_max:.2f}mm  V_rem={stk.V_remaining:.1f}uL  "
                      f"St=[{stk.St.min():.3f},{stk.St.max():.3f}]")
            except Exception as e:
                rec["loadings"][tag] = {"error": str(e)}
                print(f"  {tag}: FAILED {e}")
        summary[label] = rec

    with open(os.path.join(OUT, "all_stations_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # escape-fraction vs downstream, per piston speed / loading
    fig, ax = plt.subplots(figsize=(7, 5))
    for piston, style in [(120, "o-"), (200, "s--")]:
        for tag, _ in LOADINGS:
            Ds, fr = [], []
            for label, _, _, p, nD in STATIONS:
                if p != piston:
                    continue
                r = summary.get(label, {}).get("loadings", {}).get(tag, {})
                if "escape_fraction" in r:
                    Ds.append(nD); fr.append(r["escape_fraction"])
            if Ds:
                ax.plot(Ds, fr, style, label=f"Up={piston}, {tag}")
    ax.set_xlabel("downstream station (D, ×40 mm)")
    ax.set_ylabel("escape fraction")
    ax.set_title("Escape fraction vs station")
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "all_stations_escape.png"), dpi=130)

    print(f"\n-> outputs/all_stations_summary.json , outputs/all_stations_escape.png")


if __name__ == "__main__":
    main()
