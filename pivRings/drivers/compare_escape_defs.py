#!/usr/bin/env python3
"""
compare_escape_defs.py — d_crit under FOV vs separatrix escape, all stations.

The FOV-rectangle escape test is FOV- and t_max-dependent (REPORT §7d); the
co-moving streamfunction separatrix is tied to the vortex structure and so
FOV-independent (REPORT §7e).  For each adopted field this seeds the core
surface across a diameter grid, advects every bubble ONCE per definition, and
reports the capture-by-diameter curve + the 50%-capture diameter d_crit for both
— quantifying how much the FOV truncation moved the threshold.

  .venv/bin/python pivRings/drivers/compare_escape_defs.py
  .venv/bin/python pivRings/drivers/compare_escape_defs.py --fields 120_5D 200_15D_scaled
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_field as bf
import seeding as sd
import separatrix as sx
import advect as ad
from constants import froude_number, stokes_number

FIELDS = os.path.join(ROOT, "fields")
ADOPTED = ["120_5D", "120_10D", "120_15D_geom", "200_5D_sf", "200_10D_sf", "200_15D_scaled"]


def d_crit_from_capture(diam, cap):
    """50%-capture diameter by linear interpolation of the capture curve."""
    cap = np.asarray(cap)
    if cap[0] < 0.5:
        return np.nan
    below = np.where(cap < 0.5)[0]
    if not len(below):
        return float(diam[-1])
    i = below[0]
    x0, x1, y0, y1 = diam[i - 1], diam[i], cap[i - 1], cap[i]
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0 - 1e-12))


def run_field(label, core, diam, n_sim, workers, t_max):
    fdir = os.path.join(FIELDS, label)
    field = bf.load_field(fdir)
    mean = bf.load_mean_field(fdir)
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core=core)
    Fr = froude_number(field.U_ring, R0_mm=field.R0)
    sep = sx.load_or_compute(fdir, field, core_xr=(ell.x_c / field.R0, ell.r_c / field.R0))
    if sep is None:
        print(f"{label}: no separatrix found — skipping"); return None

    rng = np.random.default_rng(0)
    pos, dia = [], []
    for dd in diam:
        pos.append(sd.seed_core_surface(ell, n_sim, rng)); dia.append(np.full(n_sim, dd))
    pos = np.vstack(pos); dia = np.concatenate(dia)
    St = stokes_number(dia, field.U_ring, field.R0)

    res_fov = ad.advect_bubbles(fdir, pos, dia, St, Fr, n_workers=workers,
                                gravity=True, t_max=t_max, escape="fov")
    res_sep = ad.advect_bubbles(fdir, pos, dia, St, Fr, n_workers=workers,
                                gravity=True, t_max=t_max, escape="separatrix",
                                sep_poly=sep.poly)
    esc_fov = np.array([r.escaped for r in res_fov])
    esc_sep = np.array([r.escaped for r in res_sep])
    cap_fov = np.array([1 - esc_fov[dia == dd].mean() for dd in diam])
    cap_sep = np.array([1 - esc_sep[dia == dd].mean() for dd in diam])
    dc_fov = d_crit_from_capture(diam, cap_fov)
    dc_sep = d_crit_from_capture(diam, cap_sep)
    print(f"{label:16s} U={field.U_ring:6.1f}  d_crit: FOV={dc_fov:.2f}  "
          f"SEP={dc_sep:.2f}  (atmosphere area={sep.area():.2f}, "
          f"FOV-truncated={sep.fov_truncated})", flush=True)
    return dict(label=label, U_ring=field.U_ring, Fr=Fr, diam=diam.tolist(),
                cap_fov=cap_fov.tolist(), cap_sep=cap_sep.tolist(),
                d_crit_fov=dc_fov, d_crit_sep=dc_sep, sep_area=sep.area(),
                fov_truncated=bool(sep.fov_truncated))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", nargs="*", default=ADOPTED)
    p.add_argument("--core", default="upper")
    p.add_argument("--diam", default="0.12,0.24,0.36,0.5,0.7,0.9,1.1,1.3,1.6")
    p.add_argument("--n-sim", type=int, default=48)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--t-max", type=float, default=20.0)
    args = p.parse_args()
    diam = np.array([float(x) for x in args.diam.split(",")])

    print(f"escape-definition comparison (core={args.core}, n_sim={args.n_sim}, "
          f"t_max={args.t_max})\n  diameter grid {diam}")
    results = [r for r in (run_field(lbl, args.core, diam, args.n_sim, args.workers,
                                     args.t_max) for lbl in args.fields) if r]

    out_json = os.path.join(ROOT, "outputs", "escape_def_comparison.json")
    json.dump(results, open(out_json, "w"), indent=2)
    print(f"-> {out_json}")

    n = len(results)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow), squeeze=False)
    for k, r in enumerate(results):
        ax = axes[k // ncol][k % ncol]
        ax.plot(r["diam"], 100 * np.array(r["cap_fov"]), "o-", color="#1f4e79",
                label=f"FOV  d_crit={r['d_crit_fov']:.2f}")
        ax.plot(r["diam"], 100 * np.array(r["cap_sep"]), "s--", color="crimson",
                label=f"separatrix  d_crit={r['d_crit_sep']:.2f}")
        ax.axhline(50, color="0.7", lw=0.8)
        ax.set_title(f"{r['label']}  (U={r['U_ring']:.0f})"
                     + ("  [atm>FOV]" if r["fov_truncated"] else ""), fontsize=9)
        ax.set_xlabel("diameter (mm)"); ax.set_ylabel("% captured")
        ax.set_ylim(-3, 103); ax.legend(fontsize=7, loc="upper right")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Capture threshold: FOV-rectangle vs streamfunction-separatrix escape")
    fig.tight_layout()
    out = os.path.join(ROOT, "outputs", "escape_def_comparison.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
