#!/usr/bin/env python3
"""
run_uniform.py — seed the IDENTICAL bubble population at every station.

Unlike run_ensembles (per-station volume-conserved seeding, which biases the
cross-station comparison because downstream stations get smaller bubbles), this
seeds one common size distribution at all stations so differences reflect the
FIELD only.  Used for a fair retention / time-development comparison.

Saves outputs/uniform.npz and writes figC_uniform_volume_vs_time.png +
figE_retention_vs_station.png.
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

import build_field as bf
import seeding as sd
import advect as ad
from digiflow_io import load_escape_csv
from constants import R0, froude_number, stokes_number

FIELDS = os.path.join(ROOT, "fields")
ESC = os.path.join(ROOT, "data", "escape")
OUT = os.path.join(ROOT, "outputs")

STATIONS = [("120_5D", 120, 5), ("120_10D", 120, 10), ("120_15D", 120, 15),
            ("200_5D", 200, 5), ("200_10D", 200, 10), ("200_15D", 200, 15)]
PCOL = {120: "#1f77b4", 200: "#d62728"}
LS = {5: "-", 10: "--", 15: ":"}
N_MERID, N_PHI, N_SEEDS, WORKERS, T_MAX = 18, 10, 3, 16, 20.0


def common_population(n):
    """One fixed size distribution for all stations: log-normal fit to the
    pooled escape data, truncated to [0.1, 1.5] mm, drawn with a fixed seed."""
    pooled = np.concatenate([load_escape_csv(os.path.join(ESC, f))
                             for f in os.listdir(ESC) if f.endswith(".csv")])
    mu, sigma = float(np.log(pooled).mean()), float(np.log(pooled).std())
    rng = np.random.default_rng(12345)
    out = []
    while len(out) < n:
        x = rng.lognormal(mu, sigma, 4 * n)
        out.extend(x[(x >= 0.1) & (x <= 1.5)].tolist())
    return np.array(out[:n]), mu, sigma


def main():
    n = N_MERID * N_PHI
    d_common, mu, sigma = common_population(n)
    print(f"common population: {n} bubbles, log-normal(mu={mu:.2f}, sigma={sigma:.2f}) "
          f"in [0.1,1.5] mm, V0={np.pi/6*np.sum(d_common**3):.2f} uL")

    rec = {k: [] for k in ("piston", "D", "seed", "d", "St", "escaped", "t_escape", "exit_face")}
    meta = {}
    for label, piston, nD in STATIONS:
        fdir = os.path.join(FIELDS, label)
        field = bf.load_field(fdir)
        mean = bf.load_mean_field(fdir)
        ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0)
        Fr = froude_number(field.U_ring, R0_mm=field.R0)
        meta[label] = {"piston": piston, "D": nD, "U_ring": field.U_ring, "Fr": Fr}

        pos_all, d_all, St_all, seed_all = [], [], [], []
        for s in range(N_SEEDS):
            P = sd.seed_positions(ell, n=N_MERID, n_phi=N_PHI, seed=s)
            St = stokes_number(d_common, field.U_ring, field.R0)
            for i in range(n):
                pos_all.append(P[i]); d_all.append(d_common[i])
                St_all.append(float(St[i])); seed_all.append(s)
        res = ad.advect_bubbles(fdir, np.array(pos_all), np.array(d_all), np.array(St_all),
                                Fr, n_workers=WORKERS, gravity=True, t_max=T_MAX)
        for i, r in enumerate(res):
            rec["piston"].append(piston); rec["D"].append(nD); rec["seed"].append(seed_all[i])
            rec["d"].append(d_all[i]); rec["St"].append(St_all[i])
            rec["escaped"].append(bool(r.escaped)); rec["t_escape"].append(r.t_escape)
            rec["exit_face"].append(r.exit_face)
        nb = sum(1 for r in res if r.exit_face == "r_top")
        print(f"  {label}: U_ring={field.U_ring:.0f} Fr={Fr:.3f}  "
              f"buoyant={nb} captured={sum(1 for r in res if not r.escaped)}/{len(res)}")

    np.savez(os.path.join(OUT, "uniform.npz"), **{k: np.array(v) for k, v in rec.items()})
    with open(os.path.join(OUT, "uniform_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    _plots(meta)
    print("-> outputs/uniform.npz, figC_uniform_volume_vs_time.png, figE_retention_vs_station.png")


def _plots(meta):
    Z = np.load(os.path.join(OUT, "uniform.npz"), allow_pickle=True)
    piston, D, d, te, esc = Z["piston"], Z["D"], Z["d"], Z["t_escape"], Z["escaped"]
    face = Z["exit_face"].astype(str)
    tgrid = np.linspace(0, T_MAX, 120)

    # Fig C uniform: remaining volume fraction vs time (same population everywhere)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, m in meta.items():
        sel = (piston == m["piston"]) & (D == m["D"])
        vol = np.pi / 6 * d[sel] ** 3
        remain = np.array([vol[(~esc[sel]) | (te[sel] > t)].sum() for t in tgrid]) / vol.sum()
        ax.plot(tgrid, remain, color=PCOL[m["piston"]], ls=LS[m["D"]], lw=1.8, label=label)
    ax.set_xlabel("dimensionless time $t^* = t\\,U_{ring}/R_0$")
    ax.set_ylabel("remaining in-ring volume fraction")
    ax.set_title("Time development — IDENTICAL population at every station (fair)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figC_uniform_volume_vs_time.png"), dpi=140)

    # Fig E: asymptotic retained-volume fraction vs station
    fig, ax = plt.subplots(figsize=(7, 5))
    for pv in (120, 200):
        Ds, ret = [], []
        for label, m in meta.items():
            if m["piston"] != pv:
                continue
            sel = (piston == m["piston"]) & (D == m["D"])
            vol = np.pi / 6 * d[sel] ** 3
            ret.append(vol[~esc[sel]].sum() / vol.sum()); Ds.append(m["D"])
        order = np.argsort(Ds)
        ax.plot(np.array(Ds)[order], np.array(ret)[order], "o-", color=PCOL[pv], label=f"Up={pv} mm/s")
    ax.set_xlabel("downstream station (D, x40 mm)"); ax.set_ylabel("retained volume fraction (t*=20)")
    ax.set_title("Ring retention of a common bubble population vs station")
    ax.set_xticks([5, 10, 15]); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figE_retention_vs_station.png"), dpi=140)


if __name__ == "__main__":
    main()
