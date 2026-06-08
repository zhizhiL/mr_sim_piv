#!/usr/bin/env python3
"""
plot_ensembles.py — deliverable figures from run_ensembles.py output.

Escape is split by exit face: buoyant detrainment leaves through the top radial
wall (r_top, z>0, gravity is +z); advective loss leaves through an axial FOV
wall (x_min/x_max) while the bubble is still orbiting — that is a FOV-truncation
artifact, not physical detrainment.

  Fig A  fate breakdown per condition (captured / buoyant / advective / r_bot)
  Fig B  critical BUOYANT-detrainment size d_crit vs station (loading-independent)
  Fig C  time development of trapped bubble volume (population, per condition)
  Fig D  governing relation: buoyant St_crit vs Fr^2  (St/Fr^2 = W*)
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import stokes_number, R0

OUT = os.path.join(ROOT, "outputs")
Z = np.load(os.path.join(OUT, "ensembles.npz"), allow_pickle=True)
META = json.load(open(os.path.join(OUT, "ensembles_meta.json")))

kind, piston, D = Z["kind"], Z["piston"], Z["D"]
loading, seed = Z["loading"], Z["seed"]
d, St, escaped, t_esc = Z["d"], Z["St"], Z["escaped"], Z["t_esc"] if "t_esc" in Z else Z["t_escape"]
face = Z["exit_face"].astype(str)

CONDS = sorted(META.keys(), key=lambda c: (META[c]["piston"], META[c]["loading"], META[c]["D"]))
PCOL = {120: "#1f77b4", 200: "#d62728"}
MARK = {5: "o", 10: "s", 15: "^"}
BUOYANT = "r_top"

# reliability: a station is trustworthy only if its co-moving atmosphere closes
# (a core-seeded tracer is trapped).  From build_from_cache -> field_summary.json.
try:
    _FS = json.load(open(os.path.join(OUT, "field_summary.json")))
    RELIABLE = {(v["piston"], v["D"]): (v.get("core_trapped", 0) >= 0.5)
                for v in _FS.values()}
except Exception:
    RELIABLE = {}


def reliable(piston_v, D_v):
    return RELIABLE.get((piston_v, D_v), True)


def _cm(c):
    m = META[c]
    return (piston == m["piston"]) & (D == m["D"]) & (loading == m["loading"])


def buoyant_dcrit(piston_v, D_v):
    """50%-buoyant-escape diameter from the sweep (loading-independent)."""
    base = (kind == 0) & (piston == piston_v) & (D == D_v)
    crits = []
    for s in np.unique(seed[base]):
        m = base & (seed == s)
        dg = np.unique(d[m])
        P = np.array([(face[m & (d == dd)] == BUOYANT).mean() for dd in dg])
        if P.max() < 0.5:
            crits.append(np.nan); continue
        if P[0] >= 0.5:
            crits.append(dg[0]); continue
        i = int(np.argmax(P >= 0.5))
        x0, x1, y0, y1 = dg[i - 1], dg[i], P[i - 1], P[i]
        crits.append(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0 + 1e-12))
    c = np.array(crits, float)
    return np.nanmean(c), np.nanstd(c), np.isfinite(c).mean()


# ---------------------------------------------------------------- Fig A
def fig_fate_breakdown():
    cats = [("captured", "#6c757d"), ("buoyant (r_top)", "#2ca02c"),
            ("advective (x-wall)", "#ff7f0e"), ("r_bot", "#9467bd")]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    bottoms = np.zeros(len(CONDS))
    frac = {name: [] for name, _ in cats}
    for c in CONDS:
        m = _cm(c) & (kind == 1)
        n = max(m.sum(), 1)
        f = face[m]
        frac["captured"].append((~escaped[m]).sum() / n)
        frac["buoyant (r_top)"].append((f == "r_top").sum() / n)
        frac["advective (x-wall)"].append(np.isin(f, ["x_min", "x_max"]).sum() / n)
        frac["r_bot"].append((f == "r_bot").sum() / n)
        labels.append(c.replace("_", "\n", 1))
    x = np.arange(len(CONDS))
    for name, col in cats:
        ax.bar(x, frac[name], bottom=bottoms, color=col, label=name, width=0.7)
        bottoms += np.array(frac[name])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("fraction of seeded bubbles")
    ax.set_title("Bubble fate by condition  "
                 "(advective = FOV-truncation loss; buoyant = physical detrainment)",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.subplots_adjust(bottom=0.2, top=0.93)
    fig.savefig(os.path.join(OUT, "figA_fate_breakdown.png"), dpi=140)


# ---------------------------------------------------------------- Fig B
def fig_buoyant_dcrit():
    """Critical detrainment size vs station.  Only stations whose co-moving
    atmosphere closes (trapping) are physically reliable; FOV-limited stations
    are drawn as faded open markers (their threshold is biased by pass-through
    bubbles) and NOT connected."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pv in (120, 200):
        rel_D, rel_mu, rel_sg = [], [], []
        for Dv in (5, 10, 15):
            mu, sg, cov = buoyant_dcrit(pv, Dv)
            if not np.isfinite(mu):
                continue
            if reliable(pv, Dv):
                rel_D.append(Dv); rel_mu.append(mu); rel_sg.append(sg)
            else:
                ax.errorbar(Dv, mu, yerr=sg, marker="o", ms=8, mfc="white",
                            color=PCOL[pv], alpha=0.5, capsize=3)
        if rel_D:
            ax.errorbar(rel_D, rel_mu, yerr=rel_sg, marker="o", capsize=3, lw=2,
                        color=PCOL[pv], label=f"Up={pv} mm/s (trapping)")
    ax.scatter([], [], marker="o", facecolor="white", edgecolor="gray",
               label="FOV-limited (unreliable)")
    ax.set_xlabel("downstream station (D, x40 mm)")
    ax.set_ylabel("critical buoyant-detrainment diameter $d_{crit}$ (mm)")
    ax.set_title("Critical detrainment size vs station\n"
                 "(solid = closed atmosphere / reliable; open = FOV-limited)")
    ax.set_xticks([5, 10, 15]); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB_buoyant_dcrit.png"), dpi=140)


# ---------------------------------------------------------------- Fig C
def fig_time_development():
    tgrid = np.linspace(0, 20, 120)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in CONDS:
        m = _cm(c) & (kind == 1)
        dd, te, esc = d[m], t_esc[m], escaped[m]
        vol = np.pi / 6 * dd ** 3
        remain = np.array([vol[(~esc) | (te > t)].sum() for t in tgrid]) / vol.sum()
        mm = META[c]
        ax.plot(tgrid, remain, color=PCOL[mm["piston"]],
                ls={5: "-", 10: "--", 15: ":"}[mm["D"]],
                alpha=0.85 if mm["loading"] == 40 else 0.5,
                lw=1.7, label=c)
    ax.set_xlabel("dimensionless time  $t^* = t\\,U_{ring}/R_0$")
    ax.set_ylabel("remaining in-ring volume fraction")
    ax.set_title("Time development of trapped bubble volume")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figC_volume_vs_time.png"), dpi=140)


# ---------------------------------------------------------------- Fig D
def fig_governing_relation():
    fig, ax = plt.subplots(figsize=(7, 5))
    Fr2, Stc = [], []   # fit uses reliable (trapping) stations only
    for pv in (120, 200):
        for Dv in (5, 10, 15):
            mu, sg, cov = buoyant_dcrit(pv, Dv)
            if not np.isfinite(mu):
                continue
            c = f"{pv}_{Dv}D"
            U = next(META[k]["U_ring"] for k in META if META[k]["piston"] == pv and META[k]["D"] == Dv)
            Fr = next(META[k]["Fr"] for k in META if META[k]["piston"] == pv and META[k]["D"] == Dv)
            stc = float(stokes_number(mu, U, R0))
            rel = reliable(pv, Dv)
            ax.errorbar(Fr ** 2, stc, marker=MARK[Dv], ms=10, color=PCOL[pv],
                        mfc=(PCOL[pv] if rel else "white"), alpha=1.0 if rel else 0.5,
                        label=c + ("" if rel else " (FOV-lim)"))
            if rel:
                Fr2.append(Fr ** 2); Stc.append(stc)
    Fr2, Stc = np.array(Fr2), np.array(Stc)
    if len(Fr2) >= 2:
        W = float(np.sum(Fr2 * Stc) / np.sum(Fr2 * Fr2))
        xx = np.linspace(0, Fr2.max() * 1.1, 50)
        ax.plot(xx, W * xx, "k--", lw=1.5, label=f"$St_{{crit}}=W^*Fr^2$,  $W^*$={W:.2f}")
    ax.set_xlabel("$Fr^2$"); ax.set_ylabel("buoyant critical Stokes $St(d_{crit})$")
    ax.set_title("Governing detrainment relation  (buoyancy number $St/Fr^2$)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figD_governing_relation.png"), dpi=140)


if __name__ == "__main__":
    fig_fate_breakdown(); fig_buoyant_dcrit(); fig_time_development(); fig_governing_relation()
    print(f"{'cond':10s} {'U_ring':>7s} {'Fr':>6s} {'d_crit(buoy)':>13s} {'resolved%':>9s}")
    for pv in (120, 200):
        for Dv in (5, 10, 15):
            mu, sg, cov = buoyant_dcrit(pv, Dv)
            U = next(META[k]["U_ring"] for k in META if META[k]["piston"] == pv and META[k]["D"] == Dv)
            Fr = next(META[k]["Fr"] for k in META if META[k]["piston"] == pv and META[k]["D"] == Dv)
            ds = f"{mu:.2f}+/-{sg:.2f}" if np.isfinite(mu) else "n/a (FOV)"
            print(f"{pv}_{Dv}D{'':3s} {U:7.1f} {Fr:6.3f} {ds:>13s} {100*cov:8.0f}%")
    print("\n-> figA_fate_breakdown, figB_buoyant_dcrit, figC_volume_vs_time, "
          "figD_governing_relation (.png in outputs/)")
