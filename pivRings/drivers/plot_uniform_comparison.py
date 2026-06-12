#!/usr/bin/env python3
"""
plot_uniform_comparison.py — all-six-station comparison of the build-plan
uniform-size sims: (a) capture vs bubble size and (b) remaining volume % vs time.

Native-trapping stations use their thin-ring field; the FOV-limited stations use
the recalibrated closure-band field (<label>_uc, the d_crit-maximising U_ring).
The two diffuse-core 15D stations are drawn dashed/faded (core-fit artifacts,
REPORT §2/§4/§7d).
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, "outputs")
PCOL = {120: "#1f77b4", 200: "#d62728"}
LS = {5: "-", 10: "--", 15: ":"}

# (summary key, display label, piston, D, reliable?)
STATIONS = [
    ("120_5D",     "120_5D (U=55.6, avg)",     120, 5,  True),
    ("120_10D",    "120_10D (U=40.8, avg)",    120, 10, True),
    ("120_15D_geom", "120_15D (R 15→23, isotropic)", 120, 15, True),
    ("200_5D_sf",  "200_5D (U=110, 1-frame)",  200, 5,  True),
    ("200_10D_sf", "200_10D (U=84, 1-frame)",  200, 10, True),
    ("200_15D_scaled", "200_15D (a_eq fixed, ×0.33 strength)", 200, 15, True),
]


def dcrit(rad, rem):
    d = 2 * np.asarray(rad)
    rem = np.asarray(rem)
    if rem[0] < 0.5:
        return np.nan
    b = np.where(rem < 0.5)[0]
    if not len(b):
        return d[-1]
    i = b[0]
    return d[i - 1] + (0.5 - rem[i - 1]) * (d[i] - d[i - 1]) / (rem[i] - rem[i - 1] - 1e-12)


def main():
    rows = []
    # ---- Fig 1: capture vs size ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for key, lab, pist, D, ok in STATIONS:
        S = json.load(open(os.path.join(OUT, f"{key}_l1_uniform_summary.json")))
        rad = np.array(S["radii_mm"]); rem = np.array(S["cores"]["upper"]["remaining_fraction"])
        dc = dcrit(rad, rem)
        rows.append((lab, S["U_ring"], S["Fr"], dc))
        ax.plot(2 * rad, 100 * rem, marker="o", ms=3.5, color=PCOL[pist], ls=LS[D],
                lw=2.0 if ok else 1.3, alpha=1.0 if ok else 0.55,
                label=f"{lab}  d_crit={dc:.2f}")
    ax.axhline(50, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("bubble diameter d (mm)"); ax.set_ylabel("remaining % (captured, upper core)")
    ax.set_title("Capture vs size — all stations\n"
                 "(blue=120, red=200; solid=well-resolved, faded=FOV/core artifact*)")
    ax.set_ylim(-2, 102); ax.set_xlim(0, 1.9); ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_capture_vs_size.png"), dpi=140)
    plt.close(fig)

    # ---- Fig 2: remaining volume % vs time ----
    tgrid = np.linspace(0, 20, 160)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for key, lab, pist, D, ok in STATIONS:
        z = np.load(os.path.join(OUT, f"{key}_l1_uniform_records.npz"), allow_pickle=True)
        ve = z["vol_each"]; ri = z["radius_idx"]; esc = z["escaped"]; te = z["t_escape"]
        w = ve[ri]
        present = (~esc)[None, :] | (te[None, :] > tgrid[:, None])
        remain = (present * w).sum(axis=1) / w.sum()
        ax.plot(tgrid, 100 * remain, color=PCOL[pist], ls=LS[D],
                lw=2.0 if ok else 1.3, alpha=1.0 if ok else 0.55, label=lab)
    ax.set_xlabel("dimensionless time  $t^* = t\\,U_{ring}/R_0$")
    ax.set_ylabel("remaining in-ring volume (%)")
    ax.set_title("Remaining bubble volume vs time — all stations (uniform-count loading)")
    ax.set_ylim(0, 102); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_volume_vs_time.png"), dpi=140)
    plt.close(fig)

    print(f"{'station':26} {'U_ring':>7} {'Fr':>6} {'d_crit':>7}")
    for lab, U, Fr, dc in rows:
        print(f"{lab:26} {U:7.1f} {Fr:6.3f} {dc:7.2f}")
    print("-> compare_capture_vs_size.png, compare_volume_vs_time.png")


if __name__ == "__main__":
    main()
