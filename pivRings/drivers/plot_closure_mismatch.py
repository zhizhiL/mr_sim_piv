#!/usr/bin/env python3
"""
plot_closure_mismatch.py — the real reason the fresh fast stations don't trap at
their thin-ring speed: a closure-speed mismatch, NOT a FOV-size limit.

For 200_5D and 200_10D the co-moving atmosphere is OPEN at the thin-ring U_ring
(streamlines pass straight through; core tracers wash out) and only CLOSES when
U_ring is lowered into the closure band (§7c).  Same field, same FOV — only the
co-moving frame speed differs.
"""
from __future__ import annotations
import json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import build_field as bf, seeding as sd, ftle as F
from constants import R0

OUT = os.path.join(ROOT, "outputs")
PANELS = [("200_5D  thin-ring U=110 (open)",  "200_5D", 110),
          ("200_5D  closure U=60 (closed)",   "200_5D", 60),
          ("200_10D thin-ring U=84 (open)",   "200_10D", 84),
          ("200_10D closure U=48 (closed)",   "200_10D", 48)]


def build_at(label, U):
    d = os.path.join(ROOT, "fields", label)
    mp = os.path.join(d, "mean_reg.pkl"); mp = mp if os.path.exists(mp) else os.path.join(d, "mean_field.pkl")
    mean = pickle.load(open(mp, "rb"))
    sign = float(np.sign(json.load(open(os.path.join(d, "meta.json"))).get("U_ring_signed", 1.0))) or 1.0
    return bf.build_field(mean, U_ring=sign * U, R0=R0), mean


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (lab, label, U) in zip(axes.ravel(), PANELS):
        field, mean = build_at(label, U)
        ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core="upper")
        xg = np.linspace(field.x_axis[0], field.x_axis[-1], 220)
        zg = np.linspace(-field.r_axis[-1], field.r_axis[-1], 200)
        Xg, Zg = np.meshgrid(xg, zg)
        P = np.column_stack([Xg.ravel(), np.zeros(Xg.size), Zg.ravel()])
        u = field.velocity(P)
        U2 = u[:, 0].reshape(Xg.shape); W2 = u[:, 2].reshape(Xg.shape)
        ax.streamplot(xg, zg, U2, W2, color=np.hypot(U2, W2), cmap="viridis",
                      density=1.4, linewidth=0.7, arrowsize=0.6)
        th = np.linspace(0, 2 * np.pi, 200); c_, s_ = np.cos(ell.tilt), np.sin(ell.tilt)
        ex = ell.x_c / field.R0 + (ell.a_xi / field.R0) * np.cos(th) * c_ - (ell.a_eta / field.R0) * np.sin(th) * s_
        er = ell.r_c / field.R0 + (ell.a_xi / field.R0) * np.cos(th) * s_ + (ell.a_eta / field.R0) * np.sin(th) * c_
        ax.plot(ex, er, "r--", lw=1.0); ax.plot(ex, -er, "r--", lw=1.0)
        seeds = sd.seed_core_surface(ell, 400, np.random.default_rng(0))
        Pf, alive = F.fluid_flow_map(field, np.column_stack([seeds[:, 0], seeds[:, 2]]),
                                     T=15.0, direction=1)
        ax.scatter(Pf[alive, 0], Pf[alive, 1], s=2, c="k", alpha=0.4, zorder=5)
        ax.set_title(f"{lab}   core tracers still inside = {alive.mean():.0%}", fontsize=10)
        ax.set_xlabel("x*"); ax.set_ylabel("z*"); ax.set_aspect("equal")
    fig.suptitle("Closure-speed mismatch (NOT FOV): same field & FOV, only the co-moving U_ring "
                 "differs.\nThin-ring speed → open atmosphere (tracers wash out); closure-band speed "
                 "→ closed atmosphere (tracers trapped).", fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT, "closure_mismatch_streamlines.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
