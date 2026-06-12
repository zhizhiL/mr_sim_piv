#!/usr/bin/env python3
"""
plot_atmosphere.py — co-moving meridional streamlines + the recirculation
atmosphere, to show WHY some stations trap and others barely do.

Evidence for §7d: the FOV extents are comparable across stations, so the weak
trapping of the fresh fast rings is NOT a FOV-size limit — it is a closure /
recirculation-strength issue (REPORT §3).  Streamlines are drawn in the (x,z)
plane; on-axis stagnation points (front+rear of a closed atmosphere) are marked,
and a core-seeded tracer blob is integrated to outline the trapped region.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import build_field as bf, seeding as sd, ftle as F

OUT = os.path.join(ROOT, "outputs")
# (display label, field dir) — same field versions as the all-six d_crit comparison
PANELS = [("120_5D  (U=55.6, avg)",     "120_5D"),
          ("120_10D (U=40.8, avg)",     "120_10D"),
          ("120_15D (R 15→23, isotropic)", "120_15D_geom"),
          ("200_5D  (U=110, 1-frame)",  "200_5D_sf"),
          ("200_10D (U=84, 1-frame)",   "200_10D_sf"),
          ("200_15D (a_eq fixed, ×0.33 strength)", "200_15D_scaled")]


def axis_stagnation(field):
    xg = np.linspace(field.x_axis[0], field.x_axis[-1], 600)
    P = np.column_stack([xg, np.zeros_like(xg), np.full_like(xg, 1e-4)])
    ux = field.velocity(P)[:, 0]
    s = np.sign(ux); idx = np.where(s[:-1] * s[1:] < 0)[0]
    return xg[idx]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (lab, fdir) in zip(axes.ravel(), PANELS):
        field = bf.load_field(os.path.join(ROOT, "fields", fdir))
        mean = bf.load_mean_field(os.path.join(ROOT, "fields", fdir))
        ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core="upper")
        xg = np.linspace(field.x_axis[0], field.x_axis[-1], 220)
        zg = np.linspace(-field.r_axis[-1], field.r_axis[-1], 200)
        Xg, Zg = np.meshgrid(xg, zg)
        P = np.column_stack([Xg.ravel(), np.zeros(Xg.size), Zg.ravel()])
        u = field.velocity(P)
        U = u[:, 0].reshape(Xg.shape); W = u[:, 2].reshape(Xg.shape)
        spd = np.hypot(U, W)
        xc = ell.x_c / field.R0                               # ring-core x*: re-centre on it
        ax.streamplot(xg - xc, zg, U, W, color=spd, cmap="viridis", density=1.4,
                      linewidth=0.7, arrowsize=0.6)
        # core ellipses (upper z>0, lower z<0)
        th = np.linspace(0, 2 * np.pi, 200); c_, s_ = np.cos(ell.tilt), np.sin(ell.tilt)
        ex = (ell.a_xi / field.R0) * np.cos(th) * c_ - (ell.a_eta / field.R0) * np.sin(th) * s_
        er = ell.r_c / field.R0 + (ell.a_xi / field.R0) * np.cos(th) * s_ + (ell.a_eta / field.R0) * np.sin(th) * c_
        ax.plot(ex, er, "r--", lw=1.0); ax.plot(ex, -er, "r--", lw=1.0)
        # trapped region: core-seeded tracers still inside after T*=15
        rng = np.random.default_rng(0)
        seeds = sd.seed_core_surface(ell, 400, rng)
        P0 = np.column_stack([seeds[:, 0], seeds[:, 2]])     # (x, z)
        Pf, alive = F.fluid_flow_map(field, P0, T=15.0, direction=1)
        ax.scatter(Pf[alive, 0] - xc, Pf[alive, 1], s=2, c="k", alpha=0.35, zorder=5)
        for xs in axis_stagnation(field):
            ax.plot(xs - xc, 0, "m*", ms=11, zorder=6)
        ax.set_title(f"{lab}   (trapped tracers={alive.mean():.0%})", fontsize=10)
        ax.set_xlabel("(x − x_core)/R0"); ax.set_ylabel("z* = z/R0 (gravity)")
        ax.set_aspect("equal"); ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 3.0)
    fig.suptitle("Co-moving atmosphere & trapped region — re-centred on the ring core, common "
                 "axes (x,z both /R0)\n(red dashed = core; magenta * = on-axis stagnation pts; "
                 "black dots = core tracers still inside at t*=15)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT, "atmosphere_streamlines.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
