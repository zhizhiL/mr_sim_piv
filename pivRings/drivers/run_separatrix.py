#!/usr/bin/env python3
"""
run_separatrix.py — co-moving streamfunction separatrix for one station.

Computes the Stokes streamfunction (least-squares / solenoidal-projected),
extracts the atmosphere separatrix polygon, caches it to
``<field>/separatrix.npz``, and writes a diagnostic figure overlaying:

  * ψ contours + the extracted separatrix loop,
  * the FOV-trapped fluid set (forward fluid flow map) — empirical atmosphere,
  * the forward-fluid FTLE ridge (should coincide with the separatrix),
  * the core ellipse, and
  * where small / large bubbles settle (attractor inside, detrainment outside).

  .venv/bin/python pivRings/drivers/run_separatrix.py --field pivRings/fields/120_5D
"""
from __future__ import annotations

import argparse
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
import ftle as F
from constants import froude_number, stokes_number, R_BUBBLE
from advect import mr_rhs, _escape_event
from scipy.integrate import solve_ivp


def _ellipse_xr(ell):
    th = np.linspace(0, 2 * np.pi, 200)
    c_, s_ = np.cos(ell.tilt), np.sin(ell.tilt)
    ex = ell.x_c / ell.R0 + (ell.a_xi / ell.R0) * np.cos(th) * c_ - (ell.a_eta / ell.R0) * np.sin(th) * s_
    er = ell.r_c / ell.R0 + (ell.a_xi / ell.R0) * np.cos(th) * s_ + (ell.a_eta / ell.R0) * np.sin(th) * c_
    return ex, er


def settle_points(field, ell, Fr, d_mm, n=40, t=40.0):
    """Final (x, r) of core-surface seeds for one bubble size (attractor map)."""
    St = float(stokes_number(d_mm, field.U_ring, field.R0))
    seeds = sd.seed_core_surface(ell, n, np.random.default_rng(1))
    pts = []
    for x0 in seeds:
        v0 = field.velocity(x0[None])[0]
        sol = solve_ivp(mr_rhs, (0, t), np.concatenate([x0, v0]), method="LSODA",
                        args=(field, St, R_BUBBLE, Fr, True), events=_escape_event(field),
                        rtol=1e-6, atol=1e-8)
        xf = sol.y[:3, -1]
        pts.append((xf[0], float(np.hypot(xf[1], xf[2])), len(sol.t_events[0]) == 0))
    return St / Fr ** 2, np.array(pts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", default=os.path.join(ROOT, "fields", "120_5D"))
    p.add_argument("--core", default="upper")
    p.add_argument("--nx", type=int, default=160)
    p.add_argument("--nr", type=int, default=140)
    p.add_argument("--method", default="leastsq", choices=["leastsq", "integral"])
    p.add_argument("--T-trap", type=float, default=18.0)
    p.add_argument("--no-cache", action="store_true", help="recompute, don't reuse npz")
    args = p.parse_args()

    field = bf.load_field(args.field)
    mean = bf.load_mean_field(args.field)
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core=args.core)
    Fr = froude_number(field.U_ring, R0_mm=field.R0)
    label = os.path.basename(args.field.rstrip("/"))
    core_xr = (ell.x_c / field.R0, ell.r_c / field.R0)
    print(f"{label} [{args.core}]  U_ring={field.U_ring:.1f}  Fr={Fr:.4f}  method={args.method}")

    sep = sx.compute_separatrix(field, core_xr=core_xr, nx=args.nx, nr=args.nr,
                                method=args.method)
    if sep is None:
        print("!! no closed separatrix loop found"); return
    print(f"  separatrix: area={sep.area():.3f}  level={sep.level:.3g}  "
          f"x[{sep.poly[:,0].min():.2f},{sep.poly[:,0].max():.2f}] "
          f"r[{sep.poly[:,1].min():.2f},{sep.poly[:,1].max():.2f}]  "
          f"FOV-truncated={sep.fov_truncated}")
    print(f"  on-axis stagnation x*={np.round(sep.meta['axis_stagnation_x'],3)}")
    if not args.no_cache:
        sep.save(os.path.join(args.field, "separatrix.npz"))
        print(f"  cached -> {os.path.join(args.field, 'separatrix.npz')}")

    # empirical atmosphere: FOV-trapped fluid set on a grid
    xmin, xmax, _, rmax = field.bounds
    gx, gz = np.meshgrid(np.linspace(xmin + 0.02, xmax - 0.02, 90),
                         np.linspace(0.02, rmax - 0.02, 80))
    P0 = np.column_stack([gx.ravel(), gz.ravel()])
    Pf, alive = F.fluid_flow_map(field, P0, T=args.T_trap, direction=1)

    # forward fluid FTLE ridge (axisymmetric (x,z); take z>0 half as (x,r))
    xg, zg, Pg = (lambda f, nx, nz: (np.linspace(f.bounds[0], f.bounds[1], nx),
                                     np.linspace(-f.bounds[3], f.bounds[3], nz),
                                     None))(field, 140, 140)
    Xg, Zg = np.meshgrid(xg, zg)
    Pgrid = np.column_stack([Xg.ravel(), Zg.ravel()])
    Pfl, alf = F.fluid_flow_map(field, Pgrid, T=8.0, direction=1)
    ftle = F.ftle_field(xg, zg, Pfl, alf, 8.0)

    # bubble settle maps
    W_small, pts_small = settle_points(field, ell, Fr, 0.30)
    W_big, pts_big = settle_points(field, ell, Fr, 1.50)

    fig, ax = plt.subplots(figsize=(9, 7))
    # psi contours
    Xc, Rc = np.meshgrid(sep.xg, sep.rg)
    ax.contour(Xc, Rc, sep.psi, levels=14, colors="0.8", linewidths=0.6)
    # FTLE ridge (z>0 half)
    half = zg >= 0
    im = ax.pcolormesh(xg, zg[half], ftle[half, :], cmap="inferno",
                       vmin=0, vmax=np.nanpercentile(ftle, 99), shading="auto", alpha=0.55)
    fig.colorbar(im, ax=ax, label="forward fluid FTLE (ridge=separatrix)", fraction=0.04)
    # empirical trapped set
    ax.scatter(P0[alive, 0], P0[alive, 1], s=4, c="#7fc97f", alpha=0.5,
               label=f"FOV-trapped fluid (T*={args.T_trap:.0f})")
    # separatrix
    ax.plot(sep.poly[:, 0], sep.poly[:, 1], "-", color="crimson", lw=2.2,
            label="streamfunction separatrix")
    # core ellipse
    ex, er = _ellipse_xr(ell)
    ax.plot(ex, er, "k--", lw=1.2, alpha=0.7, label="core ellipse")
    # settled bubbles
    ax.scatter(pts_small[:, 0], pts_small[:, 1], s=26, c="dodgerblue",
               edgecolor="k", lw=0.3, label=f"settle d=0.3 (W*={W_small:.1f})", zorder=5)
    ax.scatter(pts_big[:, 0], pts_big[:, 1], s=26, marker="^", c="orange",
               edgecolor="k", lw=0.3, label=f"settle d=1.5 (W*={W_big:.1f})", zorder=5)
    # axis stagnation points
    for xs in sep.meta["axis_stagnation_x"]:
        ax.plot(xs, 0, "kx", ms=7, mew=1.5)
    ax.set_xlabel("x*"); ax.set_ylabel("r*")
    ax.set_xlim(xmin, xmax); ax.set_ylim(0, rmax)
    ax.set_title(f"Atmosphere separatrix — {label} [{args.core}]  ({args.method} ψ)\n"
                 "red loop = dividing streamline; green = empirically trapped; "
                 "kx = on-axis stagnation")
    ax.legend(loc="upper right", fontsize=8)
    out = os.path.join(ROOT, "outputs", f"{label}_{args.core}_separatrix.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
