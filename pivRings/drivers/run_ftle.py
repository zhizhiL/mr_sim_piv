#!/usr/bin/env python3
"""
run_ftle.py — FTLE fields for one station in the invariant (x,z) plane.

  * fluid FORWARD FTLE  -> repelling LCS = the ring's atmosphere separatrix;
  * fluid BACKWARD FTLE -> attracting LCS (where tracers accumulate);
  * inertial FORWARD FTLE for several bubble sizes -> the capture/escape
    basin boundary, and how it pinches off as W*=St/Fr^2 grows.

The known trapping attractor (from investigate_attractor.py) and the core
ellipse are overlaid for validation.

  .venv/bin/python pivRings/drivers/run_ftle.py --field pivRings/fields/120_5D
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
import ftle as F
from constants import froude_number, stokes_number


def _grid(field, nx, nz, pad=0.0):
    xmin, xmax, _, rmax = field.bounds
    xg = np.linspace(xmin + pad, xmax - pad, nx)
    zg = np.linspace(-rmax + pad, rmax - pad, nz)
    Xg, Zg = np.meshgrid(xg, zg)
    P0 = np.column_stack([Xg.ravel(), Zg.ravel()])
    return xg, zg, P0


def _ellipse_xz(ell):
    th = np.linspace(0, 2 * np.pi, 200)
    c_, s_ = np.cos(ell.tilt), np.sin(ell.tilt)
    ex = ell.x_c / ell.R0 + (ell.a_xi / ell.R0) * np.cos(th) * c_ - (ell.a_eta / ell.R0) * np.sin(th) * s_
    er = ell.r_c / ell.R0 + (ell.a_xi / ell.R0) * np.cos(th) * s_ + (ell.a_eta / ell.R0) * np.sin(th) * c_
    return ex, er


def _axis_stagnation_count(field_U):
    """Number of on-axis (r->0) stagnation points of the co-moving axial flow,
    i.e. sign changes of Ux*(x, 0) along x.  A closed ring atmosphere has a
    front+rear pair (count>=2); an open/through-flow field has 0."""
    xg = np.linspace(field_U.x_axis[0], field_U.x_axis[-1], 400)
    P = np.column_stack([xg, np.zeros_like(xg), np.full_like(xg, 1e-4)])
    ux = field_U.velocity(P)[:, 0]                 # dimensionless co-moving Ux*
    s = np.sign(ux)
    return int(np.sum(s[:-1] * s[1:] < 0))


def uring_sweep(args, field, mean):
    """Atmosphere closure vs U_ring (REPORT 6.1): rebuild the co-moving field at
    each candidate U_ring and measure (a) the trapped fraction of tracers seeded
    around the core (Lagrangian atmosphere size) and (b) the on-axis stagnation
    count (topological closure)."""
    label = os.path.basename(args.field.rstrip("/"))
    f0, f1 = [float(s) for s in args.u_frac.split(",")]
    Us = np.linspace(f0 * field.U_ring, f1 * field.U_ring, args.u_n)
    R0, y_axis = field.R0, field.y_axis_mm

    # tracer seed box around the upper core (in *dimensionless* x,z)
    ell = sd.fit_core_ellipse(mean, y_axis=y_axis, R0=R0, core=args.core)
    xc, rc = ell.x_c / R0, ell.r_c / R0
    gx, gz = np.meshgrid(np.linspace(xc - 1.2, xc + 1.2, 40),
                         np.linspace(max(rc - 1.2, 0.05), rc + 1.2, 40))
    seeds = np.column_stack([gx.ravel(), gz.ravel()])

    trapped, nstag = [], []
    for U in Us:
        fU = bf.build_field(mean, U, R0=R0, y_axis=y_axis)
        inside = F._alive(fU, seeds)
        Pf, alive = F.fluid_flow_map(fU, seeds[inside], T=18.0, direction=1)
        trapped.append(alive.mean() if inside.any() else 0.0)
        nstag.append(_axis_stagnation_count(fU))
        print(f"  U_ring={U:7.1f}  trapped={trapped[-1]:5.1%}  on-axis stagn={nstag[-1]}",
              flush=True)
    trapped = np.array(trapped); nstag = np.array(nstag)

    # closure speed = highest U with a closed atmosphere (trapped>=50% AND >=2 stagn)
    closed = (trapped >= 0.5) & (nstag >= 2)
    U_close = float(Us[closed].max()) if closed.any() else np.nan

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(Us, 100 * trapped, "o-", color="#1f4e79", label="trapped fraction (core seeds)")
    ax1.axvline(field.U_ring, ls="--", color="crimson",
                label=f"thin-ring U_ring={field.U_ring:.0f}")
    if np.isfinite(U_close):
        ax1.axvline(U_close, ls=":", color="green", label=f"closure U≈{U_close:.0f}")
    ax1.set_xlabel("U_ring (mm/s)"); ax1.set_ylabel("trapped fraction (%)", color="#1f4e79")
    ax1.set_ylim(-2, 102)
    ax2 = ax1.twinx()
    ax2.step(Us, nstag, where="mid", color="#888", alpha=0.8, label="on-axis stagnation pts")
    ax2.set_ylabel("# on-axis stagnation points", color="#888")
    ax2.set_yticks([0, 1, 2, 3, 4])
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.set_title(f"Atmosphere closure vs U_ring — {label} [{args.core}]\n"
                  "closed (trapping) below the closure speed; thin-ring value marked")
    out = os.path.join(ROOT, "outputs", f"{label}_{args.core}_uring_closure.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"thin-ring U_ring={field.U_ring:.1f};  closure U≈"
          f"{U_close:.1f} mm/s  ({'thin-ring TRAPS' if field.U_ring <= U_close else 'thin-ring OPEN'})"
          if np.isfinite(U_close) else "no closed atmosphere found in range")
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", default=os.path.join(ROOT, "fields", "120_5D"))
    p.add_argument("--core", default="upper")
    p.add_argument("--nx", type=int, default=180)
    p.add_argument("--nz", type=int, default=180)
    p.add_argument("--T-fluid", type=float, default=8.0)
    p.add_argument("--T-inertial", type=float, default=6.0)
    p.add_argument("--radii", default="0.10,0.30,0.46,0.70",
                   help="bubble radii (mm) for inertial FTLE panels")
    p.add_argument("--uring-sweep", action="store_true",
                   help="sweep U_ring and detect atmosphere closure (skips FTLE panels)")
    p.add_argument("--u-frac", default="0.4,2.0",
                   help="U_ring sweep range as fractions of the station U_ring")
    p.add_argument("--u-n", type=int, default=15)
    args = p.parse_args()

    field = bf.load_field(args.field)
    mean = bf.load_mean_field(args.field)
    if args.uring_sweep:
        uring_sweep(args, field, mean)
        return
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core=args.core)
    Fr = froude_number(field.U_ring, R0_mm=field.R0)
    label = os.path.basename(args.field.rstrip("/"))
    xg, zg, P0 = _grid(field, args.nx, args.nz)
    ex, er = _ellipse_xz(ell)
    print(f"{label} [{args.core}]  U_ring={field.U_ring:.1f}  Fr={Fr:.4f}  "
          f"grid {args.nx}x{args.nz}")

    # ---------- fluid FTLE: forward (separatrix) and backward (attracting) ----
    print("fluid forward FTLE ...", flush=True)
    Pf, al = F.fluid_flow_map(field, P0, args.T_fluid, direction=1)
    ftle_fwd = F.ftle_field(xg, zg, Pf, al, args.T_fluid)
    print("fluid backward FTLE ...", flush=True)
    Pb, alb = F.fluid_flow_map(field, P0, args.T_fluid, direction=-1)
    ftle_bwd = F.ftle_field(xg, zg, Pb, alb, args.T_fluid)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, fl, ttl in ((axes[0], ftle_fwd, "forward (repelling = atmosphere separatrix)"),
                        (axes[1], ftle_bwd, "backward (attracting)")):
        vmax = np.nanpercentile(fl, 99)
        im = ax.pcolormesh(xg, zg, fl, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
        ax.plot(ex, er, "c--", lw=1.0, alpha=0.8)
        ax.plot(ex, -er, "c--", lw=1.0, alpha=0.8)
        ax.set_xlabel("x*"); ax.set_ylabel("z* (gravity)"); ax.set_title(ttl, fontsize=10)
        ax.set_aspect("equal"); fig.colorbar(im, ax=ax, label="FTLE")
    fig.suptitle(f"Fluid FTLE — {label} [{args.core}]  (T*={args.T_fluid})")
    fig.tight_layout()
    out1 = os.path.join(ROOT, "outputs", f"{label}_{args.core}_ftle_fluid.png")
    fig.savefig(out1, dpi=140); plt.close(fig)
    print(f"-> {out1}")

    # ---------- inertial FTLE per bubble size ----
    radii = [float(s) for s in args.radii.split(",")]
    ncol = len(radii)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 5.0), squeeze=False)
    for k, rr in enumerate(radii):
        St = float(stokes_number(2 * rr, field.U_ring, field.R0))
        W = St / Fr ** 2
        print(f"inertial FTLE r={rr} mm (St={St:.4f}, W*={W:.2f}) ...", flush=True)
        Pi, ali = F.inertial_flow_map(field, P0, St, Fr, args.T_inertial)
        fl = F.ftle_field(xg, zg, Pi, ali, args.T_inertial)
        ax = axes[0][k]
        vmax = np.nanpercentile(fl, 99)
        im = ax.pcolormesh(xg, zg, fl, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
        ax.plot(ex, er, "c--", lw=0.9, alpha=0.8); ax.plot(ex, -er, "c--", lw=0.9, alpha=0.8)
        ax.set_title(f"r={rr} mm  W*={W:.1f}", fontsize=10)
        ax.set_xlabel("x*"); ax.set_aspect("equal")
        if k == 0:
            ax.set_ylabel("z* (gravity)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Inertial (forward) FTLE vs bubble size — {label} [{args.core}]  "
                 f"(T*={args.T_inertial}); ridge = capture/escape basin boundary")
    fig.tight_layout()
    out2 = os.path.join(ROOT, "outputs", f"{label}_{args.core}_ftle_inertial.png")
    fig.savefig(out2, dpi=140); plt.close(fig)
    print(f"-> {out2}")


if __name__ == "__main__":
    main()
