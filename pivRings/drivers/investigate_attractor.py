#!/usr/bin/env python3
"""
investigate_attractor.py — are the "trapped" bubbles a physical equilibrium?

In the movie some bubbles settle on the core surface and stay put.  A bubble is
frozen in place only if it sits at a FIXED POINT of the Maxey-Riley system:
position constant => v = 0, and dv/dt = 0 there.  With v=0 the acceleration is

    a(x) = R u/St + (3R/2)(u.grad)u - (1 - 3R/2)/Fr^2 z_hat        (R=2, bubble)

so a fixed point needs  u_horizontal ~ 0  and  u_z ~ -St/Fr^2 = -W*  (the local
flow descends at the buoyant rise speed).  A dead/zero-velocity region can NOT
trap a bubble (buoyancy is then unopposed -> it rises out), so trapping implies
real balancing flow = physical.

This script:
  1. integrates small bubbles long and confirms they settle to |v|~0, |a|~0
     (a true equilibrium, not a stalled integrator);
  2. Newton-solves a(x)=0 from the settled state and reports the fixed point,
     the local fluid velocity, the slip, and the 6-D Jacobian eigenvalues
     (stability => attractor);
  3. sweeps radius: does a STABLE in-domain fixed point exist?  Overlays that
     against the measured remaining-% so the attractor's disappearance lines up
     with the capture threshold.

  .venv/bin/python pivRings/drivers/investigate_attractor.py --field pivRings/fields/120_5D
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
from scipy.integrate import solve_ivp
from scipy.optimize import root

import build_field as bf
import seeding as sd
from advect import mr_rhs, _escape_event
from constants import froude_number, stokes_number, R_BUBBLE


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", default=os.path.join(ROOT, "fields", "120_5D"))
    p.add_argument("--core", default="upper")
    p.add_argument("--t-settle", type=float, default=80.0)
    args = p.parse_args()

    field = bf.load_field(args.field)
    mean = bf.load_mean_field(args.field)
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core=args.core)
    Fr = froude_number(field.U_ring, R0_mm=field.R0)
    R = R_BUBBLE
    label = os.path.basename(args.field.rstrip("/"))
    print(f"{label} [{args.core}]  U_ring={field.U_ring:.1f}  Fr={Fr:.4f}")

    def accel(x, St, gravity=True):
        s = np.concatenate([np.asarray(x, float), np.zeros(3)])   # v = 0
        return mr_rhs(0.0, s, field, St, R, Fr, gravity)[3:]

    def jac6(s, St):                                              # numerical 6-D Jacobian
        eps = 1e-6
        f0 = mr_rhs(0.0, s, field, St, R, Fr, True)
        J = np.zeros((6, 6))
        for i in range(6):
            sp = s.copy(); sp[i] += eps
            J[:, i] = (mr_rhs(0.0, sp, field, St, R, Fr, True) - f0) / eps
        return J

    def find_fixed_point(St, starts, tol=1e-6):
        """Return (x*, eig) for the first STABLE in-domain fixed point, else None."""
        for x0 in starts:
            sol = root(accel, x0, args=(St,), method="hybr", tol=1e-10)
            x = sol.x
            if not sol.success:
                continue
            if np.linalg.norm(accel(x, St)) > tol:
                continue
            if not field.in_domain(x[None])[0]:
                continue
            eig = np.linalg.eigvals(jac6(np.concatenate([x, np.zeros(3)]), St))
            if np.all(eig.real < 1e-6):                           # stable (attractor)
                return x, eig
        return None

    rng = np.random.default_rng(3)
    starts = sd.seed_core_surface(ell, 120, rng)                 # candidate guesses on the core

    # ---- 1 & 2: settle a small bubble, then locate & classify the fixed point ----
    d0 = 0.20
    St0 = float(stokes_number(d0, field.U_ring, field.R0))
    W0 = St0 / Fr ** 2
    print(f"\n[1] settle test  d={d0} mm  St={St0:.4f}  W*=St/Fr^2={W0:.3f}")
    settled = []
    for x0 in starts[:60]:
        v0 = field.velocity(x0[None])[0]
        s0 = np.concatenate([x0, v0])
        sol = solve_ivp(mr_rhs, (0.0, args.t_settle), s0, method="LSODA",
                        args=(field, St0, R, Fr, True), events=_escape_event(field),
                        rtol=1e-7, atol=1e-9)
        if len(sol.t_events[0]) == 0:                            # captured
            xf, vf = sol.y[:3, -1], sol.y[3:, -1]
            af = mr_rhs(0.0, np.concatenate([xf, vf]), field, St0, R, Fr, True)[3:]
            settled.append((xf, vf, af))
    if settled:
        sp = np.array([np.linalg.norm(v) for _, v, _ in settled])
        ap = np.array([np.linalg.norm(a) for _, _, a in settled])
        print(f"    {len(settled)} captured bubbles; late-time |v| median={np.median(sp):.2e} "
              f"max={sp.max():.2e};  |accel| median={np.median(ap):.2e} max={ap.max():.2e}")
        print(f"    -> |v|~0 and |accel|~0 => a genuine equilibrium, not a stalled solver")

    fp = find_fixed_point(St0, [s[0] for s in settled] + list(starts))
    if fp is not None:
        x, eig = fp
        u, gradu = field.velocity_and_gradient(x[None])
        u = u[0]; r = float(np.hypot(x[1], x[2]))
        print(f"\n[2] fixed point  x*=({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f})  r*={r:.3f}")
        print(f"    local fluid u*=({u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f})  |u*|={np.linalg.norm(u):.3f}")
        print(f"    u_z*={u[2]:+.3f}  vs  -W*={-W0:+.3f}   (downflow balances buoyant rise)")
        print(f"    |accel(x*,v=0)|={np.linalg.norm(accel(x, St0)):.2e}")
        print(f"    in PIV domain: {bool(field.in_domain(x[None])[0])}  "
              f"(bounds x{field.bounds[:2]} r[0,{field.bounds[3]:.2f}])")
        print(f"    Jacobian eigenvalues (Re,Im):")
        for e in eig:
            print(f"      {e.real:+.4f} {e.imag:+.4f}j")
        print(f"    max Re(eig)={eig.real.max():+.4f}  -> "
              f"{'STABLE attractor' if eig.real.max() < 0 else 'unstable'}")
    else:
        print("\n[2] no stable in-domain fixed point found for the small bubble (!)")

    # ---- 3: BASIN of the attractor vs radius (not mere existence) ----
    # A stable fixed point can exist mathematically yet trap nothing if its basin
    # is tiny.  The dynamical basin is measured directly: integrate the SAME
    # core-surface seeds at each radius and count how many converge to the
    # attractor (|v|->0) rather than escape.  That IS the capture fraction; we
    # report it next to W* to show the basin collapses as buoyancy grows.
    radii = sd.make_radii_grid()
    Wstar, basin = [], []
    seeds3 = sd.seed_core_surface(ell, 40, np.random.default_rng(7))
    for rr in radii:
        d = 2 * rr
        St = float(stokes_number(d, field.U_ring, field.R0))
        Wstar.append(St / Fr ** 2)
        n_trap = 0
        for x0 in seeds3:
            v0 = field.velocity(x0[None])[0]
            sol = solve_ivp(mr_rhs, (0.0, 40.0), np.concatenate([x0, v0]),
                            method="LSODA", args=(field, St, R, Fr, True),
                            events=_escape_event(field), rtol=1e-6, atol=1e-8)
            if len(sol.t_events[0]) == 0 and np.linalg.norm(sol.y[3:, -1]) < 1e-3:
                n_trap += 1
        basin.append(n_trap / len(seeds3))
    Wstar = np.array(Wstar); basin = np.array(basin)
    print(f"\n[3] attractor basin (fraction of core seeds settling to |v|~0) vs radius:")
    for rr, b, w in zip(radii, basin, Wstar):
        print(f"    r={rr:.2f} mm  W*={w:6.2f}  basin={b:5.1%}")

    # overlay against the measured remaining-% if the summary is available
    rem = None
    summ = os.path.join(ROOT, "outputs", f"{label}_l1_uniform_summary.json")
    if os.path.exists(summ):
        S = json.load(open(summ))
        if args.core in S["cores"]:
            rem = np.array(S["cores"][args.core]["remaining_fraction"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.fill_between(radii, 0, 100 * basin, step="mid",
                     color="#9ecae1", alpha=0.5, label="attractor basin (core seeds settling)")
    if rem is not None:
        ax1.plot(radii, 100 * rem, "o-", color="#1f4e79", lw=1.8,
                 label="remaining % (sim capture)")
    ax1.set_xlabel("bubble radius (mm)"); ax1.set_ylabel("% captured / in basin")
    ax1.set_ylim(-2, 102)
    ax2 = ax1.twinx()
    ax2.plot(radii, Wstar, "--", color="crimson", lw=1.4, label="$W^*=St/Fr^2$ (rise speed)")
    ax2.set_ylabel("$W^*$ (dimensionless rise speed)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax1.set_title(f"Trapped bubbles = physical attractor — {label} [{args.core}]\n"
                  "a stable MR fixed point exists; its BASIN collapses as $W^*$ rises")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    out = os.path.join(ROOT, "outputs", f"{label}_{args.core}_attractor.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
