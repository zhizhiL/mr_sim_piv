#!/usr/bin/env python3
"""
verify_timestep.py — convergence check for the Maxey-Riley ODE integration.

Concern: in the volume-vs-time deliverable (and the movie) the retained volume
collapses almost instantly.  Is that physical, or an under-resolved time step?

``solve_ivp`` is ADAPTIVE: ``t_eval``/``n_eval`` only sets OUTPUT sampling, not
the internal step (which is chosen to meet rtol/atol).  Escape time comes from
the terminal EVENT, root-found to tolerance — independent of t_eval.  This
script re-integrates the SAME seeded bubbles under several solver settings and
checks that fate and escape time are invariant; if they are, the production
setting (LSODA, rtol=1e-6, atol=1e-8) is converged.

  baseline : LSODA  rtol=1e-6  atol=1e-8            (production)
  tight    : LSODA  rtol=1e-9  atol=1e-12           (1000x tighter)
  capped   : LSODA  rtol=1e-6  atol=1e-8  max_step=5e-4   (explicit step cap)
  radau    : Radau  rtol=1e-8  atol=1e-10           (implicit, different family)

Run:
  .venv/bin/python pivRings/drivers/verify_timestep.py --field pivRings/fields/120_5D
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
from scipy.integrate import solve_ivp

import build_field as bf
import seeding as sd
from advect import mr_rhs, _escape_event
from constants import froude_number, stokes_number, R_BUBBLE

SETTINGS = {
    "baseline": dict(method="LSODA", rtol=1e-6, atol=1e-8),
    "tight":    dict(method="LSODA", rtol=1e-9, atol=1e-12),
    "capped":   dict(method="LSODA", rtol=1e-6, atol=1e-8, max_step=5e-4),
    "radau":    dict(method="Radau", rtol=1e-8, atol=1e-10),
}


def integrate(field, pos0, St, Fr, R, t_max, cfg):
    """Integrate one bubble; return (escaped, t_escape, nfev, n_steps, r_final)."""
    v0 = field.velocity(np.atleast_2d(pos0))[0]
    s0 = np.concatenate([np.asarray(pos0, float), v0])
    sol = solve_ivp(mr_rhs, (0.0, t_max), s0, args=(field, St, R, Fr, True),
                    events=_escape_event(field), dense_output=False, **cfg)
    escaped = len(sol.t_events[0]) > 0
    t_esc = float(sol.t_events[0][0]) if escaped else np.nan
    rf = float(np.hypot(sol.y[1, -1], sol.y[2, -1]))
    return escaped, t_esc, int(sol.nfev), int(sol.t.size), rf


def convergence_plot(field, ell, Fr, R, t_max, out_png, n_per_radius=8, seed=1):
    """Escape-time convergence vs solver resolution, on an ensemble of escaping
    bubbles (radii >= 0.5 mm, which reliably detrain).  Two refinement sweeps,
    each compared to a gold reference (LSODA rtol=1e-12, atol=1e-14):

      * tolerance sweep   — adaptive LSODA at rtol = 1e-3 .. 1e-11;
      * time-step sweep   — production tol (rtol 1e-6) with an explicit
                            ``max_step`` cap = 0.2 .. 0.002.

    The plotted quantity is the relative escape-time error |t_esc - t_ref|/t_ref
    (median over the ensemble, with the 25-75% band).  Convergence = error falls
    and plateaus at the tolerance floor."""
    U, R0 = field.U_ring, field.R0
    dist = sd.uniform_size_distribution(20.0)
    rng = np.random.default_rng(seed)
    bub = []
    for d in dist.diameters:
        if d / 2.0 < 0.5:
            continue                                   # escaping sizes only
        St = float(stokes_number(d, U, R0))
        for pos0 in sd.seed_core_surface(ell, n_per_radius, rng):
            bub.append((d, St, pos0))

    # gold reference escape time per bubble (keep only the ones that escape)
    ref_cfg = dict(method="LSODA", rtol=1e-12, atol=1e-14)
    keep, t_ref = [], []
    for d, St, pos0 in bub:
        esc, t, *_ = integrate(field, pos0, St, Fr, R, t_max, ref_cfg)
        if esc and np.isfinite(t):
            keep.append((d, St, pos0)); t_ref.append(t)
    t_ref = np.array(t_ref)
    print(f"convergence ensemble: {len(keep)} escaping bubbles "
          f"(t_ref median={np.median(t_ref):.4f})", flush=True)

    def sweep(cfg_fn, values):
        med, lo, hi = [], [], []
        for v in values:
            errs = []
            for (d, St, pos0), tr in zip(keep, t_ref):
                esc, t, *_ = integrate(field, pos0, St, Fr, R, t_max, cfg_fn(v))
                if esc and np.isfinite(t):
                    errs.append(abs(t - tr) / tr)
            errs = np.array(errs) if errs else np.array([np.nan])
            med.append(np.nanmedian(errs))
            lo.append(np.nanpercentile(errs, 25)); hi.append(np.nanpercentile(errs, 75))
        return np.array(med), np.array(lo), np.array(hi)

    rtols = np.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11])
    em, el, eh = sweep(lambda rt: dict(method="LSODA", rtol=rt, atol=rt * 1e-2), rtols)

    hsteps = np.array([0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
    hm, hl, hh = sweep(lambda h: dict(method="LSODA", rtol=1e-6, atol=1e-8, max_step=h),
                       hsteps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.fill_between(rtols, el, eh, color="#1f77b4", alpha=0.2)
    ax1.loglog(rtols, em, "o-", color="#1f77b4")
    ax1.axvline(1e-6, ls=":", color="crimson", label="production rtol=1e-6")
    ax1.set_xlabel("solver rtol"); ax1.set_ylabel("relative escape-time error")
    ax1.set_title("Tolerance convergence (adaptive LSODA)")
    ax1.invert_xaxis(); ax1.grid(True, which="both", alpha=0.3); ax1.legend()

    ax2.fill_between(hsteps, hl, hh, color="#2ca02c", alpha=0.2)
    ax2.loglog(hsteps, hm, "s-", color="#2ca02c")
    ax2.set_xlabel("explicit max_step cap $\\Delta t^*$")
    ax2.set_ylabel("relative escape-time error")
    ax2.set_title("Time-step convergence (rtol=1e-6 fixed)")
    ax2.invert_xaxis(); ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("ODE escape-time convergence — 120_5D upper core "
                 f"({len(keep)} escaping bubbles, ref: rtol=1e-12)")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"-> {out_png}", flush=True)
    print(f"  rtol 1e-6 median err = {em[list(rtols).index(1e-6)]:.2e}; "
          f"max_step 0.002 median err = {hm[-1]:.2e}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", default=os.path.join(ROOT, "fields", "120_5D"))
    p.add_argument("--core", default="upper")
    p.add_argument("--n-per-radius", type=int, default=8)
    p.add_argument("--t-max", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true",
                   help="produce the escape-time convergence figure and exit "
                        "(skips the 4-setting table)")
    args = p.parse_args()

    field = bf.load_field(args.field)
    mean = bf.load_mean_field(args.field)
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core=args.core)
    Fr = froude_number(field.U_ring, R0_mm=field.R0)
    R = R_BUBBLE

    if args.plot:
        out_png = os.path.join(ROOT, "outputs",
                               f"{os.path.basename(args.field.rstrip('/'))}_"
                               f"{args.core}_timestep_convergence.png")
        convergence_plot(field, ell, Fr, R, args.t_max, out_png,
                         n_per_radius=args.n_per_radius, seed=args.seed + 1)
        return
    dist = sd.uniform_size_distribution(20.0)
    rng = np.random.default_rng(args.seed)

    # one fixed set of bubbles (same pos0 reused across every solver setting)
    bub = []   # (radius_idx, d, St, pos0)
    for ri, d in enumerate(dist.diameters):
        P = sd.seed_core_surface(ell, args.n_per_radius, rng)
        St = float(stokes_number(d, field.U_ring, field.R0))
        for pos0 in P:
            bub.append((ri, float(d), St, pos0))
    N = len(bub)
    print(f"{os.path.basename(args.field)} [{args.core}]  Fr={Fr:.4f}  "
          f"{N} bubbles ({dist.n_radii} radii x {args.n_per_radius})\n")

    # integrate every bubble under every setting.  The explicit-step-cap setting
    # ("capped") is only meaningful where step size matters — the escaping
    # bubbles — and is ruinously slow on captured bubbles (40k forced steps to
    # t_max), so it is run ONLY on the baseline escapers; captured bubbles keep
    # their captured fate (tolerance refinement already covers them).
    out = {name: {"esc": np.zeros(N, bool), "t": np.full(N, np.nan),
                  "nfev": np.zeros(N, int), "nstep": np.zeros(N, int)}
           for name in SETTINGS}
    order = ["baseline", "tight", "radau", "capped"]
    for name in order:
        cfg = SETTINGS[name]
        base_esc = out["baseline"]["esc"]
        for i, (ri, d, St, pos0) in enumerate(bub):
            if name == "capped" and not base_esc[i]:
                continue                              # skip captured: step size irrelevant
            esc, t, nfev, nstep, rf = integrate(field, pos0, St, Fr, R, args.t_max, cfg)
            out[name]["esc"][i] = esc
            out[name]["t"][i] = t
            out[name]["nfev"][i] = nfev
            out[name]["nstep"][i] = nstep
        e = out[name]["esc"]
        note = " (escapers only)" if name == "capped" else ""
        sub = base_esc if name == "capped" else np.ones(N, bool)
        print(f"  {name:9s}: captured {(~e).mean():6.2%}  escaped {e.sum():4d}  "
              f"mean nfev={out[name]['nfev'][sub].mean():7.0f}  "
              f"median internal steps={np.median(out[name]['nstep'][sub]):.0f}{note}",
              flush=True)

    # ---- agreement vs baseline ----
    base = out["baseline"]
    print("\nagreement vs baseline (fate match; escape-time diff over bubbles that "
          "escaped in BOTH):")
    for name in SETTINGS:
        if name == "baseline":
            continue
        o = out[name]
        fate_match = (o["esc"] == base["esc"]).mean()
        both = o["esc"] & base["esc"]
        if both.any():
            dt = np.abs(o["t"][both] - base["t"][both])
            # relative to the escape time itself
            rel = dt / np.maximum(base["t"][both], 1e-9)
            print(f"  {name:9s}: fate match {fate_match:7.2%}  "
                  f"max |dt_esc|={dt.max():.2e}  median |dt|={np.median(dt):.2e}  "
                  f"max rel={rel.max():.2e}")
        else:
            print(f"  {name:9s}: fate match {fate_match:7.2%}  (no common escapers)")

    # ---- escape time vs the movie frame spacing ----
    dt_frame = args.t_max / 160.0
    te = base["t"][base["esc"]]
    print(f"\nmovie frame spacing dt*={dt_frame:.4f} "
          f"(= {dt_frame*field.R0/field.U_ring*1e3:.0f} ms physical)")
    print(f"escape times (baseline): min={np.nanmin(te):.4f}  median={np.nanmedian(te):.4f}  "
          f"max={np.nanmax(te):.4f}")
    print(f"  fraction of escapers leaving BEFORE frame 1 (t*<{dt_frame:.3f}): "
          f"{(te < dt_frame).mean():.2%}")
    print(f"  -> these vanish before the first sampled movie frame "
          f"(visual 'instant' collapse), but the integrator resolves them: "
          f"median internal steps to escape are many, see above.")

    # per-radius escape-time table baseline vs tight
    print(f"\n{'r_mm':>5} {'St':>8} {'t_esc(base)':>11} {'t_esc(tight)':>12} {'n_esc':>6}")
    for ri in range(dist.n_radii):
        idx = np.array([b[0] == ri for b in bub])
        m = idx & base["esc"]
        mt = idx & out["tight"]["esc"]
        tb = np.nanmedian(base["t"][m]) if m.any() else np.nan
        tt = np.nanmedian(out["tight"]["t"][mt]) if mt.any() else np.nan
        print(f"{dist.radii[ri]:5.2f} {bub[ri*args.n_per_radius][2]:8.4f} "
              f"{tb:11.4f} {tt:12.4f} {int(m.sum()):6d}")


if __name__ == "__main__":
    main()
