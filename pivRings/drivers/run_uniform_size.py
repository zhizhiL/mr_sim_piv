#!/usr/bin/env python3
"""
run_uniform_size.py — build-plan seeding deliverables for ONE station.

Implements the four build-plan changes:
  1. seeds on the CORE SURFACE with the core azimuth uniformly randomised
     (:func:`seeding.seed_core_surface`);
  2. a UNIFORM-IN-COUNT discrete size distribution (radii 0.06..0.90 mm by 0.04;
     equal physical count per radius; total volume = loading, l1=20 / l3=40 uL);
  3. deliverables: a movie, remaining-percentage per radius, and residual time;
  4. run for BOTH the upper and lower core.

For each (core, radius) it seeds ``--n-sim`` bubbles on the core surface, advects
them under Maxey-Riley (gravity on), and records fate + escape time.  Per-radius
remaining-% is the captured fraction (a probability, independent of count);
the uniform-count physical weight (count_per_radius x v_b) only sets the
aggregate retained-VOLUME bookkeeping that integrates to the loading.

Example (a good case):
  .venv/bin/python pivRings/drivers/run_uniform_size.py \
      --field pivRings/fields/120_5D --loading l1 --n-sim 120 --workers 16 --movie
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
import advect as ad
from constants import froude_number, stokes_number

OUT = os.path.join(ROOT, "outputs")
CORE_COL = {"upper": "#1f77b4", "lower": "#d62728"}


def run_core(fdir, field, mean, core, dist, Fr, n_sim, workers, t_max, rng,
             escape="fov", sep_poly=None):
    """Seed ``n_sim`` bubbles per radius on one core surface, advect, and return
    per-bubble records + the fitted ellipse.  ``escape`` selects the FOV-rectangle
    or the streamfunction-separatrix exit test (REPORT §7e)."""
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core=core)
    pos_list, d_list, ri_list = [], [], []
    for ri, d in enumerate(dist.diameters):
        P = sd.seed_core_surface(ell, n_sim, rng)
        pos_list.append(P)
        d_list.append(np.full(n_sim, d))
        ri_list.append(np.full(n_sim, ri))
    positions = np.vstack(pos_list)
    diameters = np.concatenate(d_list)
    radius_idx = np.concatenate(ri_list)
    stokes = stokes_number(diameters, field.U_ring, field.R0)

    res = ad.advect_bubbles(fdir, positions, diameters, stokes, Fr,
                            n_workers=workers, gravity=True, t_max=t_max,
                            escape=escape, sep_poly=sep_poly)
    escaped = np.array([r.escaped for r in res])
    t_esc = np.array([r.t_escape for r in res])
    face = np.array([r.exit_face for r in res], dtype=object).astype(str)
    return ell, dict(radius_idx=radius_idx, d=diameters, St=stokes,
                     escaped=escaped, t_escape=t_esc, exit_face=face)


def per_radius_table(dist, rec, t_max):
    """Per-radius remaining-% (captured fraction), buoyant-detrainment fraction
    and residual time (mean escape time; t_max for censored captured bubbles)."""
    n = dist.n_radii
    out = {k: np.full(n, np.nan) for k in
           ("remaining", "buoyant", "advective", "t_res_escaped", "t_res_all")}
    for ri in range(n):
        m = rec["radius_idx"] == ri
        if not m.any():
            continue
        esc = rec["escaped"][m]
        face = rec["exit_face"][m]
        te = rec["t_escape"][m]
        out["remaining"][ri] = float((~esc).mean())              # captured fraction
        out["buoyant"][ri] = float((face == "r_top").mean())
        out["advective"][ri] = float(np.isin(face, ["x_min", "x_max"]).mean())
        out["t_res_escaped"][ri] = float(np.nanmean(te[esc])) if esc.any() else np.nan
        # residence time treating captured bubbles as censored at t_max
        te_all = np.where(esc, te, t_max)
        out["t_res_all"][ri] = float(np.nanmean(te_all))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", required=True)
    p.add_argument("--loading", choices=["l1", "l3"], default="l1")
    p.add_argument("--cores", default="upper,lower",
                   help="comma list of cores to run (upper,lower)")
    p.add_argument("--n-sim", type=int, default=120,
                   help="simulated bubbles per radius per core")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--t-max", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--movie", action="store_true")
    p.add_argument("--escape", choices=["fov", "separatrix"], default="fov",
                   help="escape test: FOV rectangle (default) or streamfunction "
                        "separatrix (REPORT §7e, FOV-independent)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    label = os.path.basename(args.field.rstrip("/"))
    # separatrix runs go to a parallel outputs/separatrix_escape/ tree (keep FOV ones)
    out_dir = OUT if args.escape == "fov" else os.path.join(OUT, "separatrix_escape")
    base = args.out or os.path.join(out_dir, f"{label}_{args.loading}_uniform")
    os.makedirs(os.path.dirname(base) if os.path.dirname(base) else ".", exist_ok=True)

    field = bf.load_field(args.field)
    mean = bf.load_mean_field(args.field)
    if mean is None:
        raise SystemExit(f"No mean_field.pkl in {args.field}; re-run preprocess_station.py")
    Fr = froude_number(field.U_ring, R0_mm=field.R0)

    # one (x,r) atmosphere for the whole axisymmetric field — shared by both cores
    sep_poly = None
    if args.escape == "separatrix":
        import separatrix as sx
        e0 = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0, core="upper")
        sep = sx.compute_separatrix(field, core_xr=(e0.x_c / field.R0, e0.r_c / field.R0))
        if sep is None:
            raise SystemExit("no closed separatrix found; cannot use --escape separatrix")
        sep_poly = sep.poly
        print(f"separatrix escape: atmosphere area={sep.area():.2f}  "
              f"FOV-truncated={sep.fov_truncated}")
    dist = sd.uniform_size_distribution(sd.LOADING_UL[args.loading])
    tau_f = field.R0 / field.U_ring   # s, t = t* tau_f
    rng = np.random.default_rng(args.seed)
    cores = [c.strip() for c in args.cores.split(",") if c.strip()]

    print(f"{label}: U_ring={field.U_ring:.1f} mm/s  Fr={Fr:.3f}  tau_f={tau_f:.3f} s")
    print(f"uniform sizes: {dist.n_radii} radii {dist.radii[0]:.2f}..{dist.radii[-1]:.2f} mm, "
          f"loading={dist.loading_uL:.0f} uL, count/radius={dist.count_per_radius:.3f}")

    tables, ells, meta = {}, {}, {}
    rec_all = {k: [] for k in ("core", "radius_idx", "d", "St",
                               "escaped", "t_escape", "exit_face")}
    for core in cores:
        ell, rec = run_core(args.field, field, mean, core, dist, Fr,
                            args.n_sim, args.workers, args.t_max, rng,
                            escape=args.escape, sep_poly=sep_poly)
        tbl = per_radius_table(dist, rec, args.t_max)
        tables[core] = tbl
        ells[core] = ell
        # aggregate retained volume fraction (uniform-count weighting)
        w = dist.count_per_radius * dist.vol_each          # physical volume per radius
        vol_ret = float(np.nansum(tbl["remaining"] * w) / w.sum())
        meta[core] = {"a_eq_mm": ell.a_eq, "x_c_mm": ell.x_c, "r_c_mm": ell.r_c,
                      "retained_volume_fraction": vol_ret,
                      "overall_captured_fraction": float((~rec["escaped"]).mean())}
        print(f"  [{core}] captured={(~rec['escaped']).mean():.2%}  "
              f"retained-volume={vol_ret:.2%}  a_eq={ell.a_eq:.2f} mm")
        for k in ("radius_idx", "d", "St", "escaped", "t_escape", "exit_face"):
            rec_all[k].append(np.asarray(rec[k]))
        rec_all["core"].append(np.full(len(rec["d"]), core, dtype=object))

    # ---- save raw records + per-radius tables ----
    np.savez(base + "_records.npz",
             radii=dist.radii, diameters=dist.diameters, vol_each=dist.vol_each,
             count_per_radius=dist.count_per_radius,
             **{k: np.concatenate(v) for k, v in rec_all.items()})
    summary = {"label": label, "loading": args.loading, "U_ring": field.U_ring,
               "Fr": Fr, "tau_f_s": tau_f, "n_sim_per_radius": args.n_sim,
               "radii_mm": dist.radii.tolist(),
               "cores": {c: {**meta[c],
                             "remaining_fraction": tables[c]["remaining"].tolist(),
                             "buoyant_fraction": tables[c]["buoyant"].tolist(),
                             "t_res_escaped_star": tables[c]["t_res_escaped"].tolist(),
                             "t_res_all_star": tables[c]["t_res_all"].tolist()}
                         for c in cores}}
    with open(base + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    _plots(base, label, dist, tables, tau_f, args.loading, args.escape)

    movie_paths = []
    if args.movie:
        from make_movie import render_movie
        n_mov = max(200 // dist.n_radii, 1)              # bubbles per radius, per core
        for core in cores:                               # ONE movie per core (separate)
            pos_l, d_l = [], []
            for d in dist.diameters:
                pos_l.append(sd.seed_core_surface(ells[core], n_mov, rng))
                d_l.append(np.full(n_mov, d))
            positions = np.vstack(pos_l)
            diameters = np.concatenate(d_l)
            stk = stokes_number(diameters, field.U_ring, field.R0)
            mp = f"{base}_{core}_movie.mp4"
            render_movie(field, [ells[core]], positions, diameters, stk, Fr, mp,
                         t_max=args.t_max, n_frames=160,
                         title_extra=f"{label}  [{args.loading} uniform, {core} core]")
            movie_paths.append(mp)

    print(f"-> {base}_records.npz, {base}_summary.json, "
          f"{base}_remaining_vs_radius.png, {base}_residual_time.png, "
          f"{base}_volume_vs_time.png" + "".join(", " + m for m in movie_paths))


def _plots(base, label, dist, tables, tau_f, loading, escape="fov"):
    radii = dist.radii
    tag = "FOV escape" if escape == "fov" else "separatrix escape"

    # Fig 1: remaining percentage vs radius (per core) + buoyant detrainment
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for core, tbl in tables.items():
        ax.plot(radii, 100 * tbl["remaining"], "o-", color=CORE_COL[core],
                lw=1.8, label=f"{core} core (captured)")
        ax.plot(radii, 100 * tbl["buoyant"], "s--", color=CORE_COL[core],
                lw=1.1, alpha=0.55, label=f"{core} buoyant detrainment")
    ax.set_xlabel("bubble radius (mm)")
    ax.set_ylabel("remaining percentage  (% captured at $t^*=20$)")
    ax.set_title(f"Remaining percentage per radius — {label} ({loading}, {tag})")
    ax.set_ylim(-2, 102); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(base + "_remaining_vs_radius.png", dpi=140)
    plt.close(fig)

    # Fig 2: residual time vs radius (physical seconds; censored at t_max)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for core, tbl in tables.items():
        ax.plot(radii, tbl["t_res_all"] * tau_f, "o-", color=CORE_COL[core],
                lw=1.8, label=f"{core} core (all, censored)")
        ax.plot(radii, tbl["t_res_escaped"] * tau_f, "^--", color=CORE_COL[core],
                lw=1.1, alpha=0.55, label=f"{core} escaped only")
    ax.set_xlabel("bubble radius (mm)")
    ax.set_ylabel("residual / residence time (s)")
    ax.set_title(f"Residual time per radius — {label} ({loading}, {tag})\n"
                 "(solid: captured censored at $t_{max}$; dashed: escaped subset)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(base + "_residual_time.png", dpi=140)
    plt.close(fig)

    # Fig 3: aggregate remaining in-ring VOLUME fraction vs time (uniform-count)
    recs = np.load(base + "_records.npz", allow_pickle=True)
    tgrid = np.linspace(0, 20, 160)
    core_arr = recs["core"].astype(str)
    ri = recs["radius_idx"]; esc = recs["escaped"]; te = recs["t_escape"]
    w = dist.count_per_radius * dist.vol_each              # volume weight per radius
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for core in tables:
        sel = core_arr == core
        wb = w[ri[sel]]                                    # per-bubble volume weight
        present = (~esc[sel])[None, :] | (te[sel][None, :] > tgrid[:, None])
        remain = (present * wb).sum(axis=1) / wb.sum()
        ax.plot(tgrid * tau_f, remain, color=CORE_COL[core], lw=1.8,
                label=f"{core} core")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("remaining in-ring volume fraction")
    ax.set_title(f"Volume retention vs time — {label} ({loading}, uniform count, {tag})")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(base + "_volume_vs_time.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
