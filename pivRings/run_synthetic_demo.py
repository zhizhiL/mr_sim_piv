#!/usr/bin/env python3
"""
run_synthetic_demo.py — end-to-end protocol on a SYNTHETIC vortex-ring field.

Exercises every stage of build_plan.md (A -> G) without the experimental
``.dfi`` data, so the whole pipeline can be reviewed quickly:

    synthetic field
        -> estimate U_c (Stage B)        -> to co-moving + residual QC
        -> time-average + smooth (C)
        -> scale-separation report
        -> build & cache 3D field (D)    -> sanity streamlines (M1)
        -> fit core ellipse + seed (E)   -> log-normal St from escape csv (M2)
        -> Maxey-Riley advection (F)     -> escape diagnostic (M3)
        -> 3-station comparison (M4)

Run:  python pivRings/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.dirname(HERE))   # repo root: digiflowio.py, vortex_core.py

import numpy as np

import synthetic
import frame_transform as ft
import averaging as avg
import build_field as bf
import seeding as sd
import advect as ad
import diagnostics as dg

OUT = os.path.join(HERE, "outputs")
FIELDS = os.path.join(HERE, "fields")
ESCAPE_CSV = os.path.join(HERE, "data", "escape", "bubbles_120_l1.csv")
os.makedirs(OUT, exist_ok=True)


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def preprocess(frames, station, smooth_sigma=1.5):
    """Stages B-D for one station; returns (Field3D dir, field, mean, reports)."""
    uc = ft.estimate_Uc(frames, method="core_linear_fit")
    co = ft.to_comoving(frames, uc.U_c)
    qc = ft.residual_unsteadiness(co)

    mean = avg.time_average(co)
    mean = avg.smooth_field(mean, sigma=smooth_sigma)
    gammas = [g for (_, g) in frames.meta["truth"]["core_positions"]]
    ssr = avg.scale_separation_report(frames, uc.U_c, R0=20.0, per_frame_gamma=gammas)

    field = bf.build_field(mean)
    fdir = os.path.join(FIELDS, station)
    bf.save_field(field, fdir, mean=mean)
    return fdir, field, mean, {"uc": uc, "qc": qc, "ssr": ssr}


def main():
    banner("STAGE A  —  synthetic vortex-ring field (3 stations)")
    stations = synthetic.make_three_stations(n_frames=24, fps=200.0,
                                             U_c=100.0, noise=2.0, seed=1)
    for k, v in stations.items():
        tr = v.meta["truth"]
        print(f"  {k}: grid {v.u.shape[1]}x{v.u.shape[2]}, {v.nframes} frames, "
              f"U_c(truth)={tr['U_c']} mm/s, a_core(truth)={tr['a_core']:.2f} mm")

    reports = {}
    primary = "station_1"
    primary_field = primary_dir = primary_mean = None

    for station, frames in stations.items():
        banner(f"STAGES B-D  —  {station}")
        fdir, field, mean, rep = preprocess(frames, station)
        uc, qc, ssr = rep["uc"], rep["qc"], rep["ssr"]
        print(f"  U_c estimated = {uc.U_c:7.2f} mm/s   (truth 100.00, "
              f"fit residual {uc.residual_rms:.3f} mm)")
        print(f"  residual unsteadiness (co-moving) = {qc['relative_unsteadiness']:.3f}")
        print(f"  scale separation: {ssr['turnovers_per_window']:.2f} turnovers/window, "
              f"Gamma drift = {ssr.get('gamma_fractional_change', float('nan')):.3f}")
        print(f"  cached -> {os.path.relpath(fdir, HERE)}")

        if station == primary:
            primary_field, primary_dir, primary_mean = field, fdir, mean
            dg.plot_field_sanity(field, os.path.join(OUT, "M1_streamlines.png"))
            print(f"  M1 sanity plot -> outputs/M1_streamlines.png")

    banner("STAGE E  —  seeding from ellipse fit + log-normal St (M2)")
    ell = sd.fit_core_ellipse(primary_mean, y_axis=primary_field.y_axis)
    print(f"  fitted core: centre (x={ell.x_c:.2f}, r={ell.r_c:.2f}) mm, "
          f"a_eq={ell.a_eq:.2f} mm  (truth a_core=4.00, R0=20.00)")

    positions = sd.seed_positions(ell, n=12, n_phi=6, measure="arclength")
    print(f"  seeded {positions.shape[0]} bubbles "
          f"(12 meridional x 6 azimuthal, arclength measure)")
    dg.plot_seeds(primary_field, positions, os.path.join(OUT, "M2_seeds.png"))

    # ring velocity scale from the meridional field (peak co-moving speed)
    Xg, Rg = np.meshgrid(primary_field.x_axis, primary_field.r_axis)
    speed = np.hypot(primary_field.sp_Ux.ev(Rg.ravel(), Xg.ravel()),
                     primary_field.sp_Ur.ev(Rg.ravel(), Xg.ravel()))
    u_ring = float(np.nanmax(speed))
    print(f"  ring velocity scale U_ring = {u_ring:.1f} mm/s")

    stk = sd.sample_stokes(positions.shape[0], ESCAPE_CSV, u_ring_mms=u_ring)
    print(f"  log-normal escape sizes: mu_log={stk.mu_log:.3f}, "
          f"sigma_log={stk.sigma_log:.3f}, d_max={stk.d_max:.3f} mm")
    print(f"  St range = [{stk.St.min():.2e}, {stk.St.max():.2e}] "
          f"(median {np.median(stk.St):.2e})")

    banner("STAGE F-G  —  advection + escape diagnostic (M3)")
    # (a) faithful run: the measured escape-size population (near-tracer St)
    results = ad.advect_bubbles(
        primary_dir, positions, stk.d, stokes=stk.St,
        n_workers=4, t_max=2.0, gravity=True, max_step=0.05)
    stats = dg.escape_statistics(results)
    print(f"  faithful sizes (d~{stk.d.min():.2f}-{stk.d.max():.2f} mm): "
          f"{stats['n_escaped']}/{stats['n']} escaped "
          f"(fraction {stats['escape_fraction']:.3f})")
    print("  -> hydrogen microbubbles at these sizes are near-tracers (captured);")
    print("     escape activates at larger St (see the size sweep below).")
    reports[primary] = stats

    # (b) escape vs size sweep: the headline capture/escape threshold
    print("\n  size sweep (escape fraction vs diameter):")
    d_sweep = np.logspace(np.log10(0.5), np.log10(80.0), 10)
    fracs, pooled = [], []
    for d in d_sweep:
        res = ad.advect_bubbles(primary_dir, positions,
                                np.full(len(positions), d),
                                n_workers=4, t_max=1.5, gravity=True, max_step=0.05)
        s = dg.escape_statistics(res)
        fracs.append(s["escape_fraction"])
        pooled.extend(res)
        print(f"    d={d:6.2f} mm  escape_fraction={s['escape_fraction']:.2f}")
    thr = next((d for d, f in zip(d_sweep, fracs) if f > 0.5), float("nan"))
    dg.plot_escape_vs_size(d_sweep, fracs, os.path.join(OUT, "M3_escape_vs_size.png"),
                           threshold=thr)
    dg.plot_escape_distribution(pooled, ESCAPE_CSV,
                                os.path.join(OUT, "M3_escape_sizes.png"))
    print(f"  capture/escape threshold ~ {thr:.1f} mm (synthetic field)")

    banner("STAGE G  —  3-station quasi-steady comparison (M4)")
    for station in ("station_2", "station_3"):
        fdir = os.path.join(FIELDS, station)
        # compare at a fixed diameter near the threshold so differences show
        res = ad.advect_bubbles(fdir, positions, np.full(len(positions), 20.0),
                                n_workers=4, t_max=2.0, gravity=True, max_step=0.05)
        reports[station] = dg.escape_statistics(res)
    # re-evaluate station_1 at the same probe size for a fair comparison
    res1 = ad.advect_bubbles(primary_dir, positions, np.full(len(positions), 20.0),
                             n_workers=4, t_max=2.0, gravity=True, max_step=0.05)
    reports[primary] = dg.escape_statistics(res1)
    cmp = dg.station_comparison(reports)
    print(f"  escape fraction per station: {cmp['per_station_escape_fraction']}")
    print(f"  spread (empirical quasi-steady bound): {cmp['spread']:.3f}")

    banner("DONE — outputs/: M1_streamlines.png, M2_seeds.png, "
           "M3_escape_vs_size.png, M3_escape_sizes.png")


if __name__ == "__main__":
    main()
