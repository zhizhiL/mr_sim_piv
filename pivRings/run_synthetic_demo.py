#!/usr/bin/env python3
"""
run_synthetic_demo.py — end-to-end protocol on a SYNTHETIC vortex-ring field.

Exercises every stage of build_plan.md (A->G) with the dimensionless seam from
dimension_conversion.md, without the experimental .dfi data:

    synthetic field (mm, mm/s)
        -> estimate U_ring (Stage B)          [vorticity-weighted centroid fit]
        -> time-average + smooth (C)
        -> nondimensionalize + build 3D field (D)   [+ §6 assertions, M1 plot]
        -> fit core ellipse + seed (E)        -> density-free St (M2)
        -> Maxey-Riley advection (F)          -> escape vs size (M3)
        -> 3-station comparison (M4)

Run:  .venv/bin/python pivRings/run_synthetic_demo.py
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
import nondimensional as nd
import seeding as sd
import advect as ad
import diagnostics as dg
from constants import R0, R_BUBBLE, froude_number

OUT = os.path.join(HERE, "outputs")
FIELDS = os.path.join(HERE, "fields")
ESCAPE_CSV = os.path.join(HERE, "data", "escape", "bubbles_120_l1.csv")
os.makedirs(OUT, exist_ok=True)


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def preprocess(frames, station, smooth_sigma=1.5):
    uc = ft.estimate_Uc(frames, method="vorticity_centroid")
    co = ft.to_comoving(frames, uc.U_c)
    qc = ft.residual_unsteadiness(co)
    mean = avg.smooth_field(avg.time_average(frames), sigma=smooth_sigma)  # lab frame
    field = bf.build_field(mean, U_ring=uc.U_c, R0=R0)
    fdir = os.path.join(FIELDS, station)
    bf.save_field(field, fdir, mean=mean)
    gammas = [g for (_, g) in frames.meta["truth"]["core_positions"]]
    ssr = avg.scale_separation_report(frames, uc.U_c, R0=R0, per_frame_gamma=gammas)
    return fdir, field, mean, {"uc": uc, "qc": qc, "ssr": ssr}


def main():
    banner("STAGE A  —  synthetic vortex-ring field (3 stations)")
    stations = synthetic.make_three_stations(n_frames=24, fps=200.0, U_c=200.0,
                                             noise=2.0, seed=1)
    for k, v in stations.items():
        tr = v.meta["truth"]
        print(f"  {k}: grid {v.u.shape[1]}x{v.u.shape[2]}, {v.nframes} frames, "
              f"U_c(truth)={tr['U_c']} mm/s, a_core(truth)={tr['a_core']:.2f} mm")

    reports = {}
    primary = "station_1"
    pf = pdir = pmean = None
    U_ring_primary = None

    for station, frames in stations.items():
        banner(f"STAGES B-D  —  {station}")
        fdir, field, mean, rep = preprocess(frames, station)
        uc, qc, ssr = rep["uc"], rep["qc"], rep["ssr"]
        print(f"  U_ring = {uc.U_c:7.2f} mm/s  ({uc.method}, residual {uc.residual_rms:.3f} mm)")
        print(f"  residual unsteadiness = {qc['relative_unsteadiness']:.3f}")
        print(f"  scale sep: {ssr['turnovers_per_window']:.2f} turnovers/window, "
              f"Gamma drift = {ssr.get('gamma_fractional_change', float('nan')):.3f}")

        # ---- §6 assertions at the dimensionless boundary ----
        Xg, Rg = np.meshgrid(field.x_axis, field.r_axis)
        Ux_star = field.sp_Ux.ev(Xg.ravel(), Rg.ravel())
        o1 = nd.assert_field_O1(Ux_star)
        wm, wp = nd.assert_terminal_velocity_consistency(
            np.array([0.5, 1.0]), uc.U_c, R0)
        print(f"  [assert] |Ux*| p99 = {o1:.2f} (<5 OK); "
              f"St/Fr^2 == v_t/U_ring OK ({wm[0]:.3e}=={wp[0]:.3e})")
        print(f"  cached -> {os.path.relpath(fdir, HERE)}  (solver-format + field.pkl)")

        if station == primary:
            pf, pdir, pmean, U_ring_primary = field, fdir, mean, uc.U_c
            dg.plot_field_sanity(field, os.path.join(OUT, "M1_streamlines.png"))
            print("  M1 sanity plot -> outputs/M1_streamlines.png")

    Fr = froude_number(U_ring_primary, R0_mm=R0)
    print(f"\n  per-ring Fr = {Fr:.3f}  (U_ring={U_ring_primary:.1f} mm/s, R={R_BUBBLE:.3f})")

    banner("STAGE E  —  seeding from ellipse fit + density-free St (M2)")
    ell = sd.fit_core_ellipse(pmean, y_axis=pf.y_axis_mm, R0=R0)
    print(f"  fitted core: centre (x={ell.x_c:.2f}, r={ell.r_c:.2f}) mm, "
          f"a_eq={ell.a_eq:.2f} mm  -> dimensionless centre "
          f"(x*={ell.x_c/R0:.2f}, r*={ell.r_c/R0:.2f})")
    positions = sd.seed_positions(ell, n=16, n_phi=8, measure="arclength")
    print(f"  seeded {positions.shape[0]} bubbles (16 x 8, arclength)")
    dg.plot_seeds(pf, positions, os.path.join(OUT, "M2_seeds.png"))

    stk = sd.sample_stokes(positions.shape[0], ESCAPE_CSV, u_ring_mms=U_ring_primary, R0_mm=R0)
    print(f"  escape sizes: log-normal mu={stk.mu_log:.3f}, sigma={stk.sigma_log:.3f}, "
          f"d_max={stk.d_max:.3f} mm")
    print(f"  density-free St range = [{stk.St.min():.3f}, {stk.St.max():.3f}] "
          f"(median {np.median(stk.St):.3f})  <- O(0.1-1), not stiff")

    banner("STAGE F-G  —  advection + escape diagnostic (M3)")
    results = ad.advect_bubbles(pdir, positions, stk.d, stk.St, Fr,
                                n_workers=8, gravity=True, t_max=20.0)
    stats = dg.escape_statistics(results)
    print(f"  measured-size population (d~{stk.d.min():.2f}-{stk.d.max():.2f} mm): "
          f"{stats['n_escaped']}/{stats['n']} escaped (fraction {stats['escape_fraction']:.3f})")

    print("\n  size sweep (escape fraction vs diameter):")
    d_sweep = np.logspace(np.log10(0.05), np.log10(3.0), 12)
    fracs, pooled = [], []
    for d in d_sweep:
        St_d = sd.stokes_number(d, U_ring_primary, R0)
        res = ad.advect_bubbles(pdir, positions, np.full(len(positions), d),
                                np.full(len(positions), St_d), Fr,
                                n_workers=8, gravity=True, t_max=20.0)
        s = dg.escape_statistics(res)
        fracs.append(s["escape_fraction"]); pooled.extend(res)
        print(f"    d={d:6.3f} mm  St={St_d:5.3f}  escape_fraction={s['escape_fraction']:.2f}")
    thr = next((d for d, f in zip(d_sweep, fracs) if f > 0.9), float("nan"))
    dg.plot_escape_vs_size(d_sweep, fracs, os.path.join(OUT, "M3_escape_vs_size.png"), threshold=thr)
    dg.plot_escape_distribution(pooled, ESCAPE_CSV, os.path.join(OUT, "M3_escape_sizes.png"))
    print(f"  capture/escape threshold ~ {thr:.2f} mm (synthetic field)")

    banner("STAGE G  —  3-station quasi-steady comparison (M4)")
    probe_d = 0.2   # threshold-band size so station differences can show
    for station in ("station_1", "station_2", "station_3"):
        fdir = os.path.join(FIELDS, station)
        f = bf.load_field(fdir)
        Fr_s = froude_number(f.U_ring, R0_mm=R0)
        St_s = sd.stokes_number(probe_d, f.U_ring, R0)
        res = ad.advect_bubbles(fdir, positions, np.full(len(positions), probe_d),
                                np.full(len(positions), St_s), Fr_s,
                                n_workers=8, gravity=True, t_max=20.0)
        reports[station] = dg.escape_statistics(res)
    cmp = dg.station_comparison(reports)
    print(f"  escape fraction per station (probe d={probe_d} mm): "
          f"{ {k: round(v,3) for k,v in cmp['per_station_escape_fraction'].items()} }")
    print(f"  spread (empirical quasi-steady bound): {cmp['spread']:.3f}")

    banner("DONE — outputs/: M1_streamlines.png, M2_seeds.png, "
           "M3_escape_vs_size.png, M3_escape_sizes.png")


if __name__ == "__main__":
    main()
