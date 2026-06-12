#!/usr/bin/env python3
"""
build_single_frame_field.py — for a station whose TIME-AVERAGE has no closed
co-moving streamline (the averaging smears the meandering vortex, §7e), pick a
representative INSTANTANEOUS frame that does close, recenter it (registration),
and cache it as if it were the time-averaged mean so the standard pipeline
(run_uniform_size, etc.) runs on a field that closes at the physical U_ring.

Representative frame = the closing frame whose trapped-tracer fraction is nearest
the median (a typical frame, not an outlier).  Saved to fields/<label>_sf/.

  .venv/bin/python pivRings/drivers/build_single_frame_field.py 200_5D bonus_test_15 coord_up 109.7
"""
from __future__ import annotations
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(ROOT))
import numpy as np
import digiflow_io as dio, frame_transform as ft, averaging as avg
import build_field as bf, seeding as sd, ftle as F
from averaging import MeanField
from constants import R0

BASE = "/mnt/d/Users/zl483/highspeedcamera"


def frame_mean(frames, k):
    return avg.smooth_field(MeanField(X=frames.X, Y=frames.Y, Ux=frames.u[k],
                                      Uy=frames.v[k], omega=frames.omega[k]), sigma=1.5)


def trapped(mean, U):
    field = bf.build_field(mean, U_ring=U, R0=R0)
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0)
    seeds = sd.seed_core_surface(ell, 150, np.random.default_rng(0))
    _, alive = F.fluid_flow_map(field, np.column_stack([seeds[:, 0], seeds[:, 2]]), T=12.0)
    return float(alive.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label"); p.add_argument("folder"); p.add_argument("coord")
    p.add_argument("u_ring", type=float, help="U_ring magnitude to build at (mm/s)")
    p.add_argument("--stride", type=int, default=3, help="frame subsample for selection")
    args = p.parse_args()

    info = json.load(open(os.path.join(ROOT, "fields", args.label, "station.json")))
    start, stop = info["window"]
    sign = float(np.sign(json.load(open(os.path.join(ROOT, "fields", args.label, "meta.json"))).get("U_ring_signed", 1.0))) or 1.0
    frames = dio.load_piv(os.path.join(BASE, args.folder, "Camera_1"),
                          os.path.join(BASE, f"{args.coord}_mapping.csv"),
                          fps=500, start=start, stop=stop)
    uc = ft.estimate_Uc(frames, method="vorticity_centroid")
    frames = ft.register_frames(frames, uc.x_track)

    ks = np.arange(0, frames.nframes, args.stride)
    tt = np.array([trapped(frame_mean(frames, k), sign * args.u_ring) for k in ks])
    closing = tt > 0.3
    if not closing.any():
        raise SystemExit(f"{args.label}: no closing frame at U={args.u_ring}")
    kc = ks[closing]; tc = tt[closing]
    # representative = closest to median closure; tie-break to the temporal centre
    # of the window (a quasi-steady frame, not an edge/transient one)
    center = frames.nframes / 2.0
    order = np.lexsort((np.abs(kc - center), np.abs(tc - np.median(tc))))
    k_rep = int(kc[order[0]])
    print(f"{args.label}: {closing.mean():.0%} of sampled frames close at U={args.u_ring}; "
          f"representative frame k={k_rep} (trapped={tt[list(ks).index(k_rep)]:.0%}, "
          f"median={np.median(tc):.0%})", flush=True)

    mrep = frame_mean(frames, k_rep)
    field = bf.build_field(mrep, U_ring=sign * args.u_ring, R0=R0)
    out = os.path.join(ROOT, "fields", f"{args.label}_sf")
    bf.save_field(field, out, mean=mrep)
    with open(os.path.join(out, "station.json"), "w") as fh:
        json.dump({**info, "label": f"{args.label}_sf", "source": "single_frame",
                   "rep_frame": k_rep, "U_ring_built": args.u_ring}, fh, indent=2)
    print(f"-> {out} (built at U_ring={field.U_ring:.1f})", flush=True)


if __name__ == "__main__":
    main()
