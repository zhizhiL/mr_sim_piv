#!/usr/bin/env python3
"""
test_frame_closure.py — does a single INSTANTANEOUS frame (instead of the time
average) recover a closed co-moving streamline at the thin-ring U_ring for the
two open cases (200_5D, 200_10D)?  (§7e remedy test.)

For each station: load + register the raw .dfi window, then for a sample of
individual registered frames build the co-moving field at the thin-ring U_ring
and at the closure-band U_ring, and report the trapped-tracer fraction.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(ROOT))
import numpy as np
import digiflow_io as dio, frame_transform as ft, averaging as avg
import build_field as bf, seeding as sd, ftle as F
from averaging import MeanField
from constants import R0

BASE = "/mnt/d/Users/zl483/highspeedcamera"
# label, folder, coord, thin-ring U, closure-band U
CASES = [("200_5D", "bonus_test_15", "coord_up", 109.7, 60.0),
         ("200_10D", "bonus_test_11", "april_bonus", 83.8, 48.0)]


def frame_mean(frames, k):
    return avg.smooth_field(MeanField(X=frames.X, Y=frames.Y, Ux=frames.u[k],
                                      Uy=frames.v[k], omega=frames.omega[k]), sigma=1.5)


def trapped(mean, U):
    field = bf.build_field(mean, U_ring=U, R0=R0)
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0)
    seeds = sd.seed_core_surface(ell, 200, np.random.default_rng(0))
    _, alive = F.fluid_flow_map(field, np.column_stack([seeds[:, 0], seeds[:, 2]]), T=12.0)
    return float(alive.mean())


def main():
    for label, folder, coord, U_thin, U_close in CASES:
        info = json.load(open(os.path.join(ROOT, "fields", label, "station.json")))
        start, stop = info["window"]
        frames = dio.load_piv(os.path.join(BASE, folder, "Camera_1"),
                              os.path.join(BASE, f"{coord}_mapping.csv"),
                              fps=500, start=start, stop=stop)
        uc = ft.estimate_Uc(frames, method="vorticity_centroid")
        frames = ft.register_frames(frames, uc.x_track)
        ks = np.linspace(0, frames.nframes - 1, 12).astype(int)
        tt_thin, tt_close = [], []
        for k in ks:
            m = frame_mean(frames, k)
            try:
                tt_thin.append(trapped(m, U_thin)); tt_close.append(trapped(m, U_close))
            except Exception:
                pass
        tt_thin, tt_close = np.array(tt_thin), np.array(tt_close)
        print(f"\n{label}: {len(ks)} single frames "
              f"(thin-ring U={U_thin}, closure-band U={U_close})", flush=True)
        print(f"  trapped @ thin-ring : median={np.median(tt_thin):.0%}  max={tt_thin.max():.0%}  "
              f"frames closing(>30%)={np.mean(tt_thin > 0.3):.0%}")
        print(f"  trapped @ closure U : median={np.median(tt_close):.0%}  max={tt_close.max():.0%}  "
              f"frames closing(>30%)={np.mean(tt_close > 0.3):.0%}")


if __name__ == "__main__":
    main()
