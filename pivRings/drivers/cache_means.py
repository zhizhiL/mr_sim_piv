#!/usr/bin/env python3
"""
cache_means.py — load + register + time-average each station ONCE, cache the
registered meridional mean (lab frame) so all downstream U_ring / field
experiments avoid re-reading the .dfi stacks from /mnt/d.

Also records, per station: the signed centroid speed, the signed lab-frame axial
velocity at the vortex core (Ux@core ~ ring propagation speed, the physical
U_ring), and the window used.  Writes fields/<label>/mean_reg.pkl and
fields/<label>/station.json.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(ROOT))

import numpy as np
from scipy.interpolate import RectBivariateSpline

import digiflow_io as dio
import frame_transform as ft
import averaging as avg
from vortex_core import find_vortex_cores_iterative

BASE = "/mnt/d/Users/zl483/highspeedcamera"
FIELDS = os.path.join(ROOT, "fields")

# label, folder, coord, piston, D, window (None -> autodetect)
STATIONS = [
    ("120_5D",  "bonus_test_17", "coord_up",        120,  5, None),
    ("120_10D", "bonus_test_01", "april_vorticity", 120, 10, None),
    ("120_15D", "bonus_test_27", "coord_down",      120, 15, None),
    ("200_5D",  "bonus_test_15", "coord_up",        200,  5, None),
    ("200_10D", "bonus_test_11", "april_bonus",     200, 10, None),
    ("200_15D", "bonus_test_34", "coord_down",      200, 15, (180, 540)),
]


def signed_ux_at_core(mean):
    """Signed lab-frame axial velocity averaged over the two vortex cores."""
    cores, _, _ = find_vortex_cores_iterative(mean.X, mean.Y, mean.omega,
                                              fit_radius=8.0, verbose=False)
    (xp, yp, _), (xn, yn, _) = cores
    x = mean.X[0, :]
    y = mean.Y[:, 0]
    sp = RectBivariateSpline(y, x, mean.Ux)
    return float(0.5 * (sp(yp, xp)[0, 0] + sp(yn, xn)[0, 0]))


def main():
    for label, folder, coord, piston, nD, win in STATIONS:
        sdir = os.path.join(BASE, folder, "Camera_1")
        cf = os.path.join(BASE, f"{coord}_mapping.csv")
        try:
            start, stop = win if win else dio.autodetect_window(sdir, cf, stride=4)
            frames = dio.load_piv(sdir, cf, fps=500, start=start, stop=stop)
            uc = ft.estimate_Uc(frames, method="vorticity_centroid")
            slope = float(np.polyfit(uc.t[np.isfinite(uc.x_track)],
                                     uc.x_track[np.isfinite(uc.x_track)], 1)[0])
            frames = ft.register_frames(frames, uc.x_track)
            mean = avg.smooth_field(avg.time_average(frames), sigma=1.5)
            ux_core = signed_ux_at_core(mean)

            outdir = os.path.join(FIELDS, label)
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "mean_reg.pkl"), "wb") as fh:
                pickle.dump(mean, fh)
            info = {"label": label, "folder": folder, "coord": coord,
                    "piston": piston, "D": nD, "window": [start, stop],
                    "centroid_signed": slope, "ux_core_signed": ux_core,
                    "n_frames": frames.nframes}
            with open(os.path.join(outdir, "station.json"), "w") as fh:
                json.dump(info, fh, indent=2)
            print(f"{label:9s} window=[{start},{stop}] n={frames.nframes} "
                  f"centroid={slope:+.1f}  Ux@core={ux_core:+.1f} mm/s", flush=True)
        except Exception as e:
            print(f"{label:9s} FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
