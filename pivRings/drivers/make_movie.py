#!/usr/bin/env python3
"""
make_movie.py — 3D animation of bubble advection in a cached ring field.

Renders the vortex-ring core as a translucent shaded donut (torus) and the
seeded bubbles advecting through it over (dimensionless) time, coloured by fate
(captured vs escaped).  Writes an .mp4 (bundled ffmpeg via imageio-ffmpeg).

Example (a good case):
  .venv/bin/python pivRings/drivers/make_movie.py \
      --field pivRings/fields/120_10D \
      --escape-csv pivRings/data/escape/bubbles_120_l1.csv --station-D 400
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
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import build_field as bf
import seeding as sd
import advect as ad
from constants import R0, froude_number


def torus(x_core, r_major, a_tube, nu=60, nv=24):
    """Torus surface with axis along x, centred at axial position x_core."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, 2 * np.pi, nv)
    U, V = np.meshgrid(u, v)
    Y = (r_major + a_tube * np.cos(V)) * np.cos(U)
    Z = (r_major + a_tube * np.cos(V)) * np.sin(U)
    X = x_core + a_tube * np.sin(V)
    return X, Y, Z


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", default=os.path.join(ROOT, "fields", "120_10D"))
    p.add_argument("--escape-csv", default=os.path.join(ROOT, "data", "escape", "bubbles_120_l1.csv"))
    p.add_argument("--station-D", type=float, default=400.0)
    p.add_argument("--n-seed", type=int, default=20)
    p.add_argument("--n-phi", type=int, default=12)
    p.add_argument("--t-max", type=float, default=20.0)
    p.add_argument("--n-frames", type=int, default=160)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "bubble_advection.mp4"))
    args = p.parse_args()

    field = bf.load_field(args.field)
    mean = bf.load_mean_field(args.field)
    ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=field.R0)
    positions = sd.seed_positions(ell, n=args.n_seed, n_phi=args.n_phi)
    stk = sd.sample_stokes(positions.shape[0], args.escape_csv, u_ring_mms=field.U_ring,
                           R0_mm=field.R0, station_D=args.station_D)
    Fr = froude_number(field.U_ring, R0_mm=field.R0)

    print(f"advecting {positions.shape[0]} bubbles, t_max={args.t_max} "
          f"(= {args.t_max * field.R0 / field.U_ring:.2f} s physical), Fr={Fr:.3f} ...")
    t_eval, traj, escaped = ad.advect_trajectories(
        field, positions, stk.d, stk.St, Fr, t_max=args.t_max, n_eval=args.n_frames)
    tau_f = field.R0 / field.U_ring   # s
    print(f"  {escaped.sum()}/{len(escaped)} escaped; rendering {args.n_frames} frames ...")

    xmin, xmax, _, rmax = field.bounds
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # translucent shaded donut = vortex-ring core
    Xt, Yt, Zt = torus(ell.x_c / field.R0, ell.r_c / field.R0, max(ell.a_eq, 2.0) / field.R0)
    ax.plot_surface(Xt, Yt, Zt, color="#7fb3d5", alpha=0.22, linewidth=0,
                    antialiased=True, shade=True, zorder=1)

    cap, esc = ~escaped, escaped
    s_cap = ax.scatter([], [], [], s=14, c="#1f4e79", depthshade=True, label="captured")
    s_esc = ax.scatter([], [], [], s=14, c="#e07b00", depthshade=True, label="escaping")

    ax.set_xlim(xmin, xmax); ax.set_ylim(-rmax, rmax); ax.set_zlim(-rmax, rmax)
    ax.set_xlabel("x* (axis)"); ax.set_ylabel("y*"); ax.set_zlabel("z* (gravity)")
    ax.legend(loc="upper right")
    try:
        ax.set_box_aspect((xmax - xmin, 2 * rmax, 2 * rmax))
    except Exception:
        pass

    def update(k):
        for sc, m in ((s_cap, cap), (s_esc, esc)):
            P = traj[m, k, :]
            P = P[~np.isnan(P[:, 0])]
            sc._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        ax.view_init(elev=22, azim=-60 + 30 * k / len(t_eval))  # slow orbit
        ax.set_title(f"Bubble advection in vortex ring  "
                     f"(t* = {t_eval[k]:5.2f},  t = {t_eval[k] * tau_f:4.2f} s)\n"
                     f"{os.path.basename(args.field)}  U_ring={field.U_ring:.0f} mm/s  "
                     f"Fr={Fr:.2f}  |  donut = ring core")
        return s_cap, s_esc

    anim = FuncAnimation(fig, update, frames=len(t_eval), interval=1000 / args.fps, blit=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    plt.close(fig)
    print(f"-> {args.out}  ({args.n_frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
