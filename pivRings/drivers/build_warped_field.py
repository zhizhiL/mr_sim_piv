#!/usr/bin/env python3
"""
build_warped_field.py — radially re-position the ring core to a target radius R
WITHOUT changing its size a_eq (§7f geometry fix for 120_15D, whose detected
R_ring=15 mm drops unphysically below its upstream ~24 mm while a_eq is in-trend).

A uniform magnification would scale R and a_eq together; instead we warp the
radial coordinate  rho(r) = r + Delta * S(r),  with S a smooth sigmoid that has
saturated (S'~0, drho/dr~1) by the core radius, so the core is translated outward
by Delta = R_target - R_core while its LOCAL scale (hence a_eq) and the streamline
shape near the ring are preserved.  The inner region (axis->core) absorbs the
stretch.  Velocities are carried with the grid (shape-preserving for the figure).

  .venv/bin/python pivRings/drivers/build_warped_field.py 120_15D 24 --u-ring 25
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(ROOT))
import numpy as np
import build_field as bf, seeding as sd
from build_field import locate_axis
from averaging import MeanField


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label"); p.add_argument("r_target", type=float)
    p.add_argument("--u-ring", type=float, required=True, help="U_ring magnitude (mm/s)")
    p.add_argument("--suffix", default="geom")
    p.add_argument("--uniform", action="store_true",
                   help="ISOTROPIC magnification (scale x AND r by the same factor "
                        "about the ring centre) — preserves streamline aspect ratio / "
                        "topology; scales a_eq too. Default: radial-only warp (keeps a_eq).")
    args = p.parse_args()
    d = os.path.join(ROOT, "fields", args.label)
    mp = os.path.join(d, "mean_reg.pkl"); mp = mp if os.path.exists(mp) else os.path.join(d, "mean_field.pkl")
    mean = pickle.load(open(mp, "rb"))
    sign = float(np.sign(json.load(open(os.path.join(d, "meta.json"))).get("U_ring_signed", 1.0))) or 1.0

    y0 = locate_axis(mean)
    ell0 = sd.fit_core_ellipse(mean, y_axis=y0, R0=20.0)
    rc = ell0.r_c
    if args.uniform:
        # isotropic magnification about the ring centre (x_c, axis): scale x AND r
        # by the same f so streamline cells keep their shape (topology preserved).
        f = args.r_target / rc
        X_new = ell0.x_c + f * (mean.X - ell0.x_c)
        Y_new = y0 + f * (mean.Y - y0)
    else:
        # radial-only warp: translate the core ring outward, preserving a_eq.
        Delta = args.r_target - rc
        r_signed = mean.Y - y0
        r_abs = np.abs(r_signed)
        S = 1.0 / (1.0 + np.exp(-(r_abs - 0.5 * rc) / (0.15 * rc)))   # saturates before rc
        X_new = mean.X.copy()
        Y_new = y0 + np.sign(r_signed) * (r_abs + Delta * S)
    warped = MeanField(X=X_new, Y=Y_new, Ux=mean.Ux.copy(), Uy=mean.Uy.copy(),
                       omega=mean.omega.copy(), meta=dict(mean.meta or {}))

    field = bf.build_field(warped, U_ring=sign * args.u_ring, R0=20.0)
    out = os.path.join(ROOT, "fields", f"{args.label}_{args.suffix}")
    bf.save_field(field, out, mean=warped)
    meta = json.load(open(os.path.join(out, "meta.json")))
    meta.update(warp_r_core_from=rc, warp_r_core_to=args.r_target)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    ell1 = sd.fit_core_ellipse(warped, y_axis=field.y_axis_mm, R0=20.0)
    print(f"{args.label}: R_core {rc:.1f} -> {ell1.r_c:.1f} mm (target {args.r_target}); "
          f"a_eq {ell0.a_eq:.1f} -> {ell1.a_eq:.1f} mm (preserved)  -> {out}")


if __name__ == "__main__":
    main()
