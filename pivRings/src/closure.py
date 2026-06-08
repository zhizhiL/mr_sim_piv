"""
closure.py — estimate U_ring by closing the co-moving recirculation atmosphere.

The vorticity-centroid / core position fits can be biased (diffuse wake
vorticity, decay), giving a co-moving speed that does NOT close the ring's
streamlines, so bubbles advect straight through and nothing traps.  Here U_ring
is defined physically: the largest co-moving speed at which the recirculation
atmosphere still closes (the separatrix transition).  Below it the atmosphere is
artificially large; above it the through-flow sweeps everything out.

``uring_by_closure`` sweeps candidate speeds, builds the field, seeds a
meridional grid of pure tracers around the core, and returns the transition
speed (largest U with trapped fraction >= threshold).
"""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np

import build_field as bf
import seeding as sd
import advect as ad


def _trapped_fraction(mean, U_signed, R0, n_grid=5, t_phys=1.5, span=1.2):
    """Fraction of a meridional tracer grid around the core that stays in the
    FOV (pure tracers: tiny St, no gravity), over a FIXED PHYSICAL time.

    ``U_signed`` may be negative (ring propagating in -x).  The integration
    horizon is ``t_max* = t_phys * |U_ring| / R0`` (with a floor), so every
    candidate sees the same physical duration -- this removes the spurious
    "everything traps at high U" bias of a fixed dimensionless t_max."""
    field = bf.build_field(mean, U_ring=U_signed, R0=R0)
    t_max = max(t_phys * abs(U_signed) / R0, 3.0)
    try:
        ell = sd.fit_core_ellipse(mean, y_axis=field.y_axis_mm, R0=R0)
    except Exception:
        return 0.0
    xc, rc = ell.x_c / R0, ell.r_c / R0
    a = max(ell.a_eq, 3.0) / R0
    offs = np.linspace(-span * a, span * a, n_grid)
    c = t = 0
    for dx in offs:
        for dr in offs:
            r = rc + dr
            if r <= 0:
                continue
            p = np.array([xc + dx, r, 0.0])
            if not field.in_domain(p[None, :])[0]:
                continue
            res = ad.advect_one(field, p, 0.3, St=1e-3, Fr=1.0, gravity=False, t_max=t_max)
            c += (not res.escaped)
            t += 1
    return c / t if t else 0.0


_WMEAN = {}


def _init(mean, R0, n_grid):
    _WMEAN["mean"], _WMEAN["R0"], _WMEAN["n_grid"] = mean, R0, n_grid


def _worker(U):
    return U, _trapped_fraction(_WMEAN["mean"], U, _WMEAN["R0"], n_grid=_WMEAN["n_grid"])


def _eval(mean, R0, cands, n_grid, n_workers):
    if n_workers <= 1:
        return [(float(U), float(_trapped_fraction(mean, U, R0, n_grid=n_grid))) for U in cands]
    with Pool(n_workers, initializer=_init, initargs=(mean, R0, n_grid)) as pool:
        out = pool.map(_worker, cands)
    return [(float(U), float(tf)) for U, tf in out]


def uring_by_closure(mean, R0=20.0, u_lo=10.0, u_hi=200.0, step=10.0,
                     thresh=0.5, n_grid=5, n_workers=16, verbose=False):
    """Return ``(U_ring, curve, status)``: the signed co-moving speed at which
    the recirculation atmosphere closes.

    Scans BOTH signs (ring may propagate in +x or -x).  Within the dominant
    sign, ``U_ring`` is the **midpoint of the trapping band** (most robust point
    of the closed-atmosphere plateau).  Uses a fixed physical integration time
    (in ``_trapped_fraction``) so the band is not biased by the time scale.
    ``status`` in {"ok", "weak"} (weak: trapping band is thin/peaky)."""
    mags = np.arange(u_lo, u_hi + step / 2, step)
    best = None
    for sign in (+1.0, -1.0):
        curve = _eval(mean, R0, [sign * m for m in mags], n_grid, n_workers)
        score = sum(tf for _, tf in curve)
        if best is None or score > best[0]:
            best = (score, sign, curve)
    _, sign, curve = best
    if verbose:
        for U, tf in curve:
            print(f"    U={U:+7.1f}  trapped={tf:.2f}")
    trap_U = [U for (U, tf) in curve if tf >= thresh]
    if not trap_U:
        # fall back to the peak-trapping candidate
        U = max(curve, key=lambda c: c[1])[0]
        return float(U), curve, "weak"
    U = float(0.5 * (min(trap_U) + max(trap_U)))     # midpoint of trapping band
    status = "ok" if len(trap_U) >= 2 else "weak"
    return U, curve, status
