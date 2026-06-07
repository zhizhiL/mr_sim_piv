"""
advect.py — Stage F: Maxey-Riley advection of bubbles in the cached field.

================================ SWAP SEAM ================================
The ODE right-hand side below (:func:`mr_rhs`) is a clean, self-contained
Maxey-Riley bubble model so the synthetic protocol runs end-to-end.  When you
drop in the validated solver from ``newCodes_norburyRings/advect_bubbles_3D_eval.py``,
replace `mr_rhs` (and, if needed, the St/Fr non-dimensionalisation) but keep:
    * the field interface  :meth:`Field3D.velocity_and_gradient`
    * the escape event      :func:`_escape_event`
    * the pool driver       :func:`advect_bubbles`
so the rest of the pipeline is unaffected.
==========================================================================

Quasi-steady: ``Du/Dt = (u . grad) u`` (no explicit ∂u/∂t).  Gravity is
optional (``gravity=True``) and acts along ``-y`` (lab vertical in the
meridional plane).
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp

from build_field import Field3D, load_field
from constants import BETA, G_MM, particle_response_time


@dataclass
class BubbleResult:
    d: float                 # diameter [mm]
    escaped: bool
    t_escape: float          # s (nan if captured)
    pos0: np.ndarray         # initial position [mm]
    pos_final: np.ndarray    # final position [mm]
    St: float = np.nan


def mr_rhs(t, s, field: Field3D, tau_p: float, beta: float, g_vec: np.ndarray):
    """Maxey-Riley RHS for one bubble.  ``s = [pos(3), vel(3)]``.

    dx/dt = v
    dv/dt = beta (u.grad)u + (1/tau_p)(u - v) + (1 - beta) g
    """
    pos = s[:3]
    vel = s[3:]
    u, gradu = field.velocity_and_gradient(pos[None, :])
    u = u[0]
    DuDt = gradu[0] @ u                      # (u.grad)u, material accel
    dvel = beta * DuDt + (u - vel) / tau_p + (1.0 - beta) * g_vec
    return np.concatenate([vel, dvel])


def _escape_event(field: Field3D):
    xmin, xmax, rmin, rmax = field.bounds
    margin = 1e-6

    def event(t, s, *args):
        x, y, z = s[0], s[1], s[2]
        r = np.sqrt(y * y + z * z)
        # signed distance to the nearest FOV wall; <=0 means escaped
        return min(x - xmin, xmax - x, r - rmin, rmax - r) - margin

    event.terminal = True
    event.direction = -1.0
    return event


def advect_one(field: Field3D, pos0, d_mm, t_max=2.0, gravity=True,
               max_step=0.02, St=np.nan, method="LSODA") -> BubbleResult:
    """Integrate a single bubble until it escapes the FOV or ``t_max``.

    The drag relaxation time ``tau_p`` for small bubbles is tiny (micro-
    seconds), which makes the Maxey-Riley system *stiff*; the default
    ``method='LSODA'`` auto-switches to a BDF integrator so near-tracer
    bubbles integrate quickly.  Use ``method='RK45'`` for larger St.
    """
    tau_p = particle_response_time(d_mm)
    g_vec = np.array([0.0, -G_MM, 0.0]) if gravity else np.zeros(3)
    # initialise the bubble at the local fluid velocity (equilibrium start)
    s0 = np.concatenate([np.asarray(pos0, float), field.velocity(pos0[None, :])[0]])

    sol = solve_ivp(
        mr_rhs, (0.0, t_max), s0, method=method,
        args=(field, tau_p, beta_for(), g_vec),
        events=_escape_event(field), max_step=max_step, rtol=1e-5, atol=1e-7,
    )
    escaped = len(sol.t_events[0]) > 0
    t_esc = float(sol.t_events[0][0]) if escaped else np.nan
    return BubbleResult(
        d=float(d_mm), escaped=escaped, t_escape=t_esc,
        pos0=np.asarray(pos0, float), pos_final=sol.y[:3, -1].copy(), St=float(St),
    )


def beta_for():
    """Density parameter beta (kept as a function so the swap-in solver can
    override the bubble model in one place)."""
    return BETA


# --------------------------------------------------------------------------
# Multiprocessing pool driver
# --------------------------------------------------------------------------
_WORKER_FIELD = None  # set in each worker via initializer


def _init_worker(field_dir):
    global _WORKER_FIELD
    _WORKER_FIELD = load_field(field_dir)


def _worker(task):
    pos0, d_mm, St, kw = task
    return advect_one(_WORKER_FIELD, pos0, d_mm, St=St, **kw)


def advect_bubbles(field_dir, positions, diameters, stokes=None,
                   n_workers=1, **kw):
    """Advect a population of bubbles (one per (position, diameter)).

    Parameters
    ----------
    field_dir : str
        Cached-field directory (reloaded in each worker; keeps the field out of
        the pickled task payload).
    positions : (N, 3) array
    diameters : (N,) array [mm]
    stokes : (N,) array, optional   (carried through to the result records)
    n_workers : int
        1 -> serial (no pool); >1 -> multiprocessing pool.
    kw : passed to :func:`advect_one` (``t_max``, ``gravity``, ``max_step``).
    """
    positions = np.atleast_2d(positions)
    diameters = np.asarray(diameters, float)
    if stokes is None:
        stokes = np.full(len(diameters), np.nan)
    tasks = [(positions[i], diameters[i], stokes[i], kw) for i in range(len(diameters))]

    if n_workers <= 1:
        field = load_field(field_dir)
        return [advect_one(field, p, d, St=St, **kw)
                for (p, d, St, _) in tasks]

    with Pool(n_workers, initializer=_init_worker, initargs=(field_dir,)) as pool:
        return pool.map(_worker, tasks)
