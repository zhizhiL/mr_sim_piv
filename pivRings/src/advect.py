"""
advect.py — Stage F: Maxey-Riley advection (DIMENSIONLESS), adapted from
``advect_bubbles_3D_eval.py``.

Same RHS form as the reused solver (``solve_ivp_active``):

    dx/dt = v
    dv/dt = R (u - v)/St  +  (3R/2)(u . grad)u                 [drag + added-mass]
    dv_z/dt += -(1 - 3R/2)/Fr^2   (gravity along z)            [buoyancy]

Everything is dimensionless (x in R0, t in tau_f=R0/U_ring).  ``St`` is per
bubble; ``R`` and ``Fr`` are per ring.  The one deliberate change from the old
file: the planar->Cartesian gradient conversion uses the CORRECTED chain rule
(``Field3D.velocity_and_gradient``, with the r->0 axis guard) instead of the old
``1/cosθ, 1/sinθ`` form, which was singular on the axes (build_plan §0).
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from matplotlib.path import Path as _MplPath
from scipy.integrate import solve_ivp

from build_field import Field3D, load_field
from constants import R_BUBBLE


@dataclass
class BubbleResult:
    d: float                 # diameter [mm]
    St: float
    escaped: bool
    t_escape: float          # dimensionless (nan if captured)
    pos0: np.ndarray         # initial position (dimensionless)
    pos_final: np.ndarray
    exit_face: str = ""      # which FOV wall it left by: x_min|x_max|r_top|r_bot|""
    # buoyant detrainment (rises out the top, z>0 via the radial wall) vs
    # advective FOV-exit (leaves through an axial wall while still orbiting).
    # With escape="separatrix" exit_face is "separatrix" (crossed the atmosphere
    # dividing streamline) or a FOV wall (atmosphere>FOV truncation fallback).


def _classify_exit(field: Field3D, s):
    """Which FOV wall the state s exited through (and top vs bottom radially)."""
    x, y, z = s[0], s[1], s[2]
    r = np.sqrt(y * y + z * z)
    xmin, xmax, rmin, rmax = field.bounds
    dists = {"x_min": x - xmin, "x_max": xmax - x, "r_wall": rmax - r}
    face = min(dists, key=dists.get)
    if face == "r_wall":
        return "r_top" if z >= 0 else "r_bot"   # gravity is +z -> top = buoyant
    return face


def mr_rhs(t, s, field: Field3D, St: float, R: float, Fr: float, gravity: bool):
    """Dimensionless Maxey-Riley RHS for one bubble.  ``s=[x,y,z,vx,vy,vz]``."""
    pos = s[:3]
    vel = s[3:]
    u, gradu = field.velocity_and_gradient(pos[None, :])
    u = u[0]
    material = gradu[0] @ u                       # (u.grad)u
    acc = R * (u - vel) / St + (3.0 * R / 2.0) * material
    if gravity:
        acc[2] += -(1.0 - 3.0 * R / 2.0) / (Fr ** 2)
    return np.concatenate([vel, acc])


def _escape_event(field: Field3D):
    xmin, xmax, rmin, rmax = field.bounds
    margin = 1e-9

    def event(t, s, *args):
        x, y, z = s[0], s[1], s[2]
        r = np.sqrt(y * y + z * z)
        return min(x - xmin, xmax - x, r - rmin, rmax - r) - margin

    event.terminal = True
    event.direction = -1.0
    return event


def _separatrix_exit(field, sol, sep_poly, fov_exit):
    """Escape vs capture from a trajectory and the atmosphere polygon.

    ``escape`` = the bubble has *sustainedly* left the separatrix: either it ran
    out the FOV, or its FINAL state is outside the dividing streamline.  We take
    the start of the last continuous outside-run as ``t_escape`` (so a captured
    bubble's transient orbit dips back inside don't count — REPORT §7e).  Returns
    ``(escaped, t_escape, pos_final, exit_face)``."""
    XY = sol.y[:3, :]
    r = np.hypot(XY[1, :], XY[2, :])
    inside = _MplPath(sep_poly).contains_points(np.column_stack([XY[0, :], r]))
    in_idx = np.where(inside)[0]

    if len(in_idx) == 0:                       # never inside -> escaped at start
        i = 0
        escaped = True
    elif in_idx[-1] == len(inside) - 1 and not fov_exit:
        escaped = False                        # ends inside, stayed in FOV -> captured
    else:
        escaped = True
        i = min(in_idx[-1] + 1, XY.shape[1] - 1)

    if not escaped:
        return False, np.nan, XY[:, -1].copy(), ""
    s_exit = sol.y[:, i]
    if inside[i]:
        # atmosphere>FOV truncation fallback: still inside the polygon at exit ->
        # the bubble actually left through a FOV wall (tag it, REPORT §7e gotcha).
        face = _classify_exit(field, s_exit)
    else:
        # classify the separatrix crossing in the same r_top/x_min/x_max vocabulary
        # as the FOV tags: a crossing of the radial dome (r near the apex) is
        # buoyant detrainment ("r_top"); a crossing near the axial ends is the
        # wash-through ("x_min"/"x_max").  Core-agnostic (uses r=hypot(y,z)).
        x_e = s_exit[0]
        r_e = np.hypot(s_exit[1], s_exit[2])
        r_apex = sep_poly[:, 1].max()
        xlo, xhi = sep_poly[:, 0].min(), sep_poly[:, 0].max()
        if r_e >= 0.6 * r_apex:
            face = "r_top"
        else:
            face = "x_min" if abs(x_e - xlo) <= abs(x_e - xhi) else "x_max"
    return True, float(sol.t[i]), s_exit[:3].copy(), face


def advect_one(field: Field3D, pos0, d_mm, St, Fr, R=R_BUBBLE, gravity=True,
               t_max=20.0, n_eval=400, method="LSODA", escape="fov",
               sep_poly=None) -> BubbleResult:
    """Integrate one bubble until it escapes the FOV or t_max (dimensionless).

    Small bubbles have small St (stiff drag R/St) and, when Fr is small, large
    buoyancy — so the default integrator is the stiff-aware ``LSODA``.  Output is
    stored only at ``n_eval`` evaluation points (``t_eval``); without this,
    solve_ivp keeps every internal step and a stiff trajectory can exhaust
    memory (this matches the original solver's ``t_eval=linspace(...,500)``).

    ``escape="fov"`` (default) flags escape when the bubble crosses the PIV-FOV
    rectangle (FOV- and ``t_max``-dependent, REPORT §7d).  ``escape="separatrix"``
    flags escape when the bubble crosses the co-moving atmosphere dividing
    streamline ``sep_poly`` (an (M,2) (x,r) polygon, FOV-independent, REPORT §7e);
    the FOV event is still integrated as an outer safety bound."""
    v0 = field.velocity(np.atleast_2d(pos0))[0]   # start at local fluid velocity
    s0 = np.concatenate([np.asarray(pos0, float), v0])
    t_eval = np.linspace(0.0, t_max, n_eval)
    sol = solve_ivp(mr_rhs, (0.0, t_max), s0, method=method, t_eval=t_eval,
                    args=(field, St, R, Fr, gravity),
                    events=_escape_event(field), rtol=1e-6, atol=1e-8)
    fov_exit = len(sol.t_events[0]) > 0

    if escape == "separatrix":
        if sep_poly is None:
            raise ValueError("escape='separatrix' requires sep_poly")
        escaped, t_esc, pos_final, exit_face = _separatrix_exit(
            field, sol, np.asarray(sep_poly, float), fov_exit)
    else:
        escaped = fov_exit
        t_esc = float(sol.t_events[0][0]) if escaped else np.nan
        if escaped and len(sol.y_events[0]):
            s_exit = sol.y_events[0][0]
            pos_final = s_exit[:3].copy()
            exit_face = _classify_exit(field, s_exit)
        else:
            pos_final = sol.y[:3, -1].copy()
            exit_face = ""
    return BubbleResult(d=float(d_mm), St=float(St), escaped=escaped,
                        t_escape=t_esc, pos0=np.asarray(pos0, float),
                        pos_final=pos_final, exit_face=exit_face)


# --------------------------------------------------------------------------
# Multiprocessing pool driver
# --------------------------------------------------------------------------
_WF = {}   # per-worker state


def _init_worker(field_dir, Fr, R, gravity, kw):
    _WF["field"] = load_field(field_dir)
    _WF["Fr"], _WF["R"], _WF["gravity"], _WF["kw"] = Fr, R, gravity, kw


def _worker(task):
    pos0, d, St = task
    return advect_one(_WF["field"], pos0, d, St, _WF["Fr"], R=_WF["R"],
                      gravity=_WF["gravity"], **_WF["kw"])


def advect_bubbles(field_dir, positions, diameters, stokes, Fr, R=R_BUBBLE,
                   gravity=True, n_workers=1, **kw):
    """Advect a population (one bubble per position/diameter/St).

    ``field_dir`` is reloaded inside each worker (keeps the field out of the
    pickled task payload).  ``kw`` -> :func:`advect_one` (t_max, method,
    escape, sep_poly).  For ``escape="separatrix"`` pass the (M,2) polygon as
    ``sep_poly`` (a small ndarray, fine in the pickled worker state)."""
    positions = np.atleast_2d(positions)
    diameters = np.asarray(diameters, float)
    stokes = np.asarray(stokes, float)
    tasks = list(zip(positions, diameters, stokes))

    if n_workers <= 1:
        field = load_field(field_dir)
        return [advect_one(field, p, d, St, Fr, R=R, gravity=gravity, **kw)
                for (p, d, St) in tasks]

    with Pool(n_workers, initializer=_init_worker,
              initargs=(field_dir, Fr, R, gravity, kw)) as pool:
        return pool.map(_worker, tasks)


def advect_trajectories(field: Field3D, positions, diameters, stokes, Fr,
                        R=R_BUBBLE, gravity=True, t_max=20.0, n_eval=160,
                        method="LSODA"):
    """Integrate a population and return full trajectories on a common time grid.

    Returns ``(t_eval, traj, escaped)`` where ``traj`` is (N, n_eval, 3) of
    positions (NaN after a bubble escapes the FOV, so it disappears in a movie)
    and ``escaped`` is (N,) bool.  Serial (no pool) — intended for a few hundred
    bubbles for visualisation."""
    positions = np.atleast_2d(positions)
    diameters = np.asarray(diameters, float)
    stokes = np.asarray(stokes, float)
    t_eval = np.linspace(0.0, t_max, n_eval)
    N = len(positions)
    traj = np.full((N, n_eval, 3), np.nan)
    escaped = np.zeros(N, bool)

    for i in range(N):
        v0 = field.velocity(positions[i:i + 1])[0]
        s0 = np.concatenate([positions[i], v0])
        sol = solve_ivp(mr_rhs, (0.0, t_max), s0, method=method, t_eval=t_eval,
                        args=(field, stokes[i], R, Fr, gravity),
                        events=_escape_event(field), rtol=1e-6, atol=1e-8)
        k = sol.y.shape[1]
        traj[i, :k, :] = sol.y[:3, :].T
        escaped[i] = len(sol.t_events[0]) > 0
    return t_eval, traj, escaped


def to_solver_bubbles_df(positions, velocities, stokes):
    """Assemble the ``bubbles_df`` layout the original solver expects:
    columns [id, x, y, z, vx, vy, vz, St] (``initial_states = df[:, 1:8]``)."""
    positions = np.atleast_2d(positions)
    velocities = np.atleast_2d(velocities)
    n = len(positions)
    ids = np.arange(n).reshape(-1, 1)
    return np.hstack([ids, positions, velocities, np.asarray(stokes).reshape(-1, 1)])
