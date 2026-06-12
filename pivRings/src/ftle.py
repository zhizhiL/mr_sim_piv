"""
ftle.py — finite-time Lyapunov exponent (FTLE) fields for the ring.

Computed in the invariant VERTICAL meridional plane y=0 (gravity is along z, the
field is axisymmetric, so y=0 with v_y=0 is an invariant plane that contains the
trapping attractor).  The 3-D field is simply evaluated at points (x, 0, z); for
y=0 the radial distance is r=|z| and u_z = U_r * sign(z).

Two flow maps, both a vectorised fixed-step RK4 advancing the whole grid at once
(field.velocity / velocity_and_gradient are vectorised over N points):

    * :func:`fluid_flow_map`    — passive tracers dx/dt = u (forward & backward);
    * :func:`inertial_flow_map` — Maxey-Riley bubbles (forward only; backward-time
      inertial dynamics are ill-posed because the slow manifold repels backward).

FTLE = 1/(2|T|) ln sqrt(lambda_max(C)),  C = F^T F,  F = d x_T / d x_0, from
central differences of the flow map on the grid.  Particles that leave the PIV
domain are frozen and flagged; FTLE is NaN wherever a finite-difference stencil
touches a dead node (ridges near the FOV walls are therefore not trusted).
"""

from __future__ import annotations

import numpy as np

from constants import R_BUBBLE


def _xyz(P2):
    """(N,2) [x,z] -> (N,3) [x,0,z] for evaluating the axisymmetric field at y=0."""
    P = np.zeros((P2.shape[0], 3))
    P[:, 0] = P2[:, 0]
    P[:, 2] = P2[:, 1]
    return P


def _alive(field, P2):
    xmin, xmax, _, rmax = field.bounds
    x, z = P2[:, 0], P2[:, 1]
    return (x >= xmin) & (x <= xmax) & (np.abs(z) <= rmax)


def fluid_flow_map(field, P0, T, dt=0.02, direction=1):
    """Advect passive tracers in the (x,z) plane for time |T|.

    ``direction`` = +1 forward, -1 backward.  ``P0`` is (N,2) of [x,z].  Returns
    ``(Pf, alive)`` with Pf (N,2) final positions (frozen at exit) and a bool
    mask of tracers that never left the domain."""
    P = np.asarray(P0, float).copy()
    n_steps = int(abs(T) / dt)
    alive = np.ones(len(P), bool)
    s = float(np.sign(direction))

    def vel(P2):
        u = field.velocity(_xyz(P2))            # (N,3)
        return s * np.column_stack([u[:, 0], u[:, 2]])

    for _ in range(n_steps):
        a = alive
        k1 = vel(P[a])
        k2 = vel(P[a] + 0.5 * dt * k1)
        k3 = vel(P[a] + 0.5 * dt * k2)
        k4 = vel(P[a] + dt * k3)
        P[a] = P[a] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        left = a & ~_alive(field, P)
        alive[left] = False
        P[left] = np.clip(P[left], [field.bounds[0], -field.bounds[3]],
                          [field.bounds[1], field.bounds[3]])
    return P, alive


def inertial_velocity(field, P2, St, Fr, R=R_BUBBLE, gravity=True):
    """Slow-manifold (Sapsis-Haller) inertial velocity at y=0 points P2 (N,2).

    To O(St) the Maxey-Riley bubble rides the field

        v = u + (St/R)[(3R/2 - 1)(u.grad)u + a_buoy]
          = u + St (u.grad)u + W* z_hat              (R=2, W*=St/Fr^2)

    which is smooth (no stiff drag), so a fixed-step flow map is stable for all
    St.  ``St`` scalar or (N,).  Returns the (N,2) [vx,vz] inertial velocity."""
    u, g = field.velocity_and_gradient(_xyz(P2))      # u (N,3), g (N,3,3)
    mat_x = g[:, 0, 0] * u[:, 0] + g[:, 0, 2] * u[:, 2]   # (u.grad)u, x  (y=0)
    mat_z = g[:, 2, 0] * u[:, 0] + g[:, 2, 2] * u[:, 2]   # (u.grad)u, z
    St = np.asarray(St, float)
    coef = St * (1.5 * R - 1.0) / R                    # = St for R=2
    vx = u[:, 0] + coef * mat_x
    vz = u[:, 2] + coef * mat_z
    if gravity:
        vz = vz - (1.0 - 1.5 * R) / R * (St / Fr ** 2)   # = + W* for R=2
    return np.column_stack([vx, vz])


def inertial_flow_map(field, P0, St, Fr, T, dt=0.01, R=R_BUBBLE, gravity=True):
    """Advect bubbles forward in the (x,z) plane on the slow-manifold inertial
    field (:func:`inertial_velocity`) for time T.  ``St`` scalar or (N,).
    Returns ``(Pf, alive)`` like :func:`fluid_flow_map`."""
    P = np.asarray(P0, float).copy()                  # (N,2) positions [x,z]
    St = np.broadcast_to(np.asarray(St, float), (len(P),)).copy()
    n_steps = int(abs(T) / dt)
    alive = np.ones(len(P), bool)

    def vel(P2, st):
        return inertial_velocity(field, P2, st, Fr, R=R, gravity=gravity)

    for _ in range(n_steps):
        a = alive
        p, st = P[a], St[a]
        k1 = vel(p, st)
        k2 = vel(p + 0.5 * dt * k1, st)
        k3 = vel(p + 0.5 * dt * k2, st)
        k4 = vel(p + dt * k3, st)
        P[a] = p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        left = a & ~_alive(field, P)
        alive[left] = False
        P[left] = np.clip(P[left], [field.bounds[0], -field.bounds[3]],
                          [field.bounds[1], field.bounds[3]])
    return P, alive


def ftle_field(x_grid, z_grid, Pf, alive, T):
    """FTLE on the grid from a flow map.

    ``x_grid`` (nx,), ``z_grid`` (nz,); ``Pf`` (nz*nx, 2) final positions ordered
    as ``meshgrid(x_grid, z_grid)`` raveled (row-major over z then x).  Returns an
    (nz, nx) FTLE array (NaN where a central-difference stencil hits a dead node
    or the grid edge)."""
    nx, nz = len(x_grid), len(z_grid)
    Xf = Pf[:, 0].reshape(nz, nx)
    Zf = Pf[:, 1].reshape(nz, nx)
    A = alive.reshape(nz, nx)
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]

    ftle = np.full((nz, nx), np.nan)
    for i in range(1, nz - 1):
        for j in range(1, nx - 1):
            if not (A[i, j] and A[i, j - 1] and A[i, j + 1] and A[i - 1, j] and A[i + 1, j]):
                continue
            dXdx = (Xf[i, j + 1] - Xf[i, j - 1]) / (2 * dx)
            dZdx = (Zf[i, j + 1] - Zf[i, j - 1]) / (2 * dx)
            dXdz = (Xf[i + 1, j] - Xf[i - 1, j]) / (2 * dz)
            dZdz = (Zf[i + 1, j] - Zf[i - 1, j]) / (2 * dz)
            F = np.array([[dXdx, dXdz], [dZdx, dZdz]])
            C = F.T @ F
            lam = np.linalg.eigvalsh(C)[-1]
            if lam > 0:
                ftle[i, j] = np.log(lam) / (4.0 * abs(T))      # 1/(2T) ln sqrt(lam)
    return ftle
