"""
vortex_core.py
==============
Iterative vortex-core finder for DigiFlow PIV data.

Usage
-----
    from vortex_core import find_vortex_cores_iterative

The function operates entirely in world coordinates (mm) and returns core
positions plus the row index of the estimated ring centreline.
"""

import numpy as np
from scipy.optimize import curve_fit


def find_vortex_cores_iterative(
    X, Y, vorticity,
    initial_guess_pos=None,
    initial_guess_neg=None,
    fit_radius=10.0,
    tol=0.1,
    max_iter=50,
    verbose=False,
):
    """
    Find both vortex core centres iteratively using quadratic vorticity fitting.

    Starting from the global vorticity extrema (or user-supplied guesses), each
    core is refined by repeatedly fitting a 2-D quadratic to the vorticity
    field within `fit_radius` of the current estimate and shifting the centre
    to the analytic extremum of the fit.  Iteration stops when the shift falls
    below `tol` or `max_iter` is reached.

    Parameters
    ----------
    X, Y : 2-D ndarray
        World-coordinate grids (mm), same shape as `vorticity`.
    vorticity : 2-D ndarray
        Vorticity field (1/s).
    initial_guess_pos : (float, float) or None
        Starting (x, y) in mm for the positive-vorticity core.
        If None, the global maximum of `vorticity` is used.
    initial_guess_neg : (float, float) or None
        Starting (x, y) in mm for the negative-vorticity core.
        If None, the global minimum of `vorticity` is used.
    fit_radius : float
        Radius (mm) of the fitting window around each core estimate.
    tol : float
        Convergence tolerance (mm); iteration stops when the shift < tol.
    max_iter : int
        Maximum number of refinement iterations per core.
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    cores : list of (x_centre, y_centre, sign)
        ``sign`` is +1 for the positive core and -1 for the negative core.
        Coordinates are in world mm.
    centerline_idx : int
        Row index of the estimated ring centreline (midpoint between the two
        cores mapped back to the Y grid).
    gammas : list of float
        Circulation [mm²/s] for each core, computed by integrating vorticity
        over the half-plane region enclosed by the zero-vorticity ellipse of
        the final quadratic fit.  ``gammas[0]`` corresponds to the positive
        core, ``gammas[1]`` to the negative core.  NaN when the ellipse
        cannot be determined from the fit.
    """
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    omega_flat = vorticity.flatten()

    def _quadratic_model(xy, a0, a1, a2, a3, a4, a5):
        xp, yp = xy
        return a0 + a1*xp + a2*xp**2 + a3*xp*yp + a4*yp + a5*yp**2

    def _iterate_core(x0, y0, sign, label):
        xc, yc = x0, y0
        popt_final = None
        if verbose:
            print(f"[{label}] start ({xc:.2f}, {yc:.2f})")

        for iteration in range(max_iter):
            dxp = X_flat - xc
            dyp = Y_flat - yc
            r_local = np.sqrt(dxp**2 + dyp**2)
            mask = r_local <= fit_radius

            if np.sum(mask) < 6:
                if verbose:
                    print(f"[{label}] iter {iteration}: fewer than 6 points in window — stopping")
                break

            xp_win = dxp[mask]
            yp_win = dyp[mask]
            om_win = omega_flat[mask]

            omega_c_guess = sign * np.abs(om_win).max()
            curv_guess = -omega_c_guess / fit_radius**2
            p0 = [omega_c_guess, 0.0, curv_guess, 0.0, 0.0, curv_guess]

            try:
                popt, _ = curve_fit(
                    _quadratic_model, (xp_win, yp_win), om_win, p0=p0, maxfev=5000
                )
                popt_final = popt
            except Exception as exc:
                if verbose:
                    print(f"[{label}] iter {iteration}: curve_fit failed ({exc}) — stopping")
                break

            a0, a1, a2, a3, a4, a5 = popt
            H = np.array([[2*a2, a3], [a3, 2*a5]])
            grad = np.array([a1, a4])

            try:
                delta = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"[{label}] iter {iteration}: singular Hessian — stopping")
                break

            shift_mag = np.sqrt(delta[0]**2 + delta[1]**2)
            # clip step to stay within the fit window
            if shift_mag > fit_radius:
                delta *= fit_radius / shift_mag
                shift_mag = fit_radius

            xc += delta[0]
            yc += delta[1]

            if verbose:
                print(f"[{label}] iter {iteration}: shift = {shift_mag:.4f} mm  →  ({xc:.2f}, {yc:.2f})")

            if shift_mag < tol:
                if verbose:
                    print(f"[{label}] converged after {iteration + 1} iterations")
                break

        return xc, yc, popt_final

    # ── Positive core ─────────────────────────────────────────────────────────
    if initial_guess_pos is None:
        idx = np.unravel_index(np.argmax(vorticity), vorticity.shape)
        x0_pos, y0_pos = X[idx], Y[idx]
    else:
        x0_pos, y0_pos = initial_guess_pos
    xc_pos, yc_pos, popt_pos = _iterate_core(x0_pos, y0_pos, sign=+1, label="Pos")

    # ── Negative core ─────────────────────────────────────────────────────────
    if initial_guess_neg is None:
        idx = np.unravel_index(np.argmin(vorticity), vorticity.shape)
        x0_neg, y0_neg = X[idx], Y[idx]
    else:
        x0_neg, y0_neg = initial_guess_neg
    xc_neg, yc_neg, popt_neg = _iterate_core(x0_neg, y0_neg, sign=-1, label="Neg")

    cores = [(xc_pos, yc_pos, +1), (xc_neg, yc_neg, -1)]

    # ── Centreline row index ──────────────────────────────────────────────────
    Y_col = Y[:, 0]
    idx_pos_row = int(np.argmin(np.abs(Y_col - yc_pos)))
    idx_neg_row = int(np.argmin(np.abs(Y_col - yc_neg)))
    centerline_idx = int(0.5 * (idx_pos_row + idx_neg_row))

    # ── Ellipse-based circulation ─────────────────────────────────────────────
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])

    def _gamma_from_ellipse(xc, yc, sign, popt):
        """Integrate vorticity over the fitted zero-vorticity ellipse half-plane."""
        if popt is None:
            return np.nan

        a0, a1, a2, a3, a4, a5 = popt
        H = np.array([[2*a2, a3], [a3, 2*a5]])
        grad = np.array([a1, a4])

        try:
            x0_fit, y0_fit = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            return np.nan

        # Peak vorticity at the fitted extremum (offset from converged centre)
        omega_c_fit = (a0 + a1*x0_fit + a2*x0_fit**2
                       + a3*x0_fit*y0_fit + a4*y0_fit + a5*y0_fit**2)

        # Eigensystem: principal axes of the ellipse
        eigenvalues, eigenvectors = np.linalg.eig(H)
        lam = eigenvalues.real
        ev  = eigenvectors.real

        if (omega_c_fit * lam[0]) >= 0 or (omega_c_fit * lam[1]) >= 0:
            return np.nan  # not a proper extremum

        a_ax0 = float(np.sqrt(-2.0 * omega_c_fit / lam[0]))
        a_ax1 = float(np.sqrt(-2.0 * omega_c_fit / lam[1]))

        # World-coord centre of the ellipse
        x_ell = xc + x0_fit
        y_ell = yc + y0_fit

        # Half-plane selection
        if sign > 0:
            X_half     = X[centerline_idx:, :]
            Y_half     = Y[centerline_idx:, :]
            omega_half = vorticity[centerline_idx:, :]
        else:
            X_half     = X[:centerline_idx, :]
            Y_half     = Y[:centerline_idx, :]
            omega_half = vorticity[:centerline_idx, :]

        # Displacement from ellipse centre → principal coordinates
        dXh = (X_half - x_ell).flatten()
        dYh = (Y_half - y_ell).flatten()
        pts_principal = ev.T @ np.stack([dXh, dYh], axis=0)   # (2, N)
        xi_flat  = pts_principal[0]
        eta_flat = pts_principal[1]

        ellipse_mask = (xi_flat / a_ax0)**2 + (eta_flat / a_ax1)**2 <= 1.0
        gamma = float(np.sum(omega_half.flatten()[ellipse_mask]) * dx * dy)
        return gamma

    gammas = [
        _gamma_from_ellipse(xc_pos, yc_pos, +1, popt_pos),
        _gamma_from_ellipse(xc_neg, yc_neg, -1, popt_neg),
    ]

    return cores, centerline_idx, gammas
