import numpy as np
import scipy as sp
import pickle
from scipy.integrate import solve_ivp
from multiprocessing import Pool
import matplotlib.pyplot as plt

R, gravity, Fr = 2, True, 0.5

path = 'velocity_fields/alpha_0.4_2D/'

x_grid = np.load(path + 'x.npy')
r_grid = np.load(path + 'y.npy')
Ux = np.load(path + 'Ux.npy')
Ur = np.load(path + 'Uy.npy')
dUxdx = np.load(path + 'dUxdx.npy')
dUxdr = np.load(path + 'dUxdy.npy')
dUrdx = np.load(path + 'dUydx.npy')
dUrdr = np.load(path + 'dUydy.npy')
geometry = np.load(path + 'geometry.npy', allow_pickle=True)
x_core, y_core, y_core_lower, x_ring, y_ring = geometry.T

# load interpolation functions
with open(path + 'interp_functions.pkl', 'rb') as f:
    interp_Ux, interp_Ur, interp_dUxdx, interp_dUxdr, interp_dUrdx, interp_dUrdr = pickle.load(f)

def solve_ivp_active(args):

    def active_tracer_traj(t,Z):
        xp, yp, zp, dxpdt, dypdt, dzpdt = Z

        r_planar = np.sqrt(yp**2 + zp**2)
        theta = np.arctan2(zp, yp)   # check if this is correct

        Uxp = interp_Ux(xp, r_planar)[0]
        Urp = interp_Ur(xp, r_planar)[0]
        Uyp = Urp * yp / r_planar
        Uzp = Urp * zp / r_planar

        dUxdx_p = interp_dUxdx(xp, r_planar)[0]
        dUxdr_p = interp_dUxdr(xp, r_planar)[0]
        dUrdx_p = interp_dUrdx(xp, r_planar)[0]
        dUrdr_p = interp_dUrdr(xp, r_planar)[0]

        # apply chain rule for derivatives in the cartesian coordinate
        dUxdy_p = dUxdr_p / np.cos(theta) if np.cos(theta) != 0 else 0
        dUxdz_p = dUxdr_p / np.sin(theta) if np.sin(theta) != 0 else 0
        
        dUydx_p = dUrdx_p * np.cos(theta)
        dUydy_p = dUrdr_p + Urp / r_planar if r_planar != 0 else 0
        dUydz_p = dUrdr_p * np.cos(theta)/np.sin(theta) - Urp * np.cos(theta) *np.sin(theta) / r_planar if (np.sin(theta) != 0) and (r_planar != 0) else 0

        dUzdx_p = dUrdx_p * np.sin(theta)
        dUzdy_p = dUrdr_p * np.sin(theta)/np.cos(theta) - Urp * np.cos(theta) * np.sin(theta) / r_planar if (np.cos(theta) != 0) and (r_planar != 0) else 0
        dUzdz_p = dUrdr_p + Urp / r_planar if r_planar != 0 else 0

        dUxdt = 0
        dUydt = 0
        dUzdt = 0

        ddxpdtt = R*(Uxp - dxpdt)/St + (3*R/2) * (dUxdt + Uxp*dUxdx_p + Uyp*dUxdy_p + Uzp*dUxdz_p)
        ddypdtt = R*(Uyp - dypdt)/St + (3*R/2) * (dUydt + Uxp*dUydx_p + Uyp*dUydy_p + Uzp*dUydz_p)
        ddzpdtt = R*(Uzp - dzpdt)/St + (3*R/2) * (dUzdt + Uxp*dUzdx_p + Uyp*dUzdy_p + Uzp*dUzdz_p) - gravity * (1-3*R/2) / (Fr**2)

        return [dxpdt, dypdt, dzpdt, ddxpdtt, ddypdtt, ddzpdtt]
    
    q0, t_span = args
    x0, y0, z0, vx0, vy0, vz0, St = q0
    sol = sp.integrate.solve_ivp(active_tracer_traj, [t_span[0], t_span[-1]], [x0, y0, z0, vx0, vy0, vz0], method='RK45', t_eval=t_span, vectorized=True)

    return sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]


def advect_bubbles(bubbles_df_to_advect, t0, tf, plot_path = False, this_ax = None, color=None):
    initial_states = bubbles_df_to_advect[:, 1:8]
    t_span = np.linspace(t0, tf, 500)

    n_proc = 12

    with Pool(n_proc) as pool:
        args = list(zip(initial_states, [t_span]*len(initial_states)))
        res = pool.map(solve_ivp_active, args)

    res_array = np.stack(res, axis=0) # shape (N_bubbles, 4, len(t_span))

    if plot_path:
        plt.sca(ax=this_ax)
        plt.scatter(res_array[:, 0].T, res_array[:, 1].T, res_array[:, 2].T, color=color,linewidths=0)
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.zlabel('z')
        plt.show()

    return res_array


def solve_ivp_active_res(args):

    def active_tracer_traj_res(t,Z):
        xp, yp, zp, dxpdt, dypdt, dzpdt = Z

        r_planar = np.sqrt(yp**2 + zp**2)
        theta = np.arctan2(zp, yp)   # check if this is correct

        Uxp = interp_Ux(xp, r_planar)[0]
        Urp = interp_Ur(xp, r_planar)[0]
        Uyp = Urp * yp / r_planar
        Uzp = Urp * zp / r_planar

        dUxdx_p = interp_dUxdx(xp, r_planar)[0]
        dUxdr_p = interp_dUxdr(xp, r_planar)[0]
        dUrdx_p = interp_dUrdx(xp, r_planar)[0]
        dUrdr_p = interp_dUrdr(xp, r_planar)[0]

        # apply chain rule for derivatives in the cartesian coordinate
        dUxdy_p = dUxdr_p / np.cos(theta) if np.cos(theta) != 0 else 0
        dUxdz_p = dUxdr_p / np.sin(theta) if np.sin(theta) != 0 else 0
        
        dUydx_p = dUrdx_p * np.cos(theta)
        dUydy_p = dUrdr_p + Urp / r_planar if r_planar != 0 else 0
        dUydz_p = dUrdr_p * np.cos(theta)/np.sin(theta) - Urp * np.cos(theta) *np.sin(theta) / r_planar if (np.sin(theta) != 0) and (r_planar != 0) else 0

        dUzdx_p = dUrdx_p * np.sin(theta)
        dUzdy_p = dUrdr_p * np.sin(theta)/np.cos(theta) - Urp * np.cos(theta) * np.sin(theta) / r_planar if (np.cos(theta) != 0) and (r_planar != 0) else 0
        dUzdz_p = dUrdr_p + Urp / r_planar if r_planar != 0 else 0

        dUxdt = 0
        dUydt = 0
        dUzdt = 0

        ddxpdtt = R*(Uxp - dxpdt)/St + (3*R/2) * (dUxdt + Uxp*dUxdx_p + Uyp*dUxdy_p + Uzp*dUxdz_p)
        ddypdtt = R*(Uyp - dypdt)/St + (3*R/2) * (dUydt + Uxp*dUydx_p + Uyp*dUydy_p + Uzp*dUydz_p)
        ddzpdtt = R*(Uzp - dzpdt)/St + (3*R/2) * (dUzdt + Uxp*dUzdx_p + Uyp*dUzdy_p + Uzp*dUzdz_p) 

        return [dxpdt, dypdt, dzpdt, ddxpdtt, ddypdtt, ddzpdtt]
    
    q0, t_span = args
    x0, y0, z0, vx0, vy0, vz0, St = q0
    sol = sp.integrate.solve_ivp(active_tracer_traj_res, [t_span[0], t_span[-1]], [x0, y0, z0, vx0, vy0, vz0], method='RK45', t_eval=t_span, vectorized=True)

    return sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]


def advect_bubbles_res(bubbles_df_to_advect, t0, tf, plot_path = False, this_ax = None, color=None):
    initial_states = bubbles_df_to_advect[:, 1:8]
    t_span = np.linspace(t0, tf, 500)

    n_proc = 12

    with Pool(n_proc) as pool:
        args = list(zip(initial_states, [t_span]*len(initial_states)))
        res = pool.map(solve_ivp_active_res, args)

    res_array = np.stack(res, axis=0) # shape (N_bubbles, 4, len(t_span))

    if plot_path:
        plt.sca(ax=this_ax)
        plt.scatter(res_array[:, 0].T, res_array[:, 1].T, res_array[:, 2].T, color=color,linewidths=0)
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.zlabel('z')
        plt.show()

    return res_array


def solve_ivp_active_pool(args):

    def active_tracer_traj_pool(t,Z):
        xp, yp, zp, dxpdt, dypdt, dzpdt = Z

        ddxpdtt = R*(- dxpdt)/St 
        ddypdtt = R*(- dypdt)/St 
        ddzpdtt = R*(- dzpdt)/St - gravity * (1-3*R/2) / (Fr**2)

        return [dxpdt, dypdt, dzpdt, ddxpdtt, ddypdtt, ddzpdtt]
    
    q0, t_span = args
    x0, y0, z0, vx0, vy0, vz0, St = q0
    sol = sp.integrate.solve_ivp(active_tracer_traj_pool, [t_span[0], t_span[-1]], [x0, y0, z0, vx0, vy0, vz0], method='RK45', t_eval=t_span, vectorized=True)

    return sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]


def advect_bubbles_pool(bubbles_df_to_advect, t0, tf, plot_path = False, this_ax = None, color=None):
    initial_states = bubbles_df_to_advect[:, 1:8]
    t_span = np.linspace(t0, tf, 100)

    n_proc = 12

    with Pool(n_proc) as pool:
        args = list(zip(initial_states, [t_span]*len(initial_states)))
        res = pool.map(solve_ivp_active_pool, args)

    res_array = np.stack(res, axis=0) # shape (N_bubbles, 4, len(t_span))

    if plot_path:
        plt.sca(ax=this_ax)
        plt.scatter(res_array[:, 0].T, res_array[:, 1].T, res_array[:, 2].T, color=color,linewidths=0)
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.zlabel('z')
        plt.show()

    return res_array