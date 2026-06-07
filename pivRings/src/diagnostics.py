"""
diagnostics.py — Stage G: capture/escape statistics and validation plots.

    * :func:`escape_statistics`        — capture vs escape, escape-size threshold.
    * :func:`plot_field_sanity`        — co-moving streamlines (M1 check).
    * :func:`plot_escape_distribution` — simulated vs measured escape sizes.
    * :func:`plot_trajectories`        — meridional bubble tracks.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_field import Field3D
from digiflow_io import load_escape_csv


def escape_statistics(results) -> dict:
    """Aggregate capture/escape outcomes from a list of ``BubbleResult``."""
    d = np.array([r.d for r in results])
    esc = np.array([r.escaped for r in results])
    t_esc = np.array([r.t_escape for r in results])
    out = {
        "n": len(results),
        "n_escaped": int(esc.sum()),
        "escape_fraction": float(esc.mean()) if len(results) else np.nan,
        "escape_size_threshold": float(d[esc].min()) if esc.any() else np.nan,
        "captured_size_max": float(d[~esc].max()) if (~esc).any() else np.nan,
        "mean_t_escape": float(np.nanmean(t_esc)) if esc.any() else np.nan,
    }
    return out


def plot_field_sanity(field: Field3D, out_path: str, n_seed=40):
    """Meridional streamlines of the co-moving mean field — should close into
    the ring core (build_plan milestone M1)."""
    x = np.linspace(field.x_axis[0], field.x_axis[-1], 220)
    r = np.linspace(field.r_axis[0], field.r_axis[-1], 200)
    Xg, Rg = np.meshgrid(x, r)
    Ux = field.sp_Ux.ev(Rg.ravel(), Xg.ravel()).reshape(Xg.shape)
    Ur = field.sp_Ur.ev(Rg.ravel(), Xg.ravel()).reshape(Xg.shape)
    speed = np.sqrt(Ux ** 2 + Ur ** 2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.streamplot(x, r, Ux, Ur, color=speed, cmap="viridis", density=1.3, linewidth=1)
    ax.set_xlabel("x  (mm, streamwise)")
    ax.set_ylabel("r  (mm, from axis)")
    ax.set_title("Co-moving meridional streamlines")
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax, label="|u| (norm.)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_seeds(field: Field3D, positions, out_path: str):
    """Seed ring projected into the meridional plane over the field domain."""
    r = np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(positions[:, 0], r, s=8, c="crimson", label="seeds")
    ax.set_xlim(field.x_axis[0], field.x_axis[-1])
    ax.set_ylim(field.r_axis[0], field.r_axis[-1])
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_title("Seed positions (meridional projection)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_escape_distribution(results, csv_path, out_path: str):
    """Histogram of simulated escape diameters vs the measured escape csv."""
    d_all = np.array([r.d for r in results])
    d_esc = np.array([r.d for r in results if r.escaped])
    d_meas = load_escape_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(d_all.max(), d_meas.max()) * 1.05, 25)
    ax.hist(d_all, bins=bins, alpha=0.35, label="seeded", color="gray", density=True)
    if len(d_esc):
        ax.hist(d_esc, bins=bins, alpha=0.6, label="escaped (sim)", color="crimson", density=True)
    ax.hist(d_meas, bins=bins, histtype="step", lw=2, label="escaped (measured)",
            color="navy", density=True)
    ax.set_xlabel("bubble diameter d (mm)")
    ax.set_ylabel("pdf")
    ax.set_title("Escape-size distribution: simulated vs measured")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_trajectories(field: Field3D, trajectories, out_path: str):
    """Meridional projection of a few full trajectories (debug aid).

    ``trajectories`` is a list of (t, y) solve_ivp-like arrays where y[:3] are
    positions; pass [] to skip."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for tr in trajectories:
        pos = tr
        r = np.sqrt(pos[1] ** 2 + pos[2] ** 2)
        ax.plot(pos[0], r, lw=0.8, alpha=0.7)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_title("Bubble trajectories (meridional)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_escape_vs_size(diameters, escape_fraction, out_path: str,
                        threshold=None):
    """Escape fraction vs bubble diameter — the headline capture/escape curve."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(diameters, escape_fraction, "o-", color="crimson")
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, ls=":", color="gray",
                   label=f"threshold ~ {threshold:.1f} mm")
        ax.legend()
    ax.set_xlabel("bubble diameter d (mm)")
    ax.set_ylabel("escape fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Escape fraction vs size (capture threshold)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def station_comparison(reports: dict) -> dict:
    """Empirical quasi-steady error bound from per-station escape fractions."""
    fracs = {k: v.get("escape_fraction", np.nan) for k, v in reports.items()}
    vals = np.array([v for v in fracs.values() if np.isfinite(v)])
    return {
        "per_station_escape_fraction": fracs,
        "spread": float(vals.max() - vals.min()) if len(vals) else np.nan,
    }
