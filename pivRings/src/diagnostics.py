"""
diagnostics.py — Stage G: capture/escape statistics and validation plots.

Field axes are dimensionless (x* = x/R0, r* = r/R0).  Escape diameters are
reported in mm.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_field import Field3D
from digiflow_io import load_escape_csv


def escape_statistics(results) -> dict:
    d = np.array([r.d for r in results])
    esc = np.array([r.escaped for r in results])
    t_esc = np.array([r.t_escape for r in results])
    return {
        "n": len(results),
        "n_escaped": int(esc.sum()),
        "escape_fraction": float(esc.mean()) if len(results) else np.nan,
        "escape_size_threshold": float(d[esc].min()) if esc.any() else np.nan,
        "captured_size_max": float(d[~esc].max()) if (~esc).any() else np.nan,
        "mean_t_escape": float(np.nanmean(t_esc)) if esc.any() else np.nan,
    }


def plot_field_sanity(field: Field3D, out_path: str):
    """Meridional streamlines of the co-moving dimensionless field — should
    close into the ring core (build_plan M1)."""
    x = np.linspace(field.x_axis[0], field.x_axis[-1], 220)
    r = np.linspace(field.r_axis[0], field.r_axis[-1], 200)
    Xg, Rg = np.meshgrid(x, r)
    Ux = field.sp_Ux.ev(Xg.ravel(), Rg.ravel()).reshape(Xg.shape)
    Ur = field.sp_Ur.ev(Xg.ravel(), Rg.ravel()).reshape(Xg.shape)
    speed = np.sqrt(Ux ** 2 + Ur ** 2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.streamplot(x, r, Ux, Ur, color=speed, cmap="viridis", density=1.3, linewidth=1)
    ax.set_xlabel("x* = x / R0"); ax.set_ylabel("r* = r / R0")
    ax.set_title("Co-moving meridional streamlines (dimensionless)")
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax, label="|u*|")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_seeds(field: Field3D, positions, out_path: str):
    r = np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(positions[:, 0], r, s=8, c="crimson", label="seeds")
    ax.set_xlim(field.x_axis[0], field.x_axis[-1])
    ax.set_ylim(field.r_axis[0], field.r_axis[-1])
    ax.set_xlabel("x*"); ax.set_ylabel("r*")
    ax.set_title("Seed positions (meridional projection)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_escape_distribution(results, csv_path, out_path: str):
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
    ax.set_xlabel("bubble diameter d (mm)"); ax.set_ylabel("pdf")
    ax.set_title("Escape-size distribution: simulated vs measured")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_escape_vs_size(diameters, escape_fraction, out_path: str, threshold=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(diameters, escape_fraction, "o-", color="crimson")
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, ls=":", color="gray", label=f"threshold ~ {threshold:.2f} mm")
        ax.legend()
    ax.set_xlabel("bubble diameter d (mm)"); ax.set_ylabel("escape fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Escape fraction vs size (capture threshold)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def station_comparison(reports: dict) -> dict:
    fracs = {k: v.get("escape_fraction", np.nan) for k, v in reports.items()}
    vals = np.array([v for v in fracs.values() if np.isfinite(v)])
    return {"per_station_escape_fraction": fracs,
            "spread": float(vals.max() - vals.min()) if len(vals) else np.nan}
