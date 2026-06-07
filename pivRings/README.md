# pivRings — PIV-driven Maxey–Riley bubble-in-vortex-ring model

Implements [`../build_plan.md`](../build_plan.md) with the dimensionless seam
spelled out in [`../dimension_conversion.md`](../dimension_conversion.md): take a
**measured PIV carrier field** at downstream stations, process it into a
quasi-steady axisymmetric **dimensionless** 3D field, and advect Maxey–Riley
bubbles (the reused `../advect_bubbles_3D_eval.py` RHS) to predict
capture/escape statistics.

**Units.** PIV ingestion is dimensional (mm, mm/s, s). Conversion to
dimensionless happens **once**, in `src/nondimensional.py`
(`L_f = R0`, `U_f = U_ring`); everything downstream is dimensionless and matches
the solver. Fluid = water @ 20 °C, particle = hydrogen bubbles
(`src/constants.py`).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_synthetic_demo.py        # full A→G protocol on synthetic data
```

Outputs (`outputs/`): `M1_streamlines.png`, `M2_seeds.png`,
`M3_escape_vs_size.png`, `M3_escape_sizes.png`.

## Dimensionless conventions (from `dimension_conversion.md`)

| group | formula | scope |
|-------|---------|-------|
| length / velocity / time | `L_f=R0=20mm`, `U_f=U_ring`, `tau_f=R0/U_ring` | per ring |
| field | `x*=x/R0`, `Ux*=(Ux−U_ring)/U_ring`, `Ur*=Ur/U_ring` | once, at ingestion |
| `St` (per bubble) | `d²·U_ring/(18 ν R0)` — density-free; density lives in `R` | from escape CSV (`d=2·R_world`) |
| `R` | `2ρ_f/(2ρ_p+ρ_f) ≈ 2` | per ring |
| `Fr` (per ring) | `U_ring·√(ρ_f/(Δρ g R0))` (§4a) | per ring |

The user's `√(ρπ²a_eq²U/(Δρ v_b g))` Froude form (§4b) is **non-dimensionless**
and disabled in `constants.froude_number_proposed`. Self-check
`St/Fr² == v_t/U_ring` runs as an assertion. The old solver's `1/cosθ,1/sinθ`
revolve (singular on the axes) is replaced by the corrected `y/r, z/r` chain rule
with an on-axis guard in `build_field.Field3D`.

## Layout

```
src/
  constants.py       St / R / Fr groups + §6 self-checks
  digiflow_io.py     PIVFrames contract; load_piv (.dfi) + load_escape_csv   [A]
  synthetic.py       synthetic translating Lamb-Oseen vortex-ring field
  frame_transform.py estimate_Uc (vorticity centroid / core fit), QC         [B]
  averaging.py       time_average, smooth, scale-separation report           [C]
  nondimensional.py  the ONE conversion boundary + §6 assertions
  build_field.py     fold, nondimensionalize, revolve→3D, solver-format cache [D]
  seeding.py         fit_core_ellipse, seed_positions, sample_stokes         [E]
  advect.py          solver dimensionless RHS, escape event, pool            [F]
  diagnostics.py     escape stats, sanity/threshold/distribution plots       [G]
drivers/
  preprocess_station.py   raw PIV → cached dimensionless field (per station)
  run_simulation.py       cached field → seed → advect → escape stats
run_synthetic_demo.py     one-shot end-to-end demo
```

## Real data

`.dfi` stacks are read **in place** from `/mnt/d/...` (never copied into the
repo). Station→folder→coord map is in `build_plan.md` §2A. Example (10D, Up=200):

```bash
.venv/bin/python drivers/preprocess_station.py \
    --station-dir /mnt/d/Users/zl483/highspeedcamera/bonus_test_11/Camera_1 \
    --coord-file  /mnt/d/Users/zl483/highspeedcamera/april_bonus_mapping.csv \
    --fps 200 --station station_10D

.venv/bin/python drivers/run_simulation.py \
    --field fields/station_10D \
    --escape-csv data/escape/bubbles_200_l1.csv --workers 8 --gravity
```

The cache under `fields/<station>/` is written in the original solver's on-disk
format (`x.npy, y.npy, Ux.npy, Uy.npy, dUxd*.npy, dUyd*.npy, geometry.npy,
interp_functions.pkl`) so `../advect_bubbles_3D_eval.py` can load it by pointing
its `path` there — plus `field.pkl` / `mean_field.pkl` for this pipeline.

## Ensemble analysis & deliverables

```bash
.venv/bin/python drivers/run_all_stations.py     # 6 fields + escape summary
.venv/bin/python drivers/run_ensembles.py        # multi-seed sweeps + populations
.venv/bin/python drivers/plot_ensembles.py        # figures from ensembles.npz
.venv/bin/python drivers/make_movie.py            # 3D advection mp4
```

Figures (`outputs/`): `figA_fate_breakdown` (captured / buoyant / advective per
condition), `figB_buoyant_dcrit` (critical detrainment size vs station),
`figC_volume_vs_time` (trapped-volume time development), `figD_governing_relation`
(`St_crit ≈ W*·Fr²`, `W*≈2.1`), `bubble_advection.mp4`.

**Escape mode matters (FOV truncation).** Each escape is tagged by the FOV wall
it crosses (`advect._classify_exit`). The PIV window is smaller than the ring's
recirculation atmosphere, so a still-orbiting bubble can exit an **axial** wall
(`x_min/x_max`) — that is an FOV-truncation artifact, *not* physical
detrainment. Physical (buoyant) detrainment leaves through the **top radial**
wall (`r_top`, z>0, gravity is +z). All size-threshold analysis uses buoyant
escape only. A core-seeded tracer is trapped only when the core sits well inside
the FOV (120_5D, 120_10D, 200_15D); elsewhere advective loss dominates the raw
escape count, so trust `figB`/`figD` over raw escape fractions.

Result: critical detrainment size falls downstream for the 120 ring
(0.72→0.40→0.27 mm), matching the measured escaped-size trend; the buoyancy
number `St/Fr²` at threshold is ~2 across conditions (200_15D is the known
field-quality outlier).

## Open decisions still to confirm (build_plan §4)

PIV components (`u_θ ≈ 0`?); `U_ring` per-station vs one canonical per-ring for
St labelling (`dimension_conversion.md` §1); seeding measure (default
`arclength`); `ε⁺` cut (default = smallest observed diameter); whether
bubble–bubble merge is in scope; `Re_b ≳ 1` nonlinear-drag caveat (§7) for the
largest bubbles.
