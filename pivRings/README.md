# pivRings — PIV-driven Maxey–Riley bubble-in-vortex-ring model

Implements the pipeline in [`../build_plan.md`](../build_plan.md): take a
**measured PIV carrier field** at downstream stations, process it into a
quasi-steady axisymmetric 3D field, and advect Maxey–Riley bubbles through it
to predict capture/escape statistics.

This is a **scaffold wired end-to-end against a synthetic vortex-ring field**
so the whole protocol can be reviewed before the real `.dfi` data and the
validated solver are plugged in. Units throughout: **mm, mm/s, s**; fluid =
water at 20 °C, particle = hydrogen gas bubbles (`src/constants.py`).

## Quick start

```bash
pip install -r requirements.txt
python run_synthetic_demo.py        # full A→G protocol on synthetic data
```

Outputs land in `outputs/`:

| file | milestone | shows |
|------|-----------|-------|
| `M1_streamlines.png`    | M1 | co-moving meridional streamlines (closed ring core) |
| `M2_seeds.png`          | M2 | seed ring placed from the fitted core ellipse |
| `M3_escape_vs_size.png` | M3 | escape fraction vs bubble diameter (capture threshold) |
| `M3_escape_sizes.png`   | M3 | simulated vs measured escape-size distribution |

## Layout

```
src/
  constants.py       physical constants + unit convention; St & Fr helpers
  digiflow_io.py     PIVFrames contract; load_piv (.dfi) + load_escape_csv     [Stage A]
  synthetic.py       synthetic translating Lamb-Oseen vortex-ring field
  frame_transform.py estimate_Uc (core linear fit), to_comoving, residual QC   [Stage B]
  averaging.py       time_average, smooth, scale-separation report             [Stage C]
  build_field.py     fold halves, gradients, revolve→3D (axis guard), cache    [Stage D]
  seeding.py         fit_core_ellipse, seed_positions, sample_stokes           [Stage E]
  advect.py          Maxey-Riley RHS, escape event, multiprocessing pool       [Stage F]
  diagnostics.py     escape stats, sanity/threshold/distribution plots         [Stage G]
drivers/
  preprocess_station.py   raw PIV → cached 3D field (per station)
  run_simulation.py       cached field → seed → advect → escape stats
run_synthetic_demo.py     one-shot end-to-end demo on synthetic data
```

## Running on the real experimental data

The DigiFlow folders and pixel→world coordinate maps are documented in
`build_plan.md` §2 Stage A. Per station:

```bash
python drivers/preprocess_station.py \
    --station-dir /mnt/d/Users/zl483/highspeedcamera/bonus_test_11/Camera_1 \
    --coord-file  /mnt/d/Users/zl483/highspeedcamera/april_bonus_mapping.csv \
    --fps 200 --station station_10D

python drivers/run_simulation.py \
    --field fields/station_10D \
    --escape-csv data/escape/bubbles_120_l1.csv \
    --workers 4 --gravity
```

(`preprocess_station.py --synthetic` runs the same path on a generated field.)

## Swap seams (what's a placeholder)

* **`advect.mr_rhs`** — a clean, self-contained Maxey–Riley bubble model so the
  demo runs. Replace it with the validated RHS from
  `newCodes_norburyRings/advect_bubbles_3D_eval.py` when available, keeping the
  `Field3D.velocity_and_gradient` interface, the escape event, and the pool
  driver. Bubble drag is **stiff** for small bubbles, so the integrator
  defaults to `LSODA`.
* **`constants.froude_number`** — implemented verbatim from the plan; as written
  the group is not dimensionless (flagged in the docstring) and should be
  reconciled against the `.tex` writeup / old solver before production use.

## Open decisions still to confirm (build_plan §4)

PIV components (is `u_θ ≈ 0`?); uniform vs spatially-varying `U_c`; seeding
measure (default `arclength`); `d→St` calibration & log-normal cut `ε⁺`
(default = smallest observed diameter); non-dimensionalisation scales;
whether bubble–bubble merge (`merge.py`) is in scope.
