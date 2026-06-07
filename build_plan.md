# Build Plan — PIV-driven Maxey–Riley bubble-in-vortex-ring model

Execution spec for Claude Code. Goal: reuse the validated Maxey–Riley
advection/collision machinery from `newCodes_norburyRings`, but replace the
analytical Norbury streamfunction with a **measured PIV carrier field** at 3
downstream stations, processed into a quasi-steady, axisymmetric 3D field.

Physics rationale and assumptions live in `model_writeup_sections.tex`. This
file is the *how-to-build*. Keep them in sync.

---

## 0. Relationship to the previous repo (reuse map)

Pull these directly from `newCodes_norburyRings` — do **not** rewrite from scratch:

| Reuse | Source file | What to keep |
|-------|-------------|--------------|
| Maxey–Riley ODE + RK45 + multiprocessing | `advect_bubbles_3D_eval.py` | `solve_ivp_active*`, `advect_bubbles*`, the planar→Cartesian chain-rule block. Swap only the field-loading source. |
| Axisymmetric revolve | `norbury_family_3D.ipynb` | meridional→3D revolution, axis handling, "donut" core surface for plotting. |
| Interpolator caching | `norbury_family.ipynb` (cells 16–17) | save `Ux/Ur` + 4 gradients as `.npy`; pickle `RectBivariateSpline` interpolators. |
| Collision/merge (optional) | `merge_bubbles_3d_v2.py` | staggered A/B grids + KDTree + merge/bounce/stick, if bubble–bubble interaction is in scope. |
| Movie | `make_movie.py` | frame→mp4. |

**Carry over the known bugs to fix, not inherit:** guard the `1/cos θ`,
`1/sin θ` singularities at `θ=0, π/2`; settle a single field path convention
(the old repo drifted across `velocity_fields/…`, `velocity_results/…`).

---

## 1. Proposed repo structure

```
pivRings/
├── data/
│   ├── piv/                 # raw PIV per station (format TBD)
│   ├── escape/              # escape-size .csv files
│   └── core_fit/            # example elliptical-fit file (to be uploaded)
├── fields/                  # generated: processed mean fields + interpolators
│   ├── station_1/  station_2/  station_3/
├── src/
│   ├── digiflow_io.py            # load/parse PIV + escape csv
│   ├── frame_transform.py   # convection-velocity estimate + subtraction
│   ├── averaging.py         # window time-average + spatial smoothing
│   ├── build_field.py       # gradients, revolve, cache interpolators
│   ├── seeding.py           # ellipse fit → placement; size → St (log-normal)
│   ├── advect.py            # MR solver (adapted from old advect_bubbles_3D_eval)
│   ├── merge.py             # optional, from merge_bubbles_3d_v2
│   └── diagnostics.py       # capture/escape stats, validation plots
├── drivers/
│   ├── preprocess_station.py    # raw PIV → cached 3D field, per station
│   └── run_simulation.py        # seed → advect → (merge) → record
├── outputs/
├── requirements.txt
└── README.md
```

---

## 2. Stage-by-stage spec

### All measurements are in the unit of mm; 
### all physical quantities should take 20degC for water as fluid and hydrogen gas bubbles as particle

### Stage A — PIV ingestion (`digiflow_io.py`)
- `load_piv(station_dir) -> frames` returning a stack of meridional
  `(u_x, u_r)` fields on a common grid `(x, r)`, plus the per-frame timestamps.
  The locations for velocity data:
    - $U_p=120$ mm/s, no preloaded bubbles:
      -- 5D downstream: bonus17; folder_path = "/mnt/d/Users/zl483/highspeedcamera/bonus_test_17" ; coord_name = "coord_up"
      -- 10D downstream: bonus01; folder_path = "/mnt/d/Users/zl483/highspeedcamera/bonus_test_01" ; coord_name = "april_vorticity"
      -- 15D downstream: bonus27; folder_path = "/mnt/d/Users/zl483/highspeedcamera/bonus_test_27" ; coord_name = "coord_down"
    - $U_p=200$ mm/s, no preloaded bubbles:
      -- 5D downstream: bonus15; folder_path = "/mnt/d/Users/zl483/highspeedcamera/bonus_test_15" ; coord_name = "coord_up"
      -- 10D downstream: bonus11; folder_path = "/mnt/d/Users/zl483/highspeedcamera/bonus_test_11" ; coord_name = "april_bonus"
      -- 15D downstream: bonus25; folder_path = "/mnt/d/Users/zl483/highspeedcamera/bonus_test_25" ; coord_name = "coord_down"
- `load_escape_csv(path) -> diameters` for the size distribution fit.
- **Open:** PIV file format (`.vc7`/`DaVis`, `.mat`, `.npy`, plain CSV?).
  The reader is already provided under cwd digiflow.py that reads in one single frame data and returns u, v, omega (see ellipseFit_example.ipynb)
- Locate the **ring axis** from reflection symmetry; record `r=0`. Optionally
  fold the two half-planes to enforce symmetry and reduce noise.

### Stage B — Reference frame (`frame_transform.py`)
- `estimate_Uc(frames, method) -> U_c` per station. Implement at least:
   Linear fit the x-displacement (average of upper and lower core position) with time (should be somewhat constant) throughout the frame
- `to_comoving(frames, U_c) -> frames'` subtracts `U_c e_x`.
- Output the **residual unsteadiness** after subtraction as a QC metric.
- **Open:** uniform `U_c` vs spatially-varying mean — default uniform; expose
  as a flag.

### Stage C — Averaging (`averaging.py`)
- `time_average(frames') -> (Ux_bar, Ur_bar)` over `T_win`.
- `smooth(field, sigma)` spatial low-pass **before** differentiation
  (PIV gradient noise).
- Emit the scale-separation report: turnovers-per-window, fractional change
  of Γ / R₀ / U_c across the window and across stations (feeds the
  quasi-steady justification — see the .tex).

### Stage D — Build 3D field (`build_field.py`)
- Compute `dUxdx, dUxdr, dUrdx, dUrdr` on the meridional grid.
- Revolve to 3D + chain-rule conversion (reuse old code). **Guard θ
  singularities.**
- Cache per station: `Ux.npy, Ur.npy`, the 4 gradients, the `(x, r)` grid,
  the core geometry, and a pickled tuple of `RectBivariateSpline`
  interpolators — **one consistent path convention**, e.g.
  `fields/station_k/`.

### Stage E — Seeding (`seeding.py`)
- `fit_core_ellipse(field) -> ellipse_params` from a vorticity iso-contour or
  the closed co-moving streamline (mirror the example fit file once uploaded).
- `seed_positions(ellipse, n, measure) -> xyz` with `measure ∈
  {arclength, area, annulus}` (default arclength). Revolve seed ring in θ for
  3D.
- `sample_stokes(n, csv_path) -> St` : fit log-normal to escape diameters,
  draw sizes from `ε⁺` up to the largest observed escape size, map
  `d → St`: $St = \tau_p/\tau_f$ where $\tau_p = \frac{\rho_p*d_b^2}{18 \mu}$ with $d_b$ being individual bubble size;
   and $\tau_f = R_0 / U_ring$ with R_0 hard-corder to be 20 mm.
  

### Stage F — Advection (`advect.py`)
- Adapt `advect_bubbles_3D_eval.py`: keep the ODE RHS and `multiprocessing`
  pool; replace the field loader to read `fields/station_k/`.
  `Fr` : $Fr=\sqrt{\frac{\rho * \pi^2 * a_{eq}^2 *U_ring}{\nabla \rho v_b g}}$ where $v_b$ is the individual bubble volume and $g$ the gravity constant;
  $a_{eq}$ is the equivalent ellipse axis length after the elliptical fit, to describe the vortex core radius.
- Keep the with/without-gravity variants. `Du/Dt = (u·∇)u` (quasi-steady, no
  `∂u/∂t`).
- Add an **out-of-domain / escape** test (bubble leaves the PIV FOV or the
  bubble atmosphere) and tag escape time + size — this is the headline
  observable.

### Stage G — Diagnostics (`diagnostics.py`)
- Escape-size threshold and distribution vs the measured escape `.csv`.
- Optional A/B run: same seeding on the **inviscid Norbury field** vs the PIV
  field, to isolate viscous detrainment (the ε→0 near-tracer comparison).
- Station-to-station comparison as an empirical quasi-steady error bound.

---

## 3. Driver flow

```
preprocess_station.py  (×3 stations):
    load_piv → estimate_Uc → to_comoving → time_average → smooth
    → gradients → revolve(3D) → cache interpolators

run_simulation.py  (per station, per parameter set):
    load cached field → fit_core_ellipse → seed_positions + sample_stokes
    → advect (pool) → [merge] → record trajectories/escape → diagnostics
```

---

## 4. Open decisions to surface to the user (do not guess silently)
Some of them have been addressed in this markdown write up.

1. PIV file format / which velocity components are available (2C vs 2D3C; is
   `u_θ` truly ~0?).
2. Mean-flow / `U_c` estimator and whether it is uniform or spatially varying.
3. Seeding measure ("uniformly around the core" = arclength / area / annulus).
4. `d → St` calibration and the log-normal parameters (await escape `.csv` +
   ellipse example file).
5. Non-dimensionalisation scales (`R`, `Fr`, length/velocity) consistent with
   the experiment.
6. Is bubble–bubble collision (`merge.py`) in scope, or single-particle
   statistics only?

---

## 5. Milestones

- **M1** One station: raw PIV → cached, smoothed, co-moving 3D field +
  sanity plots (streamlines closed in ring frame).
- **M2** Seeding from ellipse fit + log-normal St from the escape csv.
- **M3** Advection reproduces qualitative capture; escape diagnostic working.
- **M4** All three stations; quasi-steady error bound report.
- **M5** PIV-vs-inviscid A/B comparison; escape-size validation against data.

---

## 6. Dependencies
`numpy scipy matplotlib opencv-python` (+ PIV-format reader TBD, e.g.
`lvpyio`/`ReadIM` for DaVis, or `h5py`/`scipy.io` for `.mat`). Python 3.
Add `requirements.txt` and a `.gitignore` for `fields/` and `outputs/`.
