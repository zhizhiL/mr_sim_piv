# pivRings — results & status report

PIV-driven Maxey–Riley model of bubble capture/escape in a vortex ring.
Six stations: piston speed 120 & 200 mm/s × downstream 5D/10D/15D (D = 40 mm).
Pipeline: dimensional PIV → dimensionless co-moving 3D field → Maxey–Riley
advection → capture/escape statistics. Conventions in `README.md` /
`dimension_conversion.md`.

---

## 1. Headline

A complete, reproducible pipeline runs end-to-end on all six real PIV stations
and on synthetic data, with multi-seed ensemble statistics and a 3D movie. The
**dominant physical limitation is the ring propagation speed `U_ring` and the
size of the PIV field of view relative to the ring's recirculation atmosphere**
— resolved/diagnosed below. With the physically-correct `U_ring`, **3 of 6
stations have a closed trapping atmosphere inside the FOV** (reliable for
capture/detrainment physics); the other 3 are FOV/field-limited.

---

## 2. The `U_ring` problem (and resolution)

`U_ring` sets the co-moving frame and the scales for `St`, `Fr`. Three estimators
were tried:

| station | centroid fit | Ux@core | **thin-ring** | piston |
|---|---|---|---|---|
| 120_5D  | +53.7 | +37.2 | **55.6** | 120 |
| 120_10D | +17.8 | +41.6 | **40.8** | 120 |
| 120_15D | +34.4 | −5.2  | **55.0\*** | 120 |
| 200_5D  | +117.3| +9.2  | **109.7** | 200 |
| 200_10D | +85.2 | +64.5 | **83.8** | 200 |
| 200_15D | +55.4 | +215.6| **83.8** | 200 |

- **Vorticity-centroid fit** (build_plan default): biased when diffuse wake
  vorticity moves differently from the core (120_10D → 17.8, clearly wrong).
- **Ux@core** (axial velocity at the vorticity core): unphysical on several
  (120_15D −5, 200_15D 216) — the detected core sits where the swirl doesn't
  null cleanly.
- **Thin-core ring formula** `U = Γ/(4πR)·(ln(8R/a) − ¼)` (from measured
  circulation Γ, ring radius R, core radius a): **adopted.** Physically
  principled, time-independent, agrees with the centroid where the centroid is
  reliable (120_5D, 200_5D, 200_10D), fixes the bad ones, and gives the
  expected **200 > 120 ordering at every station**.
  (\*120_15D's bump is an artifact of an anomalously small detected R = 14 mm —
  its core detection needs attention.)

---

## 3. Central finding — FOV vs. ring atmosphere

At the thin-ring `U_ring`, a pure tracer seeded at the core is trapped only for:

| | 5D | 10D | 15D |
|---|---|---|---|
| **120** | ✅ trap | ✅ trap | ❌ (R=14 artifact) |
| **200** | ❌ | ❌ | ✅ trap |

The non-trapping stations are **not** a `U_ring` error (adopting the principled
speed didn't change which trap). Their **co-moving streamlines do not close**
inside the FOV — the recirculation atmosphere is either weaker than the
through-flow or larger than the PIV window at the true propagation speed, so
bubbles advect through. Evidence: `outputs/streamlines_*.png` (120_5D closes;
200_10D passes straight through), and the core-trapped fractions above.

**Consequence for interpretation.** Each "escape" is tagged by which FOV wall it
crosses (`advect._classify_exit`): leaving through the **top radial wall**
(`r_top`, z>0, gravity is +z) is **physical buoyant detrainment**; leaving
through an **axial wall** (`x_min/x_max`) is **FOV-truncation loss** — the bubble
may still be orbiting in the ring. All size-threshold analysis uses buoyant
escape only. Reliable stations: **120_5D, 120_10D, 200_15D**.

---

## 4. Results (see figures in `outputs/`)

- **figA_fate_breakdown** — captured / buoyant / advective per condition.
- **figB_buoyant_dcrit** — critical buoyant-detrainment diameter vs station.
- **figC_uniform_volume_vs_time / figE_retention_vs_station** — fair
  cross-station retention with an IDENTICAL seeded population.
- **figD_governing_relation** — `St_crit ≈ W*·Fr²` (buoyancy threshold).
- **bubble_advection.mp4** — 3D movie (translucent donut + advecting bubbles).
- diagnostics: `seed_locations.png`, `core_vs_stagnation.png`, `streamlines_*.png`.

### Reliable-station results (trapping atmosphere closes)

| station | U_ring (mm/s) | Fr | d_crit (mm) | common-pop retention |
|---|---|---|---|---|
| 120_5D  | 55.6 | 0.126 | **0.96** | **0.86** |
| 120_10D | 40.8 | 0.092 | **0.40** | **0.06** |
| 200_15D | 83.8 | 0.189 | **0.80** | 0.72 |

(FOV-limited stations 120_15D / 200_5D / 200_10D escape ~100% advectively;
their `d_crit` (0.31 / 0.59 / 0.45) is biased and shown as open markers in figB/figD.)

### What makes physical sense
- **`U_ring` 200 > 120** at every station (thin-ring), decreasing downstream.
- **120-ring retention falls monotonically downstream** with a fair common
  population: **0.86 → 0.06 → 0** across 5D→10D→15D (figE) — seeding bias removed.
- **120-ring `d_crit` falls downstream** (0.96 → 0.40 mm) — bigger bubbles
  required to detrain near the fresh, stronger ring; matches the measured
  escaped-size trend (5D≈1.5 mm → 15D≈0.34 mm).
- **Buoyant detrainment scales as `St/Fr²`** (rise speed ÷ ring speed): the 3
  reliable stations give `St_crit ≈ W*·Fr²`, `W* ≈ 4.8` (figD).
- **Volume budget**: for the only escape CSV spanning from ~5D
  (`bubbles_120_l3`), escaped bubble volume recovers ~79% of the 40 µL loading.

### What does not (yet) make sense — and why
- **Half the stations don't trap** → FOV smaller / averaged vortex weaker than
  the ring atmosphere at the true `U_ring`. Real measurement limitation.
- **`d_crit` only physical for trapping stations**; advection-dominated stations
  give a biased threshold (measured from pass-through bubbles).
- **120_15D R = 14 mm** anomaly → core detection picked too-close cores.

---

## 5. Limitations

1. `U_ring` uncertainty is the dominant lever (St ∝ U, Fr ∝ U). Thin-ring is
   principled but depends on Γ, R, a, which carry PIV uncertainty.
2. **PIV FOV < ring atmosphere** for the fast/fresh rings → advective truncation.
3. Seeds ride the **vorticity core**, ~0.2 R₀ outboard of the velocity
   recirculation eye (`core_vs_stagnation.png`) → some start near the separatrix.
4. PIV outliers (spurious vectors to ~2000 mm/s) — handled by MAD rejection.
5. Window selection per movie; long windows need spatial registration (added).

---

## 6. Next steps (prioritized)

1. **Enlarge the effective domain**: extrapolate the co-moving field beyond the
   FOV (taper to the −U_ring far field) or mosaic adjacent windows, so orbits
   aren't truncated and 200_5D/10D can be evaluated.
2. **Stream-function atmosphere**: define capture by the closed-ψ separatrix
   (and `U_ring` by its rear axial stagnation) instead of tracer-in-FOV —
   removes the FOV confound and gives a second, independent `U_ring`.
3. **Seed on the recirculation eye** (velocity stagnation), not the vorticity
   core — removes the ~0.2 R₀ seeding offset.
4. **Fix 120_15D core detection** (R = 14 mm artifact); re-pick its window.
5. **Independent ring-speed** from the raw high-speed video (track the front)
   to cross-check the thin-ring `U_ring`.

---

## 7. How to reproduce

```bash
.venv/bin/python pivRings/drivers/cache_means.py        # once: registered means
.venv/bin/python pivRings/drivers/build_from_cache.py    # fields w/ thin-ring U_ring
.venv/bin/python pivRings/drivers/run_ensembles.py       # sweeps + populations
.venv/bin/python pivRings/drivers/run_uniform.py         # fair common-population
.venv/bin/python pivRings/drivers/plot_ensembles.py      # figures
.venv/bin/python pivRings/drivers/make_movie.py --field pivRings/fields/120_5D \
    --escape-csv pivRings/data/escape/bubbles_120_l1.csv --station-D 200
```
