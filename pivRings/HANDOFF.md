# Handoff notes — pivRings (bubble-in-vortex-ring, Maxey–Riley)

Continuation notes for the next thread. Full detail lives in `REPORT.md` (esp.
§7a–§7f) and `build_plan.md`. This file is the "where we are / what's next".

---

## DONE — separatrix-based "escape" definition (FOV-independent) ✅

Implemented and validated. The §7d FOV-rectangle caveat is now resolved with a
**co-moving Stokes streamfunction separatrix** escape test (REPORT §7e, now
"implemented").

**What was built:**
- `src/separatrix.py` — least-squares / Helmholtz-projected Stokes `ψ` (matches
  both velocity components, the principled fix for non-solenoidal PIV), separatrix
  extraction (largest closed `ψ`-loop enclosing the core, **closed along the
  axis** so the recirculation bubble's lower boundary is `r=0`), point-in-polygon
  `Separatrix` object, and `save/load`/`load_or_compute` caching to
  `<field>/separatrix.npz`.
- `advect.py` — new `escape="separatrix"` mode (+ `sep_poly` arg, threaded through
  the pool). Escape = the bubble **sustainedly** leaves the polygon (final state
  outside, or ran out the FOV); `t_escape` = start of the last outside-run; the
  FOV event is kept as an outer safety bound. `exit_face` is `"separatrix"` or a
  FOV wall (atmosphere>FOV fallback). `BubbleResult` docs updated.
- `drivers/run_separatrix.py` — per-field ψ + separatrix figure overlaying the
  forward-FTLE ridge, empirically-trapped fluid, core ellipse, and settled
  bubbles (`outputs/<field>_upper_separatrix.png`); caches the npz.
- `drivers/compare_escape_defs.py` — re-runs `d_crit` with both escape definitions
  (`outputs/escape_def_comparison.{png,json}`).

**Key result: `d_crit` is escape-definition-independent for all six adopted
fields** (FOV ≡ separatrix to 2 d.p.; see REPORT §7e table). So the FOV-truncation
caveat does **not** bias the capture statistics — the separatrix *confirms* the
existing numbers. `200_15D_scaled` is the only genuinely FOV-truncated atmosphere
(area 11.8, auto-flagged `fov_truncated=True`); it still matches via the fallback.

**Validation checks that held:**
- inside-separatrix ⟹ trapped is **exact** (no inside-polygon tracer ever leaves
  the FOV over T*≤40).
- forward-FTLE ridge coincides with the dividing streamline (see the figures).
- per-size agreement with the FOV flag is exact across d=0.2…2.5 once the polygon
  is axis-closed (the axis-closure was the crucial fix — see §7e implementation).

**Uniform-seeding deliverables regenerated under separatrix.** `run_uniform_size.py`
now has `--escape {fov,separatrix}`; separatrix runs land in
`outputs/separatrix_escape/` (FOV canonical figures kept). One (x,r) atmosphere is
shared by both cores (axisymmetric field). Remaining-% / retained-volume endpoints
match the FOV runs (set by d_crit); the volume-retention curve decays a little
earlier (separatrix crossing precedes FOV-exit). Buoyant/advective detrainment is
re-tagged geometrically from the crossing location (`advect._separatrix_exit`).

**Possible follow-ups (not blocking):**
- For `200_15D_scaled` (atmosphere>FOV) a larger-FOV rebuild would let the
  separatrix close inside the domain; today it's handled by the flagged fallback.
- Movies still freeze on the FOV box (`render_movie`), not the separatrix — wire
  the polygon through if a separatrix-consistent movie is wanted.

---

## ADOPTED FIELD PER STATION (use these dirs, not the raw thin-ring builds)

| station | field dir | how it was built | U_ring | d_crit |
|---|---|---|---|---|
| 120_5D  | `fields/120_5D`        | native time-avg            | 55.6 | 1.04 |
| 120_10D | `fields/120_10D`       | native time-avg            | 40.8 | 0.31 |
| 120_15D | `fields/120_15D_geom`  | isotropic magnify R 15→23  | 25.0 | 0.24 |
| 200_5D  | `fields/200_5D_sf`     | single representative frame| 109.7| 0.78 |
| 200_10D | `fields/200_10D_sf`    | single representative frame| 83.8 | 0.56 |
| 200_15D | `fields/200_15D_scaled`| 1-frame (a_eq fix) + strength×0.33 | 27.7 | 0.46 |

Both series now decay smoothly downstream and the ring geometry (R, a_eq) is
continuous. Comparison figures: `outputs/compare_capture_vs_size.png`,
`compare_volume_vs_time.png`. The all-six streamline/atmosphere figure:
`outputs/atmosphere_streamlines.png` (re-centred on core, common axes).

---

## KEY CONVENTIONS & FINDINGS (so you don't relearn them)

- **Scales:** `R0=20 mm` (hard-coded), `τ_f = R0/U_ring`, `St = d²U/(18νR0)`,
  `Fr = U√(ρ/(Δρ g R0))`, `W* = St/Fr² ∝ 1/U` (buoyant rise ÷ ring speed).
- **Strength scaling ≡ U_ring scaling.** The dimensionless field is scale-
  invariant: scaling the ring's circulation by `s` keeps the dimensionless field
  (and `a_eq`) fixed and only lowers `U_ring` in St/Fr (so `W*∝1/s` rises,
  `d_crit` falls). NB: `scan_dcrit_uring.py` *rebuilds* the field (changes the
  co-moving FRAME, ≠ strength); `scan_strength.py` does the TRUE strength scale.
- **"Open atmosphere" was a time-averaging smear**, not real: the meandering
  far-downstream vortex blurs under averaging (even after centroid registration),
  so the co-moving atmosphere fails to close at the thin-ring speed. **Single
  instantaneous frames close at the physical thin-ring U** (→ the `_sf` fields).
- **Attractor (§7b):** trapped bubbles sit at a genuine stable 3-D Maxey–Riley
  fixed point (the `y=0` plane is invariant; the analysis is 3-D, not 2-D).
  Capture threshold is a basin-of-attraction effect; `W*≈8` at 120_5D.
- **`U_ring` by atmosphere closure (§7c):** `run_ftle.py --uring-sweep` — closure
  speed reproduces the §3 trapping pattern; thin-ring overestimates U for non-
  thin cores.
- **15D core artifacts (§7f):** 200_15D `a_eq`=14.6 was a smear (→7.6 via single
  frame); 120_15D `R`=14 is genuine/persistent (→ isotropic magnification, R
  fixed, streamline topology preserved, `d_crit` unchanged).
- **Numerics (§7a):** LSODA, `rtol=1e-6`; verified converged (`verify_timestep.py`,
  error ~2 ppm, tolerance-limited not step-limited). Re_b~10³ for big bubbles →
  Stokes-drag out of regime (a model caveat, not numerics).

---

## TOOLS BUILT THIS THREAD (all in `drivers/` unless noted)

`run_uniform_size.py` (build-plan seeding: core-surface, uniform-count sizes, both
cores, d_crit/basin/residual/movie) · `run_ftle.py` (fluid+inertial FTLE,
`--uring-sweep` closure) · `src/ftle.py` · `investigate_attractor.py` ·
`scan_dcrit_uring.py` (frame sweep) · `scan_strength.py` (true strength scale) ·
`build_single_frame_field.py` · `diag_15D_core.py` · `build_warped_field.py`
(`--uniform` isotropic / default radial warp) · `plot_uniform_comparison.py` ·
`plot_atmosphere.py` · `plot_closure_mismatch.py` · `test_closure_remedies.py` ·
`test_frame_closure.py` · `run_separatrix.py` (separatrix figure+cache) ·
`compare_escape_defs.py` (FOV vs separatrix d_crit). `src/build_field.py` gained a
`core={both,upper,lower}` selector; `src/separatrix.py` (new) + `advect.py`
`escape="separatrix"` mode implement the FOV-independent escape test.

## SECONDARY / OPEN ITEMS (not blocking the separatrix task)
- 200-series ring radius: 200_5D R=18.6 is the small one (200_10D/15D ≈24); the
  user said leave the 200 geometry for now (monotonic downstream expansion).
- Residual `200_5D d_crit (0.78) < 120_5D (1.04)` — mild, robust to frame choice.
- Reproduce pipeline: see REPORT §7 "How to reproduce" + the adopted-field table
  above (the `_sf`/`_scaled`/`_geom` fields are the current canonical ones).
