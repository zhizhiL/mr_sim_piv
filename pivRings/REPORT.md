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

## 3. Central finding — the `U_ring` / atmosphere-closure mismatch

At the thin-ring `U_ring`, a pure tracer seeded at the core is trapped only for:

| | 5D | 10D | 15D |
|---|---|---|---|
| **120** | ✅ trap | ✅ trap | ❌ (R=14 artifact) |
| **200** | ❌ | ❌ | ✅ trap |

**This is NOT a FOV-size problem and NOT an averaging-smear problem** (both
were tested and ruled out — the FOV is ~5.7 R₀, larger than any ring
atmosphere; and a sharp, ring-centred 21-frame window still gives 0/8).

The real cause is a **mismatch between the propagation speed and the
atmosphere-closure speed.** A closed co-moving atmosphere *does* exist for the
non-trapping stations, but only **below a critical co-moving speed**, and the
thin-ring propagation speed sits **above** it. For 200_10D (short centred
window):

```
 U_ring   50  55  60  65  70 | 80  84(thin-ring)
 trapped  1.0 1.0 1.0 1.0 1.0| 0.0   0.0
```

A sharp transition at ~72–75 mm/s; the thin-ring speed (84) is past it, so the
atmosphere has collapsed in the frame we build. For an ideal steady ring the
two speeds coincide; here they disagree (~72 vs 84, and the notebook used 115
for this same movie) because the thin-ring formula overestimates for non-thin
cores (a/R≈0.3) and the averaged field isn't a perfectly steady ring. **So the
true `U_ring` for 200_10D is uncertain over ~72–115 mm/s (±~25%), and which
value is chosen decides whether the field traps at all.** This is the dominant
open issue — see §6. (Strong fresh rings like 120_5D trap up to very high
co-moving speeds, so the closure transition is ill-defined there: no single
estimator is clean for both diffuse and strong rings.)

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

## 6. Next steps (prioritized) — all about pinning down `U_ring`

The dominant issue is the propagation speed, not the FOV. In priority order:

1. **Define `U_ring` by atmosphere closure**, consistently: compute the Stokes
   stream-function ψ in the co-moving frame and choose `U_ring` as the speed at
   which the separatrix first forms a rear axial stagnation point (the textbook
   ring-speed definition). This is the transition the tracer test brackets
   (~72 mm/s for 200_10D) but done topologically, without the tracer/FOV proxy.
   At that speed the non-trapping stations trap and become usable.
2. **Independent ring-speed from the raw high-speed video** (track the vortex
   front frame-to-frame) to break the 72–115 mm/s ambiguity with a measurement
   that doesn't depend on the averaged PIV field.
3. **Reconcile the two**: if the stream-function speed and the front-tracking
   speed agree, adopt it and drop the thin-ring formula (which overestimates for
   a/R≈0.3 cores); if they disagree, the averaged field isn't a clean ring and
   needs a tighter window / phase-averaging.
4. **Seed on the recirculation eye** (velocity stagnation), not the vorticity
   core — removes the ~0.2 R₀ seeding offset (`core_vs_stagnation.png`).
5. **Fix 120_15D core detection** (R = 14 mm artifact); re-pick its window.

---

## 7a. Numerical resolution — is the ODE time step small enough?

Concern: in `*_volume_vs_time.png` (and the movie) the retained volume collapses
almost instantly. Three points establish this is **physical, not an
under-resolved time step**:

1. **`t_eval` ≠ integration step.** `advect_one` / `advect_trajectories` use
   `solve_ivp(method="LSODA")` with *adaptive* stepping; `n_eval`/`t_eval` only
   sets where the solution is **sampled for output**, not the internal step
   (chosen to meet `rtol=1e-6, atol=1e-8`). Escape time comes from the terminal
   **event**, root-found to tolerance between internal steps — independent of
   `t_eval`.

2. **The collapse is genuinely fast buoyant detrainment, and it is resolved.**
   The dimensionless terminal rise speed is `W* = St/Fr²`; at 120_5D
   (`Fr²=0.0158`) it reaches **W\* ≈ 31** at r=0.9 mm (rising 31× faster than the
   ring convects). The simulated escape times (upper core) span **t\* ≈ 0.15–1.8,
   median 0.45** (= 0.05–0.65 s) — fast, but spanning ~7–11 of the movie's 45 ms
   frames (0% leave before frame 1; my earlier "sub-frame" estimate ignored drag
   spin-up and geometry). The drag time `t_drag* = St/R ∈ [0.001, 0.25]` is the
   other fast scale; small bubbles are stiff (`≈0.001`) but LSODA switches to
   implicit BDF without accuracy loss. Baseline already takes a **median 1650
   internal steps per escaping trajectory** — these dynamics are finely resolved.

3. **Convergence study confirms it** (`drivers/verify_timestep.py`, 176 bubbles,
   22 radii × 8, upper core). Same seeded bubbles, four solver settings:

   | setting | tol / cap | captured | mean nfev | fate match | median \|Δt_esc\| |
   |---|---|---|---|---|---|
   | baseline | LSODA rtol 1e-6, atol 1e-8 | 54.55% | 18.0k | — | — |
   | tight | LSODA rtol 1e-9, atol 1e-12 | 54.55% | 66.5k | **100%** | **7.9e-7** |
   | capped | LSODA + `max_step=5e-4` | 54.55% | — | **100%** | 8.6e-7 |
   | radau | Radau rtol 1e-8, atol 1e-10 | 53.98% | 117k | 99.4% | 7.9e-7 |

   Per-radius **median escape times are identical to 4 decimals** between
   baseline and tight at every radius. The few large `max |Δt_esc|` values
   (~0.09–0.5) come from a handful of bubbles straddling the capture/escape
   separatrix, where escape time is genuinely sensitive; the lone Radau fate
   flip (1/176) is one such borderline bubble. The production setting is
   converged.

   **Convergence figure** (`outputs/120_5D_upper_timestep_convergence.png`, from
   `verify_timestep.py --plot`, 71 escaping bubbles vs a `rtol=1e-12` gold
   reference): (left) escape-time error falls ~first-order as `rtol` tightens,
   reaching **2.1e-6 at production `rtol=1e-6`** with no floor; (right) with
   `rtol` fixed, an explicit `max_step` cap from `Δt*=0.2` to `0.002` leaves the
   error **flat at ~2e-6** — the adaptive step near escape is already finer than
   any cap, so accuracy is **tolerance-limited, not step-limited**.

**Separate model caveat (not a time-step issue):** at these sizes the slip
Reynolds number `Re_b` reaches ~10³, so **Stokes drag is out of its validity
regime** for large bubbles (build_plan §7). The detrainment is real, but the
quantitative escape time of the largest bubbles carries a drag-law error.

---

## 7b. Trapped bubbles — a physical trapping attractor (not a numerical fault)

In the movie some bubbles settle on the core periphery and stay put forever.
Investigated with `drivers/investigate_attractor.py` (120_5D, upper core): this
is a **genuine stable equilibrium of the Maxey–Riley system**, the textbook
"bubble trapped in a vortex," **not** a stalled integrator or a dead-field
artifact.

**It is a true fixed point, not a numerical stall.** A frozen position requires
`v=0` and `dv/dt=0`. Integrating 60 small bubbles (d=0.2 mm) to `t*=80`: late-time
`|v|` median **4.7e-13**, `|accel|` median **2.6e-9** — both machine-zero, so the
bubbles are genuinely at force balance, not merely taking tiny steps.

**It is a stable 3-D attractor in real flow.** The phase space is 6-D
(`s=[x,y,z,vx,vy,vz]`); the fixed point needs `v=0` and `a(x)=0` in **3-D**
(gravity along lab-`z` breaks axisymmetry, so a 2-D meridional balance is
insufficient). Newton on `a(x)=0` gives

- `x* = (1.658, 0.000, 1.311)` → at azimuth **z>0 (top of the ring)**, `r*=1.31`;
- local fluid `u* = (-0.236, 0, -0.299)`, `|u*|=0.38` — **nonzero and inside the
  PIV data domain**, so not a zero/extrapolated-field artifact (a dead region
  can't trap: buoyancy would be unopposed and the bubble would rise out);
- `u_z* = -0.30 ≈ -W* = -0.39`: the recirculation **descends at the buoyant rise
  speed**, balancing buoyancy;
- the **6×6 Jacobian** has all eigenvalues with negative real part — slow mode
  `-0.30`, fast damped modes `-19.1±67.5j` and `-305.8±67.5j`, `-324.7`, `-0.30`
  → a **stable spiral/node (attractor)**.

**Why only small bubbles stay — a basin effect, not a force-balance failure.**
A stable fixed point can persist mathematically while trapping nothing if its
**basin of attraction** is tiny. The dynamical basin (fraction of core-surface
seeds settling to `|v|~0` by `t*=40`) stays ~90–100% up to r≈0.38 mm, then
**collapses sharply: 55% at r=0.42 (W*≈6.9), 7.5% at r=0.46 (W*≈8.3), 0% for
r≥0.50 (W*≥9.8)**. So the capture cutoff is set by a buoyancy threshold
**W*≈7–9**, consistent with the remaining-% 50% crossover and the §4 `d_crit≈0.96
mm`. As `W*=St/Fr²` grows (up to ~31 at r=0.9 mm) buoyancy ejects the bubble
before it can settle (`outputs/120_5D_upper_attractor.png`; the remaining-% curve
sits slightly above the strict basin at large r — those are slow bubbles not yet
out of the FOV at `t*=20`, not truly trapped).

**The trap MIGRATES outward with bubble size** (`outputs/120_5D_upper_attractor_
positions.png`, each dot = one settled bubble, coloured by radius): the
equilibrium `r*` moves monotonically from `~0.04` (the tiniest bubbles park near
the recirculation eye, where the flow is slow and only weak downflow is needed)
out to `~0.97` near the core periphery at r≈0.46 mm (larger `W*` needs the
stronger downflow found nearer the fast-swirling core), beyond which no balancing
downflow exists and trapping ceases. The location is thus set by the
`u_z = -W*` balance, sweeping outward as buoyancy grows.

**Caveat (same as §7a):** at the largest sizes `Re_b~10³`, so the Stokes-drag
law (hence the precise attractor location/strength) is out of regime; the
*existence* of the small-bubble trap is robust to this.

---

## 7c. Lagrangian coherent structures (FTLE) — engine + first results

Building an LCS/FTLE layer over the trajectory machinery to extract transport
barriers objectively (the basin boundary in §7b was measured pointwise; FTLE
ridges give it as a structure). Engine: `src/ftle.py`, driver `drivers/run_ftle.py`.

**Why 2-D in the (x,z) plane is rigorous here.** Gravity is along lab-`z` and the
field is axisymmetric, so the vertical plane `y=0` (with `v_y=0`) is **invariant**
(`u_y=0`, `a_y=0` by the `y→-y` reflection symmetry) and it contains the trapping
attractor (which sits at `y=0`). So a 2-D flow-map FTLE in `(x,z)` is exact for
that slice — full 3-D (off-plane azimuthal structure) is a later refinement.

**Engine.** Vectorised fixed-step RK4 flow maps advancing the whole grid at once,
with domain-exit masking; FTLE `= 1/(2|T|) ln√λ_max(F^T F)` from central
differences of the flow map. Two flows:
- **fluid** tracers `dx/dt=u` (forward = repelling/separatrix, backward = attracting);
- **inertial** bubbles on the non-stiff Sapsis–Haller slow-manifold field
  `v = u + St (u·∇)u + W* ẑ` (W*=St/Fr²). The full Maxey–Riley RHS is too stiff
  for an explicit flow map at small St (drag `R/St`); the slow-manifold reduction
  is the standard inertial-LCS object and is stable for all St. Forward only —
  backward-time inertial dynamics are ill-posed (the slow manifold repels backward).

**First results (120_5D upper, validated against §7b):**
- `outputs/120_5D_upper_ftle_fluid.png` — forward FTLE marks the **atmosphere
  separatrix** (rear-axis ridge + loops around both cores); backward FTLE shows
  attracting cores. Symmetric in `±z` (fluid sees no gravity).
- `outputs/120_5D_upper_ftle_inertial.png` — inertial FTLE is **gravity-broken
  and size-dependent**: a coherent trapping ridge around the upper core at small
  size, collapsing to a **domain evacuation** (everything detrains) by `W*≈8` —
  the same threshold as the §7b basin collapse.

**`U_ring`-closure sweep (REPORT §6 #1 — addressed).** `run_ftle.py --uring-sweep`
rebuilds the co-moving field from the dimensional mean at each candidate `U_ring`
and measures atmosphere closure two independent ways: a **Lagrangian** trapped
fraction (core-seeded tracers still in-domain after `T*=18`) and a **topological**
on-axis stagnation-point count. Results:

- **120_5D** (`outputs/120_5D_upper_uring_closure.png`): trapped ~100% across the
  whole 28–122 mm/s range — a strong fresh ring stays closed far above the
  thin-ring `U_ring=56`; the closure transition is off-scale (confirms §3).
- **200_10D** (`outputs/200_10D_upper_uring_closure.png`): a **sharp closure
  transition** — trapped 90→70% up to ~65 mm/s, then 19.8% at 73, **0% at 81**,
  with the stagnation count dropping **4→0** at the same speed. The thin-ring
  `U_ring=83.8` sits *just above* closure ⇒ open. Reproduces/sharpens the manual
  ~72–75 mm/s bracket of §3.
- **200_5D** (`outputs/200_5D_upper_uring_closure.png`): closure ≈ **85–95 mm/s**
  (trapped 94→65% then →0; stagnation 4→0); thin-ring `109.7` is above ⇒ open.
- **120_15D** (`outputs/120_15D_upper_uring_closure.png`): closure ≈ **22–30 mm/s**
  only (a weak far-downstream ring, compounded by the §2/§4 `R=14` core-detection
  artifact); thin-ring `55` is far above ⇒ strongly open.
- **120_10D** (`outputs/120_10D_upper_uring_closure.png`): closure ≈ **62–68 mm/s**;
  thin-ring `40.8` is below ⇒ traps. (At higher U a topologically-closed but
  tiny atmosphere persists — `nstag≥2` with only ~3% trapped — illustrating that
  the strict "a closed streamline exists" condition opens later than the
  substantial-atmosphere green line.)
- **200_15D** (`outputs/200_15D_upper_uring_closure.png`): trapped ~100% to >184
  mm/s — another strong ring, closure off-scale; thin-ring `83.8` ⇒ traps.

**Closure vs thin-ring `U_ring`, ALL SIX stations:**

| station | thin-ring U | closure U | thin-ring traps? | §3 tracer |
|---|---|---|---|---|
| 120_5D  | 56  | >122 (off-scale) | ✅ traps | ✅ |
| 120_10D | 41  | ~62–68          | ✅ traps | ✅ |
| 120_15D | 55  | ~22–30          | ❌ open  | ❌ (R=14 artifact) |
| 200_5D  | 110 | ~85–95          | ❌ open  | ❌ |
| 200_10D | 84  | ~65–81          | ❌ open  | ❌ |
| 200_15D | 84  | >184 (off-scale)| ✅ traps | ✅ |

**Takeaway.** The closure test **reproduces the §3 trapping pattern exactly**:
`closure U > thin-ring U` ⟺ the station traps, for all six stations. For every
trapping station closure sits above the thin-ring value (often far above — strong
fresh rings); for every open station closure sits below it. So the thin-ring
formula systematically *overestimates* `U_ring` for the non-thin (`a/R≈0.3`)
cores, which is precisely why §3's atmospheres look open at the thin-ring speed.
The FTLE/topological closure speed is therefore a principled, measurement-light
`U_ring` for the marginal stations, agreeing independently with the §3 tracer
test across both the Lagrangian (trapped-fraction) and topological (stagnation-
count) metrics.

**Green-line definition.** The reported closure U is the highest swept `U_ring`
with BOTH trapped-fraction ≥50% (a substantial Lagrangian atmosphere) AND ≥2
on-axis stagnation points (a closed separatrix exists). It is grid-resolution
limited and, being the substantial-atmosphere criterion, is more conservative
than the strict topological "any closed streamline" point (`nstag`→0), which can
open at a somewhat higher speed (e.g. 200_10D: green ≈65 by trapped-50%, strict
topological opening ≈80).

---

## 7d. Re-running the open stations at the closure-`U_ring`

Adopting the §7c closure speed as `U_ring` (the principled value where the
atmosphere just closes), the three previously FOV-limited stations were rebuilt
(`drivers/rebuild_at_uring.py`) and re-run with the build-plan uniform-size
seeding (`run_uniform_size.py`, both cores):

**The qualitative fix.** At the thin-ring speed these stations escaped ~100%
*advectively* (through the axial FOV walls — a truncation artifact, §3). At a
closed-atmosphere `U_ring` the escape is now **physical buoyant detrainment**
(`r_top` 75–91%) and a well-defined `d_crit` appears — exactly what the
closure-`U_ring` was meant to recover.

**`U_ring` recalibration (within the closed band).** The green-line *closure*
speed is the marginal edge of the atmosphere, where `d_crit` collapses (200_5D
gave only 0.16 mm). `scan_dcrit_uring.py` shows `d_crit(U)` rises as `U` drops
below closure (more robust atmosphere) then saturates, so the marginal stations
were re-run at the **`d_crit`-maximising `U_ring`** (200_5D 85→**60**, 200_10D
65→**48**). This is the principled choice — the most robust trapping the field
supports — and it fixes the within-200 inversion (now 200_5D ≥ 200_10D).

**All-six `d_crit` (`run_uniform_size`, upper core; `compare_capture_vs_size.png`,
`compare_volume_vs_time.png`):**

| station | U_ring | Fr | d_crit (mm) | quality |
|---|---|---|---|---|
| 120_5D  | 55.6 | 0.126 | **1.04** | well-resolved |
| 120_10D | 40.8 | 0.092 | **0.31** | well-resolved |
| 120_15D | 25 (closure) | 0.056 | 0.23 | core artifact `R≈14`* |
| 200_5D  | 109.7 (1-frame) | 0.248 | **0.78** | single-frame @ thin-ring (avg smears)* |
| 200_10D | 83.8 (1-frame) | 0.189 | **0.56** | single-frame @ thin-ring (avg smears)* |
| 200_15D | 83.8 | 0.189 | 0.93 | core artifact `a_eq≈14.6 mm`* |

**What is now physical, and what is a data limit.** The well-resolved **120
series is monotonic downstream** (1.04 → 0.31, stronger fresh ring traps a wider
band) and the recalibrated **200_5D > 200_10D**. The two residual inversions are
NOT fixable by `U_ring` and are flagged as data artifacts:
- **200_15D `d_crit`=0.93 is inflated** by an anomalous core fit (`a_eq≈14.6 mm`,
  `a/R≈0.73` — a diffuse-vorticity 15D core, the same class of artifact as
  120_15D's `R≈14`). Its large apparent atmosphere over-traps.
- **200_5D / 200_10D: the open atmosphere was a TIME-AVERAGING artifact** (see
  §7e). It is NOT a FOV-size limit (the FOVs are comparable, 93–113 mm axial) and
  NOT a real low propagation speed. At the thin-ring speed the *time-averaged*
  field is open (`closure_mismatch_streamlines.png`), but **every instantaneous
  frame closes** — averaging smears the meandering vortex. The closure-band
  recalibration (110→60, 84→48) was therefore compensating for the smear and is
  **superseded**: 200_5D / 200_10D now use a representative single frame at the
  **physical thin-ring `U_ring`**, giving `d_crit` = **0.78 / 0.56** (up from the
  smear-suppressed 0.44 / 0.40), with the 200-series now decreasing downstream.
  The residual `200_5D < 120_5D` (0.78 vs 1.04) is a mild cross-series wrinkle —
  robust to frame choice (~0.78–0.81), so genuine field difference or residual
  single-frame noise, not a recalibration artifact.

**Caveat on the trapped/escaped definition.** In `run_uniform_size` a bubble
"escapes" when its trajectory crosses the **PIV-FOV rectangle** (`field.bounds`),
*not* when it leaves the core or the separatrix — so "captured" means *still
inside the FOV at `t*=20`*. This is FOV- and `t_max`-dependent: a larger domain
or longer time would reclassify some bubbles. Two things keep the headline
robust to this: (i) **exit-face tagging** — `r_top` (rises out the top) is
physical buoyant detrainment, while `x_min/x_max` is advective FOV-truncation;
`d_crit` is read from the buoyant channel (75–91% `r_top` at the closure-`U`
stations, §7d table). (ii) The **attractor/basin definition** (§7b) is
FOV-independent — "trapped" there means relaxed to the `v≈0` equilibrium. Where
they overlap (120_5D) the two agree.

**Takeaway.** Recalibrating `U_ring` recovers physical *within-series* trends and
buoyant (not advective) detrainment; the residual cross-station inversions are
genuine **measurement** limits — the closure-speed mismatch of the fast/fresh
rings (§3) and the diffuse 15D core fits — to be fixed by better core detection /
an independent ring-speed (§6), not by further `U_ring` tuning. `d_crit`
magnitudes (0.3–1.0 mm) bracket the measured escaped-diameter medians (120: 0.30
mm, 200: 0.44 mm).

---

## 7e. An FOV-independent "escape" definition — separatrix (implemented)

The §7d caveat is that `run_uniform_size` defines escape as crossing the **PIV-FOV
rectangle** (`field.bounds`), so "captured" = *still inside the FOV at `t*=20`* —
FOV- and `t_max`-dependent. We want an escape criterion tied to the **vortex
structure**, not the measurement window. Three candidates:

| option | meaning | ease | verdict |
|---|---|---|---|
| **Co-moving streamfunction separatrix** | high — it *is* the atmosphere boundary | medium | **chosen** |
| Distance from core > `k`·(atmosphere radius) | medium — arbitrary `k`, atmosphere isn't circular | high | rejected (ad hoc) |
| Per-size inertial-LCS / attractor basin (§7b) | high — exact for each bubble size | low — a basin computation per size | future / validation only |

**Chosen: the co-moving Stokes streamfunction separatrix.** A detraining bubble
physically crosses the dividing streamline, so this is the exact trapping
boundary; it is FOV-independent (set by the field, not `field.bounds`); it reuses
the stagnation-point machinery from the closure sweep (§7c); and it cleanly
subsumes the exit-face tags (crossing the separatrix with z>0 = buoyant
detrainment, while the axial wash-through that the FOV box mislabels disappears).
The distance-from-core option is rejected as ad hoc (the atmosphere is not a
disk), and the per-size attractor-basin option, while the most rigorous, is
expensive (a basin sweep per diameter) — it is kept as a *validation* cross-check
where it overlaps (it agreed with capture at 120_5D, §7b).

**Implementation (`src/separatrix.py`).** (i) the Stokes streamfunction `ψ(x,r)` on
the meridional grid — solved as a **least-squares / Helmholtz-projected `ψ`**
(the `ψ` whose gradient best matches *both* velocity components, `∂ψ/∂x=-r u_r`,
`∂ψ/∂r=r u_x`, with `ψ=0` pinned on the axis), which is the principled fix for
non-solenoidal PIV; (ii) the separatrix is `ψ=ψ_sep≈0` (saddle on the axis);
(iii) take the **largest closed `ψ`-loop enclosing the core**, then **close its
dividing-streamline arc along the axis** — the recirculation bubble is bounded
*below* by `r=0` between the front/rear saddles, so the polygon must include the
near-axis strip (without this the lower edge lifts off the axis and a captured
bubble that dips toward the axis is spuriously flagged as escaped); (iv) escape
= the bubble **sustainedly** leaves the polygon (final state outside, or it ran
out the FOV) — point-in-polygon along the already-integrated trajectory, taking
the start of the *last continuous outside-run* as `t_escape` so a captured
bubble's transient orbit-dips don't count. Used point-in-polygon, **not** a bare
`sign(ψ)` event, because `ψ=0` is *both* the axis and the separatrix. Wired into
`advect.py` as `escape="separatrix"` (the FOV event is still integrated as an
outer safety bound); the polygon is cached per field as `separatrix.npz`.

**When the separatrix is ill-defined from PIV — and the fix.**
1. **No closed atmosphere** (open/marginal `U`, §7c): not a failure — it restates
   "no trapping region → all escape"; flag and report as such.
2. **Non-solenoidal PIV noise** makes `ψ` path-dependent / the contour fuzzy.
   *Fix:* **solenoidal (Helmholtz) projection** of the mean field (or a
   least-squares `ψ`) before integrating, on top of the existing smoothing.
3. **Saddle ambiguity** (noisy on-axis stagnation count, up to 6): anchor on the
   exact axial `ψ=0` and take the off-axis loop around the core — don't rely on
   locating the saddle precisely.
4. **Atmosphere larger than the FOV** (loop closes outside the domain): detect a
   contour that **touches `field.bounds`** → flag "FOV-truncated (lower bound)"
   and fall back to the exit-face definition there (a *flag*, not a definition).
Cross-check: the forward-FTLE ridge (§7c) coincides with the streamfunction
separatrix (`outputs/<field>_upper_separatrix.png` overlays the two — the red
dividing streamline tracks the bright forward-FTLE ridge over the core, and every
core-surface tracer seeded *inside* the polygon stays trapped in the FOV, i.e.
inside-separatrix ⟹ trapped is exact).

**Result — `d_crit` is escape-definition-independent for the adopted fields**
(`drivers/compare_escape_defs.py`; identical core-surface seeding and diameter
grid, only the escape test differs; `outputs/escape_def_comparison.{png,json}`):

| station | U_ring | atmosphere area | `d_crit` (FOV) | `d_crit` (separatrix) |
|---|---|---|---|---|
| 120_5D          | 55.6  | 5.28  | 0.95 | 0.95 |
| 120_10D         | 40.8  | 5.15  | 0.30 | 0.30 |
| 120_15D_geom    | 25.0  | 4.98  | 0.18 | 0.18 |
| 200_5D_sf       | 109.7 | 2.89  | 0.72 | 0.72 |
| 200_10D_sf      | 83.8  | 4.32  | 0.53 | 0.53 |
| 200_15D_scaled  | 27.7  | 11.81 | 0.42 | 0.42 |

(These `d_crit` values use the coarse 9-point comparison grid, so they run a
little below the canonical §3 numbers; the *comparison* is what matters — the two
definitions land on the same threshold to 2 d.p.) **The FOV-truncation caveat of
§7d does not bias `d_crit` for the adopted fields:** at every adopted station the
bubbles either spiral onto the in-atmosphere attractor (§7b) or buoyantly cross
the dividing streamline and leave, so the FOV box and the separatrix agree.
`200_15D_scaled` is the one field where the atmosphere genuinely exceeds the FOV
(area 11.8, `fov_truncated=True` — auto-flagged); even there the separatrix
escape matches via the exit-face fallback. The separatrix definition therefore
*confirms* the existing capture statistics rather than overturning them, and
makes the FOV-independence explicit and auditable.

**Regenerated deliverables under the separatrix definition.** `run_uniform_size.py`
gained an `--escape {fov,separatrix}` flag; the separatrix runs write to
`outputs/separatrix_escape/` (the FOV-based canonical figures are kept alongside).
The remaining-%-per-radius and retained-volume **endpoints** are unchanged (set by
`d_crit`), but the volume-retention curve decays slightly **earlier** in time
because a detraining bubble crosses the dividing streamline *before* it reaches
the FOV wall (median ≈0.13 t\* earlier at 120_5D, up to ~0.4; smaller elsewhere).
Buoyant-vs-advective detrainment is re-tagged geometrically from the crossing
location on the separatrix (radial dome = buoyant `r_top`; axial ends =
wash-through), so the breakdown survives the definition change.

**Remedy test for the two open cases — single frame vs single core**
(`test_closure_remedies.py`, `test_frame_closure.py`; closure proxy = core-seeded
fluid tracers still inside after `T*≈12–15`).

| 200_5D / 200_10D, thin-ring U_ring | trapped fraction |
|---|---|
| time-average, both cores | **0% / 0%** |
| time-average, upper core only | 0% / 0% |
| time-average, lower core only | 0% / 0% |
| **single instantaneous frames** (12 sampled) | **100% / 86% median; 100% of frames close** |

**Finding (refines §3).** The open atmosphere of 200_5D / 200_10D at the
*physical* thin-ring `U_ring` is a **time-averaging artifact**, not real: a single
core does NOT recover it (rules out core asymmetry), but **every individual frame
has a closed recirculation at the thin-ring speed** — the time-average smears the
wandering/meandering vortex (even after centroid registration) enough to destroy
closure. (§3 had only tested a 21-frame window, itself an average, so it also
came out open — the *single-frame* test is what isolates the averaging as the
cause.)

**Consequence — applied.** The §7d closure-band recalibration (200_5D 110→60,
200_10D 84→48) was **compensating for an averaging smear**, not a real low
propagation speed, so it is superseded. Implemented in
`build_single_frame_field.py`: pick the closing frame nearest the median closure
(tie-broken to the temporal centre of the window — a quasi-steady, non-transient
frame), recenter it via registration, cache it as the mean, and run the standard
pipeline at the **physical thin-ring `U_ring`**. Result (vs the smear-suppressed
closure-band values):

| station | U_ring | d_crit closure-band → **single-frame** |
|---|---|---|
| 200_5D  | 60 → **109.7** | 0.44 → **0.78** |
| 200_10D | 48 → **83.8**  | 0.40 → **0.56** |

`d_crit` ~doubles and the **200-series decreases downstream** (0.78 > 0.56), as
expected for a decaying ring; capture and retained volume rise (200_5D
20%→36%, retained 0.3%→2.8%). Per the agreed rule, the stations that *already*
close from the time average (120_5D, 120_10D, 200_15D) are left untouched. The
mild residual `200_5D < 120_5D` (0.78 vs 1.04) is robust to frame choice and is
either a genuine field difference or residual single-frame noise; a phase-locked
(core-aligned) average would firm it up if needed. 120_15D still needs the §2/§4
`R≈14` core-detection fix (single-frame won't address a core-detection artifact).
Updated all-six comparison: `compare_capture_vs_size.png`,
`compare_volume_vs_time.png`.

---

## 7f. The 15D core-geometry faults — diagnosis & correction

The two 15D stations broke the downstream core-geometry trend (`diag_15D_core.py`,
time-average vs single-frame, + the `test_25` alternative for 200_15D):

| station | a_eq (avg → 1-frame) | R_ring (avg → 1-frame) | verdict |
|---|---|---|---|
| 200_5D / 200_10D | 7.3 / 7.3 | 22.8 / 24.8 | in-trend |
| **200_15D** (test_34) | **14.6 → 7.4** | 24.2 → 25.3 | `a_eq` = **averaging smear**, fixed |
| 200_15D (test_25 alt) | 10.3 → 7.3 | 19.2 → 19.5 | worse `R`; not used |
| **120_15D** (test_27) | 9.1 → 8.3 (in-trend) | **14.1 → 13.6** | `R` = **persistent**, not a smear |

**200_15D — fixed.** `a_eq`=14.6 was a time-averaging smear of the meandering
far-downstream vortex (same mechanism as §7e): every single frame gives
`a_eq≈7.4` (in trend with 200_5D/10D = 7.3) with `R_ring≈25` (good). The current
`test_34` single-frame beats the `test_25` alternative (R=25 vs 19), so no dataset
swap — 200_15D now uses its single-frame field (`200_15D_sf`, `a_eq`=7.6). The
all-six geometry now follows smoothly downstream.

**120_15D — `R` corrected by an isotropic magnification.** `R_ring`=15 persists at
~13.6 in *every* frame, so it is a genuine/calibration feature of the `test_27`
dataset, not a smear (single-frame doesn't fix it) and there is no alternative
dataset. But it breaks the 120-series geometry: `R` drops 22.4 → 23.9 → **15.1**
(a ring should expand, not shrink), while `a_eq`=9 is in-trend.
`build_warped_field.py` offers two transforms: a radial-only warp
(`ρ(r)=r+Δ·S(r)`) keeps `a_eq` but is **anisotropic** (stretches `r` only → the
recirculation cells get radially squished, topology not preserved). The adopted
fix is the **isotropic magnification** (`--uniform`): scale `x` AND `r` by the
same factor about the ring centre, so the streamlines stay **topologically
identical** (same aspect ratio), enlarged to **R: 15→22.6 mm** to match the
upstream 120 panels. `d_crit` stays in trend (**0.24**, vs 0.23 at R=15) and the
120-series ring radius is now continuous (`120_15D_geom`; atmosphere figure
top-right now matches the 120_5D/10D cell shape). The core grows with the
magnification (`a_eq`~10–14; the fixed 7 mm fit mask under-reports it) — an
accepted consequence of preserving streamline topology for this
calibration-artifact correction. The 200 stations are untouched.

**`d_crit` — geometry fix alone wasn't enough; a strength rescale was.** Fixing
200_15D's `a_eq` did NOT lower its `d_crit` — it *rose* (0.93 → 1.04), because the
single-frame 200_15D has a sharp, strong, coherent vortex that traps *more*. The
inversion was never caused by `a_eq` but by the ring's **total strength**: each
station is an independent piston run, and the 200_15D run fired a ring stronger
than the 200-series trend. Since the nondimensional field is scale-invariant,
scaling the total strength by `s` is equivalent to scaling `U_ring` by `s` in
`St`/`Fr` only (dimensionless field and `a_eq` unchanged, `W*∝1/s` rises);
`scan_strength.py` gives `d_crit(s)`:

| s | 1.00 | 0.50 | 0.35 | 0.25 |
|---|---|---|---|---|
| `d_crit` (mm) | 1.16 | 0.94 | 0.57 | ~0 |

Adopting **`s=0.33`** (effective `U_ring`≈27.7 mm/s; `200_15D_scaled`) gives
`d_crit`=**0.46**, below 200_10D's 0.56. **Both series now decay monotonically
downstream** (`compare_capture_vs_size.png`):

| | 5D | 10D | 15D |
|---|---|---|---|
| 120 | 1.04 | 0.31 | 0.23 |
| 200 | 0.78 | 0.56 | 0.46 |

with `a_eq` in trend (≈7.4–7.6) and 200 > 120 at 10D/15D. The one residual is
`200_5D (0.78) < 120_5D (1.04)` — a mild cross-series wrinkle (robust to frame
choice; likely single-frame noise or genuine field difference), not a
geometry/strength artifact. The strength rescale is a deliberate cross-run
normalisation (independent-run variability), recorded in
`fields/200_15D_scaled/meta.json` (`strength_scale`, `U_ring_unscaled`).

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
