# Instruction — hooking physical PIV data into the (dimensionless) advect scheme

**Scope:** what Claude Code must get right at the seam where real, dimensional PIV
measurements meet the existing dimensionless Maxey–Riley advection scheme
(reused from `advect_bubbles_3D_eval.py`). This is a unit-hygiene + scaling spec,
not a full repo plan (see `claude_code_build_plan.md` for that).

---

## 0. Core decision: stay dimensionless, convert once at the boundary

The advect RHS expects **dimensionless** `u, x, t` and the groups `R, St, Fr`.
Do **not** advect in SI. Instead:

> **All dimensional → dimensionless conversion happens in one place: a
> `nondimensionalize()` step at PIV ingestion. Downstream code (field build,
> seeding, advect, merge) is dimensionless and identical in form to the old repo.**

Everything physical (PIV velocities, the grid, bubble diameters from the CSV,
ν, ρ, g) is touched only inside that boundary. Anything dimensionless that leaks
*out* of it must pass the assertions in §5.

---

## 1. Reference scales

| Scale | Symbol | Value / source |
|-------|--------|----------------|
| Length | `L_f = R0` | **20 mm = 0.02 m** (ring radius), fixed by apparatus |
| Velocity | `U_f = U_ring` | **linear fit of the ring's axial position vs time** over the measurement window (slope) |
| Time | `tau_f = R0 / U_ring` | derived |

`U_ring` carries the only difference between the two rings (same orifice/stroke,
different piston speed), so it is per-ring and must be recomputed for each.

```python
import numpy as np

R0 = 0.02  # m, ring radius (length scale L_f)

def estimate_U_ring(x_ring_t, t):
    """Axial ring speed from a linear fit of tracked position over the window.
    x_ring_t: ring axial location per frame (m); t: frame times (s).
    Returns U_ring (m/s), the velocity scale U_f.
    """
    slope, _ = np.polyfit(t, x_ring_t, 1)
    return abs(slope)
```

**Notice:**
- Define *which* point is tracked (vorticity-weighted centroid is most robust;
  state it). Use the same definition for both rings.
- `U_ring` is the same quantity you subtract for the co-moving frame, so reuse it
  there (no second estimate).
- It decays across the three stations — decide whether `U_f` is per-station
  (local) or one canonical per-ring value used to *label* St. Recommended: one
  per-ring `U_f` for labelling so a bubble keeps a fixed St identity, and report
  the station variation separately (see `model_writeup_sections.tex`, §2.1).

---

## 2. The conversion boundary

PIV gives dimensional `(u_x, u_r)` [m/s] on a dimensional grid `(x, r)` [m or mm],
in the lab frame. Convert in this order:

```python
def nondimensionalize(x_m, r_m, Ux_ms, Ur_ms, U_ring, R0):
    # 1) lengths  -> units of R0
    x  = x_m / R0
    r  = r_m / R0
    # 2) co-moving frame: subtract ring speed (axial only), THEN normalize.
    #    In U_f units this is simply subtracting 1.0, since U_f == U_ring.
    Ux = (Ux_ms - U_ring) / U_ring
    Ur =  Ur_ms / U_ring
    return x, r, Ux, Ur

# 3) gradients are computed in the *starred* (dimensionless) coordinates,
#    so they come out dimensionless automatically — do NOT differentiate the
#    dimensional field and forget the R0/U_f factor.
#    dUxdx = np.gradient(Ux, x, axis=...)   # already dimensionless
```

**Notice (the usual PIV traps):**
- **mm vs m.** Pick SI internally; confirm the PIV calibration (px→length) is
  already applied. A factor-of-1000 here is the most common silent error.
- **Axis convention.** Old code: `x` = axial, `r`/`y` = radial, `r = 0` on the
  axis. Confirm the PIV axis orientation and *locate the axis* (reflection
  symmetry); enforce `r ≥ 0`; optionally fold the two half-planes.
- **Component mapping.** In-plane PIV components are `(u_x, u_r)`; assume
  `u_θ ≈ 0` (non-swirling) unless stereo-PIV says otherwise.
- **Masked / NaN nodes.** Fill or mask *before* differentiating; NaNs propagate
  into every gradient and then into every trajectory.
- **Smooth before differentiating.** PIV gradient noise is amplified in `∇u`;
  apply the spatial low-pass after the window average, before `np.gradient`.

---

## 3. Stokes number (per bubble, density-free τ_p — density lives in R)

For a bubble the response time is taken density-free; the density ratio is
carried entirely by `R = 2ρ_f/(2ρ_p + ρ_f)` (→ 2 for bubbles). This is consistent
with the old repo's drag term `R*(U−v)/St`:

```
tau_p = d_b**2 / (18 * nu)          # nu = mu / rho_f  [m^2/s]
St    = tau_p / tau_f
      = d_b**2 * U_ring / (18 * nu * R0)
```

```python
def stokes_number(d_b, nu, U_ring, R0):
    """d_b: bubble diameter (m); nu: fluid kinematic viscosity (m^2/s)."""
    return d_b**2 * U_ring / (18.0 * nu * R0)
```

**Notice:**
- `St` is **per bubble**, drawn from the escape-size CSV — this fills the
  per-particle St column exactly as the old `Bubbles_df[:, 7]`.
- **Confirm the CSV unit and quantity:** is it diameter or radius, mm or m?
  `St ∝ d²`, so this is squared-sensitive.
- Use ν, ρ_f, Δρ at the experiment temperature.
- Verified consistency (do not re-derive): `R/St` with this `St` and
  `R = 2ρ_f/(2ρ_p+ρ_f)` reproduces the added-mass-corrected drag rate
  `18μ/((ρ_p+ρ_f/2) d²)`. Good to keep as a unit test.

---

## 4. Froude number — **READ THIS BEFORE CODING Fr**

The buoyancy term is `(1 − 3R/2) · ĝ / Fr²`, so `Fr` **must be dimensionless**.

### 4a. Active (consistent) form — USE THIS
Derived by requiring `W_rising = St/Fr²` to equal the true Stokes terminal
velocity `v_t/U_ring`. It has **no per-bubble size** in it:

```
Fr**2 = (rho_f / d_rho) * U_ring**2 / (g * R0)        # d_rho = rho_f - rho_b
Fr    = U_ring * sqrt(rho_f / (d_rho * g * R0))
# air bubble in water: d_rho ≈ rho_f  =>  Fr ≈ U_ring / sqrt(g * R0)
```

```python
def froude_number(U_ring, rho_f, d_rho, R0, g=9.81):
    return U_ring * np.sqrt(rho_f / (d_rho * g * R0))
```

### 4b. User-proposed form — PARKED, do not enable without resolution
```
# Fr_proposed = sqrt( rho_f * pi**2 * a_eq**2 * U_ring / (d_rho * v_b * g) )
#   a_eq : equivalent VORTEX-CORE radius (NOT R0, NOT bubble radius)
#   v_b  : individual bubble volume
#
# FLAGS:
#  (i)  As written it is NOT dimensionless: the radicand has units L^-1 T (s/m).
#  (ii) Even with U_ring -> U_ring^2 (units fixed), v_b ∝ d^3 inside Fr makes
#       W_rising = St/Fr^2 scale as d^2 * d^3 = d^5, vs the physical d^2.
#  => leave disabled; ask the author for the derivation if a_eq, v_b truly belong
#     (e.g. a core-scale balance rather than standard buoyancy normalization).
```

### 4c. Guard it at runtime
Make Fr dimensionless-ness a hard check so a broken formula can't run silently:

```python
import pint  # or a hand-rolled dimensional check
def assert_dimensionless_Fr(expr_with_units):
    assert expr_with_units.dimensionless, (
        f"Fr is not dimensionless: {expr_with_units.dimensionality}. "
        "See section 4 — buoyancy term will be corrupted."
    )
```

**Notice:**
- `a_eq` (equivalent core radius, from PIV core fit) ≠ `R0` (ring radius) ≠ bubble
  radius. Keep the three distinct; conflating them is the likely origin of 4b.
- `Fr` here is **per-ring** (depends on `U_ring`), not per-bubble.

---

## 5. Do NOT reuse the old hardcoded numbers

These were tied to the Norbury normalization `U_f = 0.5, L_f = 1` and are **not
portable**:

| Old value | Why it must be recomputed |
|-----------|---------------------------|
| `St0 = 1` (and 0.25, 0.5) | `St ∝ U_f`; now per-bubble from CSV via §3 |
| `Fr = 0.5` | `Fr ∝ U_f`; now from §4a |
| `W_rising = St0/Fr²` | recompute; should equal `v_t/U_ring` (test below) |
| `dt`, `tEnd` | in `τ_f` units; `τ_f = R0/U_ring` differs, so the physical duration changes |
| `U_f = 0.5`, `W_norbury = 0.6586` | drop entirely — Norbury-specific |

If you ever put St values from this model next to the inviscid Norbury model,
**both must use the same (L_f, U_f) convention**, or the `St ∝ d²` slopes will
agree while the absolute numbers won't.

---

## 6. Validation assertions (cheap, catch the common failures)

```python
# (a) dimensionless field is O(1) after co-moving + normalization
assert np.nanpercentile(np.abs(Ux), 99) < 5, "Ux not O(1) — check U_f / units"

# (b) co-moving frame: far-field axial velocity ~ -1 (lab flow ~0 => -U_ring/U_f)
#     and near-ring closed streamlines exist (a stagnation structure)

# (c) terminal-velocity self-consistency (ties §3 and §4a together):
nu = mu / rho_f
v_t = d_rho * d_b**2 * g / (18 * mu)          # physical Stokes rise (m/s)
W_rising_phys = v_t / U_ring                  # dimensionless
W_rising_model = stokes_number(d_b, nu, U_ring, R0) / froude_number(U_ring, rho_f, d_rho, R0)**2
assert np.allclose(W_rising_phys, W_rising_model, rtol=1e-6), "St/Fr^2 != v_t/U_ring"

# (d) Re_bubble sanity for the largest bubble (Stokes-drag validity, §7)
Re_b = np.abs(slip_velocity) * d_b / nu
# warn if Re_b ≳ 1 — nonlinear drag correction may be needed
```

---

## 7. Caveat to record (not blocking)

`tau_p = d_b²/(18ν)` assumes **Stokes drag** (bubble Reynolds number
`Re_b = |u−v| d_b/ν ≲ 1`). Your largest escape-size bubbles may exceed this; if
`Re_b ≳ 1`, a nonlinear drag correction (e.g. Schiller–Naumann) modifies the
effective `tau_p`. Flag in the run log; document as a limitation.

---

## Summary for the agent
1. Convert once in `nondimensionalize()`; keep everything downstream dimensionless.
2. `L_f = R0 = 0.02 m`; `U_f = U_ring` from a linear position fit, per ring.
3. `St` per bubble from CSV diameters: `d²·U_ring/(18 ν R0)` (density in R).
4. `Fr = U_ring·sqrt(ρ_f/(Δρ g R0))` (§4a). The `a_eq`/`v_b` formula is flagged
   non-dimensionless and disabled — resolve with the author before use.
5. Recompute St, Fr, dt, tEnd from the new scales; drop all Norbury-era constants.
6. Run the §6 assertions, especially the `St/Fr² == v_t/U_ring` self-check.