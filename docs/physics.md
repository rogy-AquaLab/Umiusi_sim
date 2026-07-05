# UMIUSI simulator — physics (hydrodynamics + thruster propulsion)

How the analytical model computes the **hydrodynamic forces** (buoyancy + drag) and the **thruster
propulsion**, as implemented in `src/umiusi_sim/`. The analytical model is the *reference* physics
(explicit, readable, tunable); MuJoCo integrates the resulting forces. Parameters live in
`configs/umiusi.yaml`; the force code is `simulator.py::_apply_external_forces` with the per-effect
functions in `physics/hydrodynamics.py` and `physics/thruster.py`.

## Conventions
- **Frame:** CAD frame, **+Y is up**; the 4 thrusters lie in the X–Z plane. Forward is +X.
- **Body 6-vectors:** `[x, y, z, roll, pitch, yaw] = [linear(3), angular(3)]`, body axes.
- **Integration:** physics timestep `0.002 s` (500 Hz); control/command rate `50 Hz`. Gravity is
  applied by MuJoCo through body mass; only **buoyancy, drag (and optional added mass)** are added
  analytically here.
- **Application (important):** every external force is applied at its **true point of action** via
  `mj_applyFT` (accumulated into `qfrc_applied`, cleared each step) — NOT lumped onto the base CoM.
  The `base_link` CoM is offset from the whole-vehicle CoM, so a resultant applied at the wrong point
  would inject a spurious torque (and, for velocity-dependent drag, a runaway feedback).

## Mass properties (`configs/umiusi.yaml`)
| | value |
|---|---|
| hull (`base_link`) mass | 9.9639 kg, CoM `[-0.2411, 0.0934, 0.0003]` m, full inertia given |
| thruster mass | 0.6292 kg × 4 |
| **total mass** | ≈ **12.48 kg** |
| neutral-buoyancy volume | 12.48 kg / 1000 kg/m³ = **0.01248 m³** |

---

## 1. Buoyancy (浮力)
`physics/hydrodynamics.py::buoyancy_force_world`

World-frame buoyancy force, opposing gravity:

```
F_buoy = − ρ · V · g          (world)     magnitude = ρ·V·|g|
```

- `ρ = 1000 kg/m³` (`water.density`), `V = 0.0126 m³` (`water.displaced_volume`), `g = [0,−9.81,0]`.
- `V` is the **displaced (external hull) volume**, set **slightly above** the neutral 0.01248 m³ so the
  vehicle is mildly positively buoyant (floats up slowly when idle). Tune in validation.

**Point of application — center of buoyancy (CoB):** placed **horizontally over the whole-vehicle CoM**
(base + 4 thrusters) and `buoyancy_offset_above_com = 0.05 m` **above** it, expressed in the base body
frame so it **rotates with the hull**:

```
cob_local = (system CoM in base frame) + [0, 0.05, 0]
cob_world = xpos[base] + R · cob_local
```

Because the buoyancy line of action sits *above* the CoM, any tilt produces a **passive righting
moment** (self-levelling). Aligning the CoB horizontally with the *system* CoM (not the base CoM)
avoids a spurious steady couple.

---

## 2. Hydrodynamic drag / damping (流体抵抗)
`physics/hydrodynamics.py::drag_wrench_body`

A diagonal linear + quadratic damping wrench in the **body frame**, componentwise over the 6 axes:

```
W_drag = − ( D_lin · v  +  D_quad · |v| · v )     (body, length-6 [force(3), torque(3)])
```

with (`configs/umiusi.yaml::drag`, body axes `[x, y, z, roll, pitch, yaw]`):

| | x | y | z | roll | pitch | yaw |
|---|---|---|---|---|---|---|
| `D_lin`  (N/(m/s), N·m/(rad/s))   | 15 | 25 | 20 | 3 | 3 | 3 |
| `D_quad` (N/(m/s)², N·m/(rad/s)²) | 40 | 80 | 60 | 5 | 5 | 5 |

(PLACEHOLDER — calibrate against expected terminal velocities.)

**Velocity used (important subtlety):** the linear term uses the **CoM translational velocity**
(`mj_subtreeVel → subtree_linvel`), **not** the body-origin velocity from `mj_objectVelocity`. The
body origin is offset from the CoM, so any rotation would inject a large `ω × r` term that couples spin
into the drag *force* and pumps a numerical runaway. The angular term is applied as a **pure moment**.

**Water current / disturbance:** velocity is taken **relative to the water**,
`v_lin_body = Rᵀ (subtree_linvel − current_world)`, so a current drags the vehicle along (used as a
disturbance in RL). The wrench is applied at the **system CoM** via `mj_applyFT`.

**Added mass:** `physics/hydrodynamics.py::added_mass_wrench_body` gives `−M_A · a` (diagonal), but
`added_mass.diag` defaults to **0 (OFF)** — explicit-integrator added mass needs numerical care; enable
only after drag/buoyancy are validated.

### 2b. Higher-fidelity hydro: lift, translation-induced moment, cross-coupling
Three config-gated terms extend the diagonal drag beyond "force opposes velocity on each axis". Their
coefficients are **PLACEHOLDERs** (physically-plausible, to be calibrated against tow/PMM tests); each
has a default that recovers the old model, and **lift + the CoP offset are ENABLED by default** (modest).

**Lift** — `physics/hydrodynamics.py::lift_force_body`, applied as a pure force at the CoM. A force
*perpendicular* to the body-frame translational velocity, growing with the **angle of attack** `α`
between the flow and the longitudinal reference axis (`lift.ref_axis = +X`, the streamlined/elongated
axis), magnitude `∝ |v|²`:

```
F_lift = coef · |v|² · sin(α)·cos(α) · n̂        (body)   coef = ½ρ·Cl·A  (lumped)
       = coef · |v|² · (v̂·ê_ref) · (ê_ref − (v̂·ê_ref) v̂)
```

`n̂` is the unit vector ⟂ to `v` in the (`v`, `ê_ref`) plane, toward `ê_ref`. The `sin·cos` (= ½ sin 2α)
shape makes lift **vanish for flow ALONG the ref axis (α=0) AND perpendicular to it (α=90°)** — so pure
surge / pure heave / pure sway give **zero lift**, and only genuinely **oblique** flow lifts. That is
why the axis-aligned `validate_sim` invariants (surge stays horizontal, heave rises without spinning)
still hold. `lift.coef = 12.0` N/(m/s)² → ~1 N max at the 0.4 m/s cruise (modest); `coef = 0` = OFF.

**Translation-induced moment via a center-of-pressure (CoP) offset** — the (linear+quadratic)
translational drag force is applied at `CoP = system CoM + R·cop_offset` (body frame) instead of exactly
at the CoM, so a **broadside translation induces a turning/righting moment** (Munk-like): the moment arm
`(CoP−CoM) × F_drag` becomes a couple. `cop_offset = [0.03, 0, 0]` m (a few cm forward); `[0,0,0]`
recovers the old model (drag through the CoM, zero translation moment). The angular-damping moment stays
a location-free free vector, so only the *translational* drag gains the arm.

**Off-diagonal (cross-axis) damping** — `physics/hydrodynamics.py::coupling_moment_body`, optional and
**default OFF**. Adds a body moment coupled to translational velocity, `M += −(c_lin·v + c_quad·|v|·v)`:
`coupling.sway_yaw` (body-Z sway → yaw about +Y, weathercocking) and `coupling.heave_pitch` (body-Y
heave → pitch about +Z). Left at 0 by default since the CoP offset already supplies a translation→moment
coupling; available if a specific cross term needs tuning.

---

## 3. Thruster propulsion (スラスタ推進力)
`physics/thruster.py` + `simulator.py::step / _apply_external_forces`

Four **azimuth thrusters** (T-shaped ducted props). Each unit has a **servo** (azimuth angle) and an
**ESC** (thrust magnitude). The servo rotates the whole unit about its **mounting arm** (~0.15 m, 45°
outboard); the **thrust axis is perpendicular to the arm**, so the servo tilts thrust between
**horizontal tangential** (servo = 0) and **vertical/up** (servo = +90°).

### Action → servo & thrust (`step`)
The 8-D action is `[servo_1..4, esc_1..4]`, each clamped to `[−1, 1]`:

```
servo_target_k = clip(a_servo_k, −1, 1) · servo_range          # servo_range = 90° = 1.5708 rad
servo_ctrl_k   = slew(servo_ctrl_k → servo_target_k, 250°/s, dt)   # rate-limited azimuth (HS-646WP)
esc_current_k  = slew(esc_current_k → clip(a_esc_k,−1,1), 4.0 /s, dt)   # rate-limited ESC command
thrust_mag_k   = esc_current_k · thrust_per_cmd                 # thrust_per_cmd = 30 N per unit cmd
```

- `slew(x→target, rate, dt) = x + clip(target−x, −rate·dt, +rate·dt)` — first-order rate limit
  (mirrors the real `max_duty_step_per_sec`). `servo_slew = 250 °/s` (derated in-water HS-646WP),
  `thrust_slew = 4.0 esc-units/s`.
- `servo_ctrl_k` is written to the MJCF servo actuator (`data.ctrl`, radians); the MJCF hinge (about the
  arm) is authoritative for how the body rotates. Servo actuator force is limited to the stall torque
  (~0.94–1.14 N·m).

### Thrust force (`_apply_external_forces`)
For each thruster the thrust acts along its **(servo-rotated) local axis**, applied at the tip site:

```
F_thr_k(world) = thrust_mag_k · ( R_body_k · thrust_axis_local_k )
mj_applyFT(F_thr_k, point = site_xpos[t k _thrust], body = thruster_k)
```

- `thrust_axis_local_k` is the **neutral (servo = 0) thrust direction** = horizontal tangential (perp to
  the arm), in the thruster body frame. As the servo rotates the thruster body, `R_body_k` tilts this
  axis, so the same stored axis produces the correct tilted thrust.
- Per-unit pivots + neutral axes are in `thrusters.units` (id → name `1=lf 2=lb 3=rb 4=rf`), e.g. unit 1
  `thrust_axis ≈ [0.708, 0, 0.706]`.
- Net effect: the servo splits thrust into a **horizontal** component (→ surge / sway / yaw) and a
  **vertical** component (→ heave / roll / pitch), `φ = atan(f_vertical / f_horizontal)` — matching the
  real `sinsei_umiusi_control` FeedForward allocation.

### Modeling scope & remaining limitations
The core damping is a **diagonal quadratic-damping** model (Fossen/Morison first order), now extended by
the §2b terms (lift, CoP-offset translation moment, optional cross-coupling). Since §2b:
- **Lift IS modeled** (angle-of-attack force ⟂ to the flow; §2b) — modest, ENABLED, placeholder `Cl·A`.
- **Translation-induced moments ARE modeled** via the CoP offset (§2b) — a broadside translation now
  induces a Munk-like turning/righting moment; ENABLED with a small (3 cm) placeholder offset.
- **Off-diagonal damping** (sway→yaw, heave→pitch) is available but **default OFF** (`coupling.* = 0`).

Remaining simplifications:
- **Direction dependence = constant per-axis coefficients** for the *drag* magnitude. The different
  effective `Cd·A` per direction is baked into the differing `D_lin`/`D_quad` per body axis (streamlined
  +X < broadside +Y); it does not dynamically recompute the projected area at an arbitrary flow angle.
- **Lift shape is a simple ½sin 2α** (no explicit stall/`Cl(α)` curve); the CoP offset is a **fixed**
  point (not velocity- or angle-dependent).
- **No added-mass coupling by default** (`added_mass.diag = 0`).
- All §2b coefficients are **PLACEHOLDERs** pending calibration (tow/PMM tests).

This is adequate for a slow, box-ish ROV doing station-keeping + low-speed cruise, and the §2b terms add
first-order lift/translation-moment fidelity for angled / broadside manoeuvres. A further step up would
be a full 6×6 damping matrix and an angle-resolved `Cl(α)` / moving CoP.

---

## 4. Feed-forward allocation (command → per-thruster) — control side
`control.py` (port of `sinsei_umiusi_control`'s FeedForward)

The inverse of §3: given a desired body wrench, an analytical **8×6 allocation matrix** maps it to the 4
`(servo_angle, esc)` pairs. Used by `competition_run` and available to the ROS controllers.
**Frame mapping note:** the FF axes ≠ the sim axes (`Vx → sim −X`, `Vz → +Y`, `Vy → yaw couple`); this is
documented in `control.py` and guarded by `validate_sim.check_ff_allocation` (pure heave rises without
spinning; pure surge stays horizontal). The FF-frame sign reconcile is a tracked follow-up.

---

## Summary of the per-step force pipeline (`_apply_external_forces`)
1. clear `qfrc_applied`;
2. **buoyancy** `−ρV g` at the CoB (above CoM → righting moment);
3. **drag** `−(D_lin v + D_quad |v| v)` (+ optional cross-coupling moment) using CoM velocity relative to
   the current, with the translational force applied at the **CoP** (`CoM + R·cop_offset` → translation
   moment) and the angular damping as a pure moment;
4. **lift** `coef·|v|²·½sin 2α · n̂` ⟂ to the flow, at the CoM (zero at axis-aligned flow);
5. optional **added mass** (off by default) + external **impulse** disturbances at the CoM;
6. **thrust** `mag · R·axis` at each thruster tip site;
7. MuJoCo integrates (gravity via body mass) at 500 Hz.

Parameters marked PLACEHOLDER (drag coeffs, lift `coef`, `cop_offset`, coupling, displaced volume,
buoyancy offset) are first estimates to be calibrated in validation; `tools/validate_sim.py` guards the
qualitative invariants (floats/levels, heave/surge decoupling, no NaN).
