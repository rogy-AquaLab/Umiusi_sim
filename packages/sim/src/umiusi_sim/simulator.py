"""UmiusiSimulator — MuJoCo stepping + analytical hydrodynamics + thruster forces.

Single physics implementation shared by the validation tool, the RL env, and the ROS 2
bridge. Exposes a clean reset()/step(action)/get_state() API.

Action (8-D, each in [-1, 1]):
    [servo x4, esc x4], channel order = config `thrusters.action_order` (default lf, lb, rb, rf —
    the sinsei_UMIUSI_autonomy POSITIONS contract; names are per-unit `name` fields in the config)
    servo_k -> target azimuth angle = servo_k * servo_range (servo model: slew + first-order lag)
    esc_k   -> thrust = esc_k * thrust_per_cmd  [N]

Frame: CAD frame, +Y up. Units: SI, radians internally.
"""

from pathlib import Path

import mujoco
import numpy as np
import yaml

from umiusi_sim.physics import hydrodynamics as hydro
from umiusi_sim.physics import thruster as thr

_PKG = Path(__file__).resolve().parent            # packages/sim/src/umiusi_sim
_ROOT = _PKG.parents[3]                            # repo root (packages/sim/src/..)
_DEFAULT_MODEL = _PKG / "description" / "umiusi.xml"
_DEFAULT_CONFIG = _ROOT / "configs" / "umiusi.yaml"


class UmiusiSimulator:
    def __init__(self, model_path=_DEFAULT_MODEL, config_path=_DEFAULT_CONFIG):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.dt = float(cfg["sim"]["timestep"])
        self.model.opt.timestep = self.dt
        self.model.opt.gravity[:] = cfg["sim"]["gravity"]
        self.substeps = max(1, round((1.0 / cfg["sim"]["control_rate_hz"]) / self.dt))

        # Water / hull
        self.density = float(cfg["water"]["density"])
        self.volume = float(cfg["water"]["displaced_volume"])
        self.buoyancy_offset = float(cfg["water"]["buoyancy_offset_above_com"])
        self.gravity = np.array(cfg["sim"]["gravity"], dtype=float)

        # Drag / added mass, order [x, y, z, roll, pitch, yaw] = [linear(3), angular(3)]
        self.drag_lin = np.array(cfg["drag"]["linear"], dtype=float)
        self.drag_quad = np.array(cfg["drag"]["quadratic"], dtype=float)
        self.added_mass_diag = np.array(cfg["added_mass"]["diag"], dtype=float)

        # Higher-fidelity hydro (all config-gated; absent key -> OFF = the old diagonal-drag model):
        #   lift        : force perpendicular to the body-frame flow, grows with angle of attack (~|v|^2)
        #   cop_offset  : apply the translational drag force at a center-of-pressure OFFSET from the CoM,
        #                 so a broadside translation induces a turning/righting (Munk-like) moment
        #   coupling    : optional off-diagonal (sway->yaw, heave->pitch) damping moments
        lift_cfg = cfg.get("lift", {})
        self.lift_coef = float(lift_cfg.get("coef", 0.0))                      # 0.5*rho*Cl*A [N/(m/s)^2]
        self.lift_ref_axis = np.array(lift_cfg.get("ref_axis", [1.0, 0.0, 0.0]), dtype=float)
        self.cop_offset = np.array(cfg.get("cop_offset", [0.0, 0.0, 0.0]), dtype=float)  # body frame [m]
        coup = cfg.get("coupling", {})
        self.coupling_sway_yaw = np.array(coup.get("sway_yaw", [0.0, 0.0]), dtype=float)
        self.coupling_heave_pitch = np.array(coup.get("heave_pitch", [0.0, 0.0]), dtype=float)

        # Thrusters
        t = cfg["thrusters"]
        self.servo_range_rad = np.radians(max(abs(v) for v in t["servo_range_deg"]))
        self.servo_slew_rad = np.radians(t["servo_slew_deg_per_s"])
        self.thrust_slew = float(t.get("thrust_slew_per_s", 1e9))  # esc units/s (mirrors max_duty_step_per_sec)
        # Servo tracking: rate = clip(err / tau, +/- slew) — slews on large errors, converges as a
        # first-order lag near the target (tau = 0 -> the old pure-slew model). See configs comment.
        self.servo_tau = float(t.get("servo_tau_s", 0.0))
        self.thrust_per_cmd = float(t["thrust_per_cmd"])
        # Propeller-law thrust curve F = sign(u)|u|^exp * thrust_per_cmd (1.0 = old linear map).
        # Fitted against the 2026-08-21 pool bag: linear overpredicts low-duty thrust ~10x.
        self.thrust_curve_exp = float(t.get("thrust_curve_exp", 1.0))
        # ESC duty cap |u| <= max_duty — the DEPLOY-PATH clamp (navigator/rl_attitude max_duty),
        # modeled in the PLANT so training sees the same saturated command chain the robot runs.
        # The 8/25 run showed a capless-trained policy demands |esc| median 0.63 against a 0.2
        # deploy cap -> permanent saturation / bang-bang (Umiusi_sim#3). 1.0 = no cap (old behavior).
        self.max_duty = float(t.get("max_duty", 1.0))
        # Action channel order (issue #3, item 1): action[k] drives the unit NAMED
        # action_order[k]. Per-unit names live in the config (geometric: lf/lb/rf/rb); the default
        # order is the autonomy-side POSITIONS contract. Without names (old config) fall back to
        # id order, preserving the old behavior.
        names = {u.get("name", f"unit_{u['id']}"): int(u["id"]) for u in t["units"]}
        order = t.get("action_order")
        if order:
            missing = [n for n in order if n not in names]
            if missing:
                raise ValueError(f"action_order names {missing} not among unit names {list(names)}")
            self.unit_ids = [names[n] for n in order]     # action index k -> config unit id
        else:
            self.unit_ids = [int(u["id"]) for u in t["units"]]
        self.unit_names = order if order else list(names)
        by_id = {int(u["id"]): u for u in t["units"]}
        # Per-thruster neutral thrust direction (thruster body frame), in ACTION order. The MJCF
        # servo hinge (about the mounting arm) rotates the body, so this axis tilts with the servo.
        self.thrust_axes = np.array([by_id[i]["thrust_axis"] for i in self.unit_ids], dtype=float)
        # Mounting-pivot positions (CAD frame), in ACTION order. Static geometry, so this is the
        # right source for consumers that need the fore/aft + port/starboard layout without
        # stepping the model (umiusi_rl.envs.mode_mixer validates its sign table against it).
        self.unit_pivots = np.array([by_id[i]["pivot"] for i in self.unit_ids], dtype=float)

        # Indices, all in ACTION order (MJCF entities keep the id-based names).
        self.base_id = self.model.body("base_link").id
        self.thr_ids = [self.model.body(f"thruster_{i}").id for i in self.unit_ids]
        self.site_ids = [self.model.site(f"t{i}_thrust").id for i in self.unit_ids]
        self.act_ids = [self.model.actuator(f"servo_{i}").id for i in self.unit_ids]
        self.servo_qadr = [self.model.jnt_qposadr[self.model.joint(f"servo_{i}").id] for i in self.unit_ids]

        # Center of buoyancy: horizontally over the whole-vehicle CoM (base + thrusters),
        # `buoyancy_offset` above it, expressed in the base body frame (rotates with the hull).
        mujoco.mj_forward(self.model, self.data)
        R0 = self.data.xmat[self.base_id].reshape(3, 3)
        sys_com_local = R0.T @ (self.data.subtree_com[self.base_id] - self.data.xpos[self.base_id])
        self.cob_local = sys_com_local + np.array([0.0, self.buoyancy_offset, 0.0])

        self._renderer = None  # lazy mujoco.Renderer for render_camera(); cached per (w, h)
        # Underwater camera degradation (perception realism): when camera_degrade is True,
        # render_camera() applies the physically-based underwater degradation (colour attenuation +
        # haze, from perception.underwater_sim) using the depth buffer, so the perception input looks
        # like real murky footage. water_params fixes the water condition for the run (set at reset);
        # None -> a moderately murky default. Off by default (clean render) so existing callers are
        # unchanged; the perception/autonomy tools opt in.
        self.camera_degrade = False
        self.water_params = None
        self._cam_rng = np.random.default_rng(0)

        self.reset()

    def set_buoyancy_offset(self, offset):
        """Move the CoB to `offset` [m] above the system CoM (keeps the horizontal placement).
        Used by domain randomization and calibration fits; mirrors the __init__ computation."""
        self.cob_local = self.cob_local + np.array([0.0, float(offset) - self.buoyancy_offset, 0.0])
        self.buoyancy_offset = float(offset)

    # -- lifecycle -------------------------------------------------------------
    def reset(self, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = pos
        self.data.qpos[3:7] = quat
        for a in self.servo_qadr:
            self.data.qpos[a] = 0.0
        self.servo_ctrl = np.zeros(4)
        self.esc_current = np.zeros(4)
        self.thrust_mag = np.zeros(4)
        self.thrust_world = np.zeros((4, 3))  # last applied per-thruster world force [N]
        self.prev_vel_body = np.zeros(6)
        self.current_world = np.zeros(3)    # water-current velocity [m/s] (disturbance; set by the env)
        self.ext_force_world = np.zeros(3)  # extra external force [N] at the CoM (impulse disturbance)
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(8)
        servo_target = np.clip(action[:4], -1.0, 1.0) * self.servo_range_rad
        esc_target = np.clip(action[4:8], -self.max_duty, self.max_duty)
        for _ in range(self.substeps):
            self.servo_ctrl = thr.track(self.servo_ctrl, servo_target, self.servo_slew_rad,
                                        self.servo_tau, self.dt)
            self.esc_current = thr.slew(self.esc_current, esc_target, self.thrust_slew, self.dt)
            u = self.esc_current
            self.thrust_mag = np.sign(u) * np.abs(u) ** self.thrust_curve_exp * self.thrust_per_cmd
            for k, aid in enumerate(self.act_ids):
                self.data.ctrl[aid] = self.servo_ctrl[k]
            self._apply_external_forces()
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    # -- forces ----------------------------------------------------------------
    def _apply_external_forces(self):
        # Forces are applied at their true points of action via mj_applyFT (accumulated into
        # qfrc_applied), NOT lumped onto the base CoM. This matters because the base_link CoM
        # is offset from the whole-vehicle CoM: a resultant force applied at the wrong point
        # injects a spurious torque (and for velocity-dependent drag, a runaway feedback).
        m, d = self.model, self.data
        d.qfrc_applied[:] = 0.0
        base = self.base_id
        R = d.xmat[base].reshape(3, 3)
        zero3 = np.zeros(3)

        # Buoyancy: force at the center of buoyancy (above the system CoM -> restoring moment).
        f_buoy = hydro.buoyancy_force_world(self.density, self.volume, self.gravity)
        cob_world = d.xpos[base] + R @ self.cob_local
        mujoco.mj_applyFT(m, d, f_buoy, zero3, cob_world, base, d.qfrc_applied)

        # Hydrodynamic damping: linear drag through the system CoM (no spurious torque),
        # angular drag as a pure moment. The linear term MUST use the CoM translational
        # velocity (subtree_linvel), not mj_objectVelocity's body-ORIGIN velocity: the
        # origin is offset from the CoM, so any rotation injects a huge omega x r term that
        # couples spin into the drag force and pumps a runaway. Coeffs are body-axis.
        mujoco.mj_subtreeVel(m, d)  # fills d.subtree_linvel (world CoM velocity)
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, base, vel6, 0)  # global
        # Relative to any water current, so a current drags the vehicle along (disturbance).
        lin_body = R.T @ (d.subtree_linvel[base] - self.current_world)
        ang_body = R.T @ vel6[:3]
        vel_body = np.concatenate([lin_body, ang_body])
        w = hydro.drag_wrench_body(vel_body, self.drag_lin, self.drag_quad)
        # Optional off-diagonal (cross-axis) damping moments (default 0 -> no change).
        if self.coupling_sway_yaw.any() or self.coupling_heave_pitch.any():
            w[3:] = w[3:] + hydro.coupling_moment_body(lin_body, self.coupling_sway_yaw,
                                                       self.coupling_heave_pitch)
        if np.any(self.added_mass_diag):  # optional; off by default (needs numerical care)
            acc_body = (vel_body - self.prev_vel_body) / self.dt
            w = w + hydro.added_mass_wrench_body(acc_body, self.added_mass_diag)
        self.prev_vel_body = vel_body
        sys_com = d.subtree_com[base].copy()
        # Apply the translational drag FORCE at the center of pressure (CoM + cop_offset): the moment
        # arm turns a broadside translation into a righting/turning moment (Munk-like). The angular
        # damping + coupling moment (w[3:]) is a free vector, applied here as a pure moment. With
        # cop_offset = 0 this is identical to applying everything at the CoM (the old model).
        cop_world = sys_com + R @ self.cop_offset
        mujoco.mj_applyFT(m, d, R @ w[:3], R @ w[3:], cop_world, base, d.qfrc_applied)

        # Lift: force perpendicular to the body-frame flow (angle-of-attack dependent), at the CoM.
        f_lift_body = hydro.lift_force_body(lin_body, self.lift_coef, self.lift_ref_axis)
        if f_lift_body.any():
            mujoco.mj_applyFT(m, d, R @ f_lift_body, zero3, sys_com, base, d.qfrc_applied)

        # External impulse disturbance (waves/bumps), a world-frame force at the CoM.
        if self.ext_force_world.any():
            mujoco.mj_applyFT(m, d, self.ext_force_world, zero3, sys_com, base, d.qfrc_applied)

        # Thrusters: thrust along the (servo-rotated) local axis, applied at each tip site.
        for k in range(4):
            bid, sid = self.thr_ids[k], self.site_ids[k]
            f_thr = thr.thrust_to_world(self.thrust_mag[k], self.thrust_axes[k], d.xmat[bid])
            self.thrust_world[k] = f_thr  # kept for the RL null-mode decomposition / diagnostics
            mujoco.mj_applyFT(m, d, f_thr, zero3, d.site_xpos[sid], bid, d.qfrc_applied)

    # -- observation -----------------------------------------------------------
    def get_state(self):
        d = self.data
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.base_id, vel6, 0)
        return {
            "pos": d.xpos[self.base_id].copy(),
            "quat": d.xquat[self.base_id].copy(),
            "lin_vel": vel6[3:].copy(),
            "ang_vel": vel6[:3].copy(),
            "servo": np.array([d.qpos[a] for a in self.servo_qadr]),
            "thrust": self.thrust_mag.copy(),
            "thrust_world": self.thrust_world.copy(),
        }

    # -- perception ------------------------------------------------------------
    def render_camera(self, camera="front_cam", width=320, height=240, degrade=None, water_params=None):
        """Return an (H, W, 3) uint8 RGB image from an onboard MJCF camera.

        Optional and self-contained: it reads the current physics state but does not
        advance or alter it. A mujoco.Renderer is created lazily and cached (re-created
        only if width/height change). Headless use needs an offscreen GL backend, e.g.
        `MUJOCO_GL=egl` (or osmesa) in the environment; a GUI/desktop GL context works too.

        If ``degrade`` (or ``self.camera_degrade`` when ``degrade is None``) is True, the frame is
        passed through the physically-based underwater degradation (``rendering.underwater_sim``)
        using the camera's depth buffer, so the perception input looks like real murky footage —
        distant red darkens/blues out, haze builds with distance. The water condition is
        ``water_params`` (or ``self.water_params``, or a moderately-murky default) — fixed for the
        run so the cast is stable; per-frame noise/caustics still vary.
        """
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        degrade = self.camera_degrade if degrade is None else degrade
        # Perception mode hides site/decoration markers (e.g. the pin_tip glyph) so only real geoms
        # show — a clean onboard view. Off-path (clean) keeps the default option for existing callers.
        opt = self._perception_scene_option() if degrade else None
        self._renderer.update_scene(self.data, camera=camera, scene_option=opt)
        rgb = self._renderer.render()
        if not degrade:
            return rgb
        from .rendering import underwater_sim as us
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera, scene_option=opt)
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        params = water_params or self.water_params or us.WaterParams()
        return us.degrade(rgb, depth, params, self._cam_rng)

    def _perception_scene_option(self):
        """Cached MjvOption for the perception camera: render real geoms only (no site/decoration
        glyphs — the pin_tip site would otherwise show as a marker in the onboard view)."""
        opt = getattr(self, "_perc_opt", None)
        if opt is None:
            opt = mujoco.MjvOption()
            opt.sitegroup[:] = 0  # no site markers
            for flag in ("mjVIS_CAMERA", "mjVIS_LIGHT", "mjVIS_JOINT", "mjVIS_ACTUATOR",
                         "mjVIS_CONTACTPOINT", "mjVIS_CONTACTFORCE", "mjVIS_INERTIA", "mjVIS_COM"):
                idx = getattr(mujoco.mjtVisFlag, flag, None)
                if idx is not None:
                    opt.flags[idx] = 0
            self._perc_opt = opt
        return opt


if __name__ == "__main__":
    # Smoke test: drift under buoyancy, then command forward thrust.
    sim = UmiusiSimulator()
    print(f"substeps/control-step = {sim.substeps}, dt = {sim.dt}")
    sim.reset(pos=(0.0, 1.0, 0.0))
    for i in range(100):
        act = np.zeros(8)
        if i >= 20:
            act[4:8] = 0.5  # all thrusters forward at half command
        s = sim.step(act)
        if i % 20 == 0:
            print(f"t={i:3d} pos={np.round(s['pos'], 3)} vel={np.round(s['lin_vel'], 3)}")
    print("OK")
