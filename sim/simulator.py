"""UmiusiSimulator — MuJoCo stepping + analytical hydrodynamics + thruster forces.

Single physics implementation shared by the validation tool, the RL env, and the ROS 2
bridge. Exposes a clean reset()/step(action)/get_state() API.

Action (8-D, each in [-1, 1]):
    [servo_1..4, esc_1..4]
    servo_k -> target azimuth angle = servo_k * servo_range (rate-limited)
    esc_k   -> thrust = esc_k * thrust_per_cmd  [N]

Frame: CAD frame, +Y up. Units: SI, radians internally.
"""

from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim.physics import hydrodynamics as hydro
from sim.physics import thruster as thr

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL = _ROOT / "sim" / "assets" / "umiusi.xml"
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

        # Thrusters
        t = cfg["thrusters"]
        self.servo_range_rad = np.radians(max(abs(v) for v in t["servo_range_deg"]))
        self.servo_slew_rad = np.radians(t["servo_slew_deg_per_s"])
        self.thrust_per_cmd = float(t["thrust_per_cmd"])
        # Per-thruster neutral thrust direction (thruster body frame). The MJCF servo hinge (about
        # the mounting arm) rotates the body, so this axis, carried by the body, tilts with the servo.
        self.thrust_axes = np.array([u["thrust_axis"] for u in t["units"]], dtype=float)

        # Indices
        self.base_id = self.model.body("base_link").id
        self.thr_ids = [self.model.body(f"thruster_{i}").id for i in (1, 2, 3, 4)]
        self.site_ids = [self.model.site(f"t{i}_thrust").id for i in (1, 2, 3, 4)]
        self.act_ids = [self.model.actuator(f"servo_{i}").id for i in (1, 2, 3, 4)]
        self.servo_qadr = [self.model.jnt_qposadr[self.model.joint(f"servo_{i}").id] for i in (1, 2, 3, 4)]

        # Center of buoyancy: horizontally over the whole-vehicle CoM (base + thrusters),
        # `buoyancy_offset` above it, expressed in the base body frame (rotates with the hull).
        mujoco.mj_forward(self.model, self.data)
        R0 = self.data.xmat[self.base_id].reshape(3, 3)
        sys_com_local = R0.T @ (self.data.subtree_com[self.base_id] - self.data.xpos[self.base_id])
        self.cob_local = sys_com_local + np.array([0.0, self.buoyancy_offset, 0.0])

        self.reset()

    # -- lifecycle -------------------------------------------------------------
    def reset(self, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = pos
        self.data.qpos[3:7] = quat
        for a in self.servo_qadr:
            self.data.qpos[a] = 0.0
        self.servo_ctrl = np.zeros(4)
        self.thrust_mag = np.zeros(4)
        self.prev_vel_body = np.zeros(6)
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(8)
        servo_target = np.clip(action[:4], -1.0, 1.0) * self.servo_range_rad
        self.thrust_mag = np.array(
            [thr.command_to_thrust(c, self.thrust_per_cmd) for c in action[4:8]]
        )
        for _ in range(self.substeps):
            self.servo_ctrl = thr.slew(self.servo_ctrl, servo_target, self.servo_slew_rad, self.dt)
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
        lin_body = R.T @ d.subtree_linvel[base]
        ang_body = R.T @ vel6[:3]
        vel_body = np.concatenate([lin_body, ang_body])
        w = hydro.drag_wrench_body(vel_body, self.drag_lin, self.drag_quad)
        if np.any(self.added_mass_diag):  # optional; off by default (needs numerical care)
            acc_body = (vel_body - self.prev_vel_body) / self.dt
            w = w + hydro.added_mass_wrench_body(acc_body, self.added_mass_diag)
        self.prev_vel_body = vel_body
        sys_com = d.subtree_com[base].copy()
        mujoco.mj_applyFT(m, d, R @ w[:3], R @ w[3:], sys_com, base, d.qfrc_applied)

        # Thrusters: thrust along the (servo-rotated) local axis, applied at each tip site.
        for k in range(4):
            bid, sid = self.thr_ids[k], self.site_ids[k]
            f_thr = thr.thrust_to_world(self.thrust_mag[k], self.thrust_axes[k], d.xmat[bid])
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
        }


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
