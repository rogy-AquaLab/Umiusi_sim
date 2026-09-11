"""Non-learned baseline: geometric allocation + PD attitude + a thrust-model velocity observer.

We have been tuning RL for weeks with NO non-learned baseline to compare against. This is that
baseline. It reuses the parts that are already exact and replaces only the part RL was doing:

    ModeMixer          6-D wrench -> 8-D (servo, esc). Already pure kinematics — atan2 fold,
                       per-unit force split, null modes absent from the basis. NOT relearned.
    this controller    what wrench to command. RL's actual job, done classically instead.

Two pathologies of the learned policy are structural and disappear here by construction:
  * hovering at ~90 % of the esc cap — the required wrench is computed, not discovered;
  * commanding heave UPWARD on a positively buoyant vehicle — buoyancy is a known constant
    (+1.17 N), so the trim term is exact.

The velocity observer exists because the deployed observation has no lateral velocity
(obs = ori_err, gyro, v_cmd, prev_action, max_duty — no DVL, no position). A one-step MLP
cannot integrate the force history, so the learned policy could not estimate drift even though
prev_action carries the information. Integrating it explicitly is cheap and self-limiting:
velocity is drag-dominated, so the estimate converges to a bias set by model error rather than
drifting without bound.

    ACCURACY IS GATED BY CALIBRATION. thrust_per_cmd and thrust_curve_exp are both uncalibrated
    and come from a contaminated fit (docs/calibration_plan.md). The observer cannot be better
    than they are; bench calibration (§3) is what improves it.

THE CONTROLLER ITSELF LIVES IN `umiusi_perception.classical`, not here. That package is the only
one installed on the robot, so the ROS node and this script drive the SAME object rather than two
implementations that have to be kept in agreement — every sim2real failure recorded on this
project was an interface bug, and a port is a new interface. This file is the sim-side adapter:
it turns a `UmiusiSimulator` into the `PlantContract` the deployed controller takes
(`contract_from_sim`), and runs the diagnostic below. `tools/export_classical.py` writes that
same contract to JSON for the Pi.

Usage:
    python tools/classical_control.py --episodes 12            # hold-station diagnostic
    python tools/classical_control.py --episodes 12 --cruise   # with velocity commands
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "packages" / "perception" / "src"))

from umiusi_perception.classical import ClassicalController as _ClassicalController  # noqa: E402
from umiusi_perception.classical import GeneralAllocator as _GeneralAllocator  # noqa: E402
from umiusi_perception.classical import (PlantContract, VelocityObserver,  # noqa: E402,F401
                                         f_max_total, rep103_from_cad)
from umiusi_rl.envs.mode_mixer import MODE_NAMES  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config  # noqa: E402


def contract_from_sim(sim, mass=None, net_buoy_up=None, cap_ref=None):
    """Build the DEPLOY contract from this simulator — the one place the two sides meet.

    This is the sim-side half of `umiusi_perception.classical`: it snapshots the calibrated
    constants and the CAD geometry into a `PlantContract`, and from there on the controller that
    runs here is byte-for-byte the controller that runs on the Pi. `tools/export_classical.py`
    writes the same object to JSON for the robot.

    Two things are computed rather than copied, because only the simulator can:
      * `pivots_from_com` — the thruster pivots relative to the CENTRE OF MASS, which needs the
        full mass tree (`subtree_com`). Using the body origin instead silently mis-scales every
        moment arm, and the attitude loop would be tuned around the error.
      * mass / net buoyancy default to the NOMINAL config, never the episode's randomized values
        (`build_controller`), for the same reason the plant constants are snapshotted.

    domain_rand REBINDS sim.thrust_per_cmd / thrust_curve_exp / drag / added mass on every reset.
    Reading them through `sim` at control time let the controller see the very truth it is
    supposed to be robust to, which silently flattered every DR number. `max_duty` is NOT in here:
    the cap is observed (obs[17]), so it is an input — only `cap_ref`, the cap the gains are
    quoted at, is a constant.
    """
    if mass is None or net_buoy_up is None:
        m, b = _hull_constants()
        mass = m if mass is None else mass
        net_buoy_up = b if net_buoy_up is None else net_buoy_up
    mujoco.mj_forward(sim.model, sim.data)
    com = sim.data.subtree_com[sim.base_id] - sim.data.xpos[sim.base_id]
    com_local = sim.data.xmat[sim.base_id].reshape(3, 3).T @ com
    return PlantContract(
        thrust_per_cmd=float(sim.thrust_per_cmd),
        thrust_curve_exp=float(sim.thrust_curve_exp),
        servo_range_rad=float(sim.servo_range_rad),
        thrust_axes=np.array(sim.thrust_axes, dtype=float),
        pivots_from_com=np.asarray(sim.unit_pivots, dtype=float) - com_local,
        drag_lin=np.array(sim.drag_lin[:3], dtype=float),
        drag_quad=np.array(sim.drag_quad[:3], dtype=float),
        added_mass_diag=np.array(sim.added_mass_diag[:3], dtype=float),
        mass=float(mass),
        net_buoy_up=float(net_buoy_up),
        control_rate_hz=float(sim.cfg["sim"]["control_rate_hz"]),
        cap_ref=float(sim.max_duty if cap_ref is None else cap_ref),
    )


# Historical name: the tools were written against `nominal_plant(sim)` and only ever read fields
# off it, which a PlantContract also provides.
nominal_plant = contract_from_sim


def _hull_constants():
    """(mass, net buoyancy up) [kg, N] from configs/umiusi.yaml — the NOMINAL config."""
    c = yaml.safe_load((_ROOT / "configs" / "umiusi.yaml").read_text())
    mass = c["hull"]["mass"] + 4 * c["thrusters"]["mass"]
    g = abs(c["sim"]["gravity"][1])
    return mass, c["water"]["density"] * c["water"]["displaced_volume"] * g - mass * g


def body_inertia_tensor(sim, r):
    """Full 3x3 rotational inertia in the BODY frame: hull + thrusters + rotational added mass.

    Two traps, both of which silently halve or rotate the predicted angular acceleration:
      * `model.body_inertia` is diagonal in the body's PRINCIPAL frame, and `body_iquat` here is a
        ~120 deg rotation, not identity. Dividing a body-frame moment by it elementwise mixes axes.
      * underwater the fluid entrained by a rotating hull is not negligible next to the structure
        (added_mass_diag[3:6] vs body_inertia ~ 0.08-0.17 vs 0.19-0.47), so it belongs here.
    Thrusters enter by the parallel-axis theorem as point masses at their pivots.
    """
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(sim.model.body_iquat[sim.base_id], dtype=float))
    R = R.reshape(3, 3)
    tensor = R @ np.diag(np.asarray(sim.model.body_inertia[sim.base_id], dtype=float)) @ R.T
    m_t = float(sim.model.body_mass[sim.thr_ids[0]])
    for k in range(4):
        d = np.asarray(r[k], dtype=float)
        tensor = tensor + m_t * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return tensor + np.diag(np.asarray(sim.added_mass_diag[3:6], dtype=float))




class GeneralAllocator(_GeneralAllocator):
    """Sim-side adapter — builds the contract from `sim`, then IS the deployed allocator.

    The allocation itself lives in `umiusi_perception.classical` so that the robot runs this code
    and not a port of it. Keep it that way: a second implementation on the ROS side is exactly the
    interface that produced the pitch/yaw swap (2026-08-21) and three more bugs of the same class.
    """

    def __init__(self, sim, **kw):
        super().__init__(contract_from_sim(sim), **kw)


class ClassicalController(_ClassicalController):
    """Sim-side adapter for the deployed controller. See `GeneralAllocator`."""

    def __init__(self, sim, mass=None, net_buoy_up=None, **kw):
        super().__init__(contract_from_sim(sim, mass=mass, net_buoy_up=net_buoy_up), **kw)


def build_controller(env, **gains):
    """A ClassicalController for this env: mass and net buoyancy come from configs/umiusi.yaml.

    The nominal config, not the episode's randomized plant — same reason as `contract_from_sim`.
    """
    return ClassicalController(env.sim, **gains)


# The CAD -> REP-103 swap has exactly one implementation, and it is the deployed one. Keeping a
# second copy here is how `fault_detect` came to feed world-frame omega into a body-frame model.
_rep103 = rep103_from_cad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--cruise", action="store_true", help="sample velocity commands (default: hold)")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--domain-rand", action="store_true",
                    help="model mismatch — the regime RL was trained for; the fair comparison")
    ap.add_argument("--kp", type=float, default=2.2)
    ap.add_argument("--kd", type=float, default=0.45)
    ap.add_argument("--ki", type=float, default=0.0)
    ap.add_argument("--k-v", type=float, default=1.2)
    args = ap.parse_args()

    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"].update(task="attitude_velocity", action_mode="modes", obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = args.domain_rand
    cfg.setdefault("disturbance", {})["enabled"] = args.disturb
    env = UmiusiPoseEnv(cfg)
    if not args.cruise:
        env.vel_cmd_zero_prob = 1.0

    ctl = build_controller(env, kp=args.kp, kd=args.kd, ki=args.ki, k_v=args.k_v)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    step = env._mode_slew_step or 1.0

    signed, absm, esc, ori, drift, verr = [], [], [], [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=5000 + ep)
        ctl.reset()
        m = np.zeros(6)
        done = False
        while not done:
            # obs は実機が持つものだけ: [ori_err 3][gyro 3][v_cmd 3][prev_action 8][max_duty 1]
            ori_err, gyro, v_cmd_o = obs[0:3], obs[3:6], obs[6:9]
            prev_action, cap = obs[9:17], float(obs[17])
            v_hat_sim = ctl.obs.update(prev_action, env.sim.get_state()["quat"], dt)
            m_des = ctl.wrench(ori_err, gyro, v_cmd_o, _rep103(v_hat_sim), cap)
            rate = np.clip((m_des - m) / step, -1.0, 1.0)
            obs, _r, term, trunc, info = env.step(rate)
            m = env._mode_prev_modes.copy()
            signed.append(m)
            absm.append(np.abs(m))
            esc.append(np.median(np.abs(info["esc_applied"])))
            if info.get("step_idx", 0) > 150:
                ori.append(float(info["ori_err"]))
            drift.append(float(info.get("vel_err", 0.0)))
            verr.append(float(np.linalg.norm(v_hat_sim - env.sim.get_state()["lin_vel"])))
            done = term or trunc
    env.close()

    sg, ab = np.array(signed).mean(0), np.array(absm).mean(0)
    print(f"classical  kp={args.kp} kd={args.kd} ki={args.ki} k_v={args.k_v}  "
          f"{'cruise' if args.cruise else 'hold-station'}  disturb={args.disturb} DR={args.domain_rand}")
    print(f"  median|esc| {np.mean(esc):.4f}   ori {np.mean(ori):.3f} rad   横流れ {np.mean(drift):.4f} m/s")
    print(f"  観測器の速度誤差 {np.mean(verr):.4f} m/s   (真値との差、較正で決まる)")
    for i, n in enumerate(MODE_NAMES):
        r = abs(sg[i]) / ab[i] if ab[i] > 1e-6 else 0.0
        print(f"   {n:3s}  符号付き {sg[i]:+.3f}   |m| {ab[i]:.3f}   比 {r:.2f}")


if __name__ == "__main__":
    main()
