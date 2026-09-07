"""Sim2real robustness: does the vehicle still fly with a dead thruster or a servo bias?

The failures that actually happen on this robot are per-unit: a thruster stops producing thrust,
or a servo sits at a biased angle. Both break the LEARNED policy silently — it was trained on
four healthy units and has no way to represent "unit 3 is gone", so it keeps commanding a wrench
the plant cannot produce and the realised wrench is wrong in a way nothing detects.

Geometric allocation can represent it. Dropping a unit's columns and re-solving gives a 6x6 of
rank 6 (condition 8.1) for ANY single failure, so all six wrench DOF survive on three thrusters.
A known servo bias folds into the same solve.

    python tools/fault_compare.py                  # full matrix
    python tools/fault_compare.py --episodes 12
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from classical_control import ClassicalController, GeneralAllocator, _rep103  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config  # noqa: E402


def _env(action_mode, dr=False, disturb=False):
    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"].update(task="attitude_velocity", action_mode=action_mode, obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = dr
    cfg.setdefault("disturbance", {})["enabled"] = disturb
    env = UmiusiPoseEnv(cfg)
    env.vel_cmd_zero_prob = 1.0                 # hold-station: the regime the faults matter in
    return env


def _plant_fault(action, dead, servo_bias_rad, servo_range_rad):
    """What the PLANT does with the command: a dead unit makes no thrust, a biased servo sits off."""
    a = np.asarray(action, dtype=float).copy()
    if servo_bias_rad:
        a[:4] = np.clip(a[:4] + servo_bias_rad / servo_range_rad, -1.0, 1.0)
    if dead is not None:
        a[4 + dead] = 0.0
    return a


def run_classical(episodes, dead, bias_deg, aware, dr, disturb):
    env = _env("esc", dr, disturb)
    c = yaml.safe_load((_ROOT / "configs" / "umiusi.yaml").read_text())
    mass = c["hull"]["mass"] + 4 * c["thrusters"]["mass"]
    g = abs(c["sim"]["gravity"][1])
    net_buoy = c["water"]["density"] * c["water"]["displaced_volume"] * g - mass * g
    ctl = ClassicalController(env.sim, mass, net_buoy)
    live = np.ones(4, bool)
    if dead is not None and aware:
        live[dead] = False
    off = np.full(4, np.radians(bias_deg)) if (bias_deg and aware) else None
    alloc = GeneralAllocator(env.sim, servo_offset_rad=off, live=live)
    f_max_tot = 4.0 * env.sim.thrust_per_cmd * env.sim.max_duty ** env.sim.thrust_curve_exp
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    esc, ori, drift = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=5000 + ep)
        ctl.reset()
        w = np.zeros(6)
        done = False
        while not done:
            v_hat = ctl.obs.update(obs[9:17], env.sim.get_state()["quat"], dt)
            m = ctl.wrench(obs[0:3], obs[3:6], obs[6:9], _rep103(v_hat), float(obs[17]))
            # modes (REP-103) -> wrench in the SIM frame the allocator works in
            w_des = np.array([m[0], m[2], -m[1], m[3], m[5], -m[4]]) * f_max_tot
            w += np.clip(w_des - w, -0.25 * f_max_tot, 0.25 * f_max_tot)   # wrench slew
            a = alloc.allocate(w, env.sim.max_duty)
            obs, _r, term, trunc, info = env.step(
                _plant_fault(a, dead, np.radians(bias_deg), env.sim.servo_range_rad))
            esc.append(np.median(np.abs(info["esc_applied"])))
            if info.get("step_idx", 0) > 150:
                ori.append(float(info["ori_err"]))
            drift.append(float(info.get("vel_err", 0.0)))
            done = term or trunc
    env.close()
    return np.mean(esc), np.mean(ori), np.mean(drift)


def run_rl(episodes, dead, bias_deg, dr, disturb, run="av_mode13"):
    env = _env("modes", dr, disturb)
    d = _ROOT / "models" / run
    from stable_baselines3 import PPO
    model = PPO.load(str(d / "final.zip"), device="cpu")
    with open(d / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon
    raw_mix = env._mixer.mix

    def mix(modes, cap, prev):          # the plant fault happens AFTER the policy's mixer
        return _plant_fault(raw_mix(modes, cap, prev), dead, np.radians(bias_deg),
                            env.sim.servo_range_rad)
    env._mixer.mix = mix
    esc, ori, drift = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=5000 + ep)
        done = False
        while not done:
            o = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
            a, _ = model.predict(o, deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            esc.append(np.median(np.abs(info["esc_applied"])))
            if info.get("step_idx", 0) > 150:
                ori.append(float(info["ori_err"]))
            drift.append(float(info.get("vel_err", 0.0)))
            done = term or trunc
    env.close()
    return np.mean(esc), np.mean(ori), np.mean(drift)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--disturb", action="store_true")
    args = ap.parse_args()
    cases = [
        ("健全",                    None, 0.0, True),
        ("lf 故障 / 未知",           0,   0.0, False),
        ("lf 故障 / 既知",           0,   0.0, True),
        ("サーボ +5deg / 未補償",    None, 5.0, False),
        ("サーボ +5deg / 補償",      None, 5.0, True),
        ("lf 故障 + サーボ / 既知",  0,   5.0, True),
    ]
    print(f"hold-station  episodes={args.episodes}  DR={args.dr} disturb={args.disturb}")
    print(f"{'条件':<26}{'制御':<10}{'median|esc|':>12}{'ori[rad]':>10}{'横流れ':>10}")
    for label, dead, bias, aware in cases:
        e, o, d = run_classical(args.episodes, dead, bias, aware, args.dr, args.disturb)
        print(f"{label:<26}{'古典':<10}{e:12.4f}{o:10.3f}{d:10.4f}")
        if aware:                       # RL has no "aware" variant — it cannot represent a fault
            e, o, d = run_rl(args.episodes, dead, bias, args.dr, args.disturb)
            print(f"{'':<26}{'RL(m13)':<10}{e:12.4f}{o:10.3f}{d:10.4f}")


if __name__ == "__main__":
    main()
