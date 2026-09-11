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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from umiusi_perception.classical import cad_wrench_from_modes  # noqa: E402
from classical_control import GeneralAllocator, _rep103, build_controller  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config  # noqa: E402


def _env(action_mode, dr=False, disturb=False, dr_cfg=None, plant=None):
    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"].update(task="attitude_velocity", action_mode=action_mode, obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = dr
    if dr_cfg is not None:      # per-knob override, for attributing WHICH part of DR hurts
        cfg["domain_rand"] = {**cfg["domain_rand"], **dr_cfg, "enabled": True}
    cfg.setdefault("disturbance", {})["enabled"] = disturb
    env = UmiusiPoseEnv(cfg)
    if plant:
        # The plant constants live in a SEPARATE yaml the simulator loads by path. _base is the
        # restore point, but _apply_domain_rand only reads it on the DR-OFF branch — with DR on it
        # writes each knob only when that knob's range is set. So patch BOTH, and the caller must
        # also null those ranges via dr_cfg or DR will overwrite the override every reset.
        _SIM_ATTR = {"servo_slew": "servo_slew_rad", "servo_tau": "servo_tau",
                     "thrust_slew": "thrust_slew", "thrust_per_cmd": "thrust_per_cmd",
                     "max_duty": "max_duty", "thrust_exp": "thrust_curve_exp"}
        env._base.update(plant)
        for k, v in plant.items():
            setattr(env.sim, _SIM_ATTR[k], v)
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


def run_classical(episodes, dead, bias_deg, aware, dr, disturb, gains=None, cruise=False,
                  dr_cfg=None, alloc=None, plant=None):
    env = _env("esc", dr, disturb, dr_cfg, plant)
    if cruise:
        env.vel_cmd_zero_prob = 0.0
    ctl = build_controller(env, **(gains or {}))
    live = np.ones(4, bool)
    if dead is not None and aware:
        live[dead] = False
    off = np.full(4, np.radians(bias_deg)) if (bias_deg and aware) else None
    alloc = GeneralAllocator(env.sim, servo_offset_rad=off, live=live, **(alloc or {}))
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    esc, ori, drift, verr, rew = [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=5000 + ep)
        ctl.reset()
        alloc.reset()
        w = np.zeros(6)
        done = False
        while not done:
            v_hat = ctl.obs.update(obs[9:17], env.sim.get_state()["quat"], dt)
            m = ctl.wrench(obs[0:3], obs[3:6], obs[6:9], _rep103(v_hat), float(obs[17]))
            # modes (REP-103) -> wrench in the SIM frame the allocator works in. The mode unit is
            # the full-cap wrench at the cap ctl.wrench normalized by — its FILTERED estimate, not
            # the raw noisy obs, or the two disagree by the noise every step.
            cap = ctl.cap
            f_max_tot = ctl.f_max_total(cap)
            w_des = cad_wrench_from_modes(m, f_max_tot)
            w += np.clip(w_des - w, -0.25 * f_max_tot, 0.25 * f_max_tot)   # wrench slew
            a = alloc.allocate(w, cap)
            obs, _r, term, trunc, info = env.step(
                _plant_fault(a, dead, np.radians(bias_deg), env.sim.servo_range_rad))
            rew.append(float(_r))
            esc.append(np.median(np.abs(info["esc_applied"])))
            if info.get("step_idx", 0) > 150:
                ori.append(float(info["ori_err"]))
            drift.append(float(info.get("vel_err", 0.0)))
            verr.append(float(np.linalg.norm(v_hat - env.sim.get_state()["lin_vel"])))
            done = term or trunc
    env.close()
    return {"esc": float(np.mean(esc)), "ori": float(np.mean(ori)),
            "drift": float(np.mean(drift)), "verr": float(np.mean(verr)),
            "rew": float(np.mean(rew))}


def run_rl(episodes, dead, bias_deg, dr, disturb, run="av_mode13", cruise=False,
           action_mode="modes"):
    env = _env(action_mode, dr, disturb)
    if cruise:
        env.vel_cmd_zero_prob = 0.0
    d = _ROOT / "models" / run
    from stable_baselines3 import PPO
    model = PPO.load(str(d / "final.zip"), device="cpu")
    with open(d / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon
    # The fault is a property of the ACTUATOR, so it must be applied to [servo, esc] — after any
    # conversion. Only an esc-action policy emits that directly; "modes" and "forces" both go
    # through a mixer first, and injecting into their raw action would corrupt a wrench component
    # (in "forces" a[:4] is the horizontal force, not a servo angle).
    if action_mode == "modes":
        raw_mix = env._mixer.mix

        def mix(modes, cap, prev):
            return _plant_fault(raw_mix(modes, cap, prev), dead, np.radians(bias_deg),
                                env.sim.servo_range_rad)
        env._mixer.mix = mix
    elif action_mode == "forces":
        raw_f2a = env._force_mixer.forces_to_action

        def f2a(h, v, cap, prev):
            return _plant_fault(raw_f2a(h, v, cap, prev), dead, np.radians(bias_deg),
                                env.sim.servo_range_rad)
        env._force_mixer.forces_to_action = f2a
    esc, ori, drift, rew = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=5000 + ep)
        done = False
        while not done:
            o = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
            a, _ = model.predict(o, deterministic=True)
            if action_mode == "esc":   # raw actuator command: the fault applies to it directly
                a = _plant_fault(a, dead, np.radians(bias_deg), env.sim.servo_range_rad)
            obs, _r, term, trunc, info = env.step(a)
            rew.append(float(_r))
            esc.append(np.median(np.abs(info["esc_applied"])))
            if info.get("step_idx", 0) > 150:
                ori.append(float(info["ori_err"]))
            drift.append(float(info.get("vel_err", 0.0)))
            done = term or trunc
    env.close()
    return np.mean(esc), np.mean(ori), np.mean(drift), np.mean(rew)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--kp", type=float, default=1.0)      # tools/classical_tune.py --stage fine
    ap.add_argument("--kd", type=float, default=0.35)
    ap.add_argument("--ki", type=float, default=0.0)
    ap.add_argument("--k-v", type=float, default=1.2)
    ap.add_argument("--cruise", action="store_true",
                    help="sample velocity commands instead of hold-station. The learned policy's "
                         "claimed edge is continuous motion, and the hold-only matrix never tested it")
    ap.add_argument("--rl-run", default="av_mode13")
    ap.add_argument("--rl-action-mode", default="modes", choices=["modes", "esc", "forces"],
                    help="MUST match the policy's meta.yaml action_mode. A forces policy scored "
                         "in esc mode has its 8 outputs read as [servo, esc] instead of (h, v) — "
                         "it tumbles (ori 1.9 rad) and the number means nothing.")
    ap.add_argument("--prefer-deg", type=float, default=None,
                    help="azimuth-singularity avoidance (tools/alloc_rate_audit.py); 60 is the measured best")
    ap.add_argument("--w-move", type=float, default=3.0)
    args = ap.parse_args()
    gains = {"kp": args.kp, "kd": args.kd, "ki": args.ki, "k_v": args.k_v}
    alloc = None if args.prefer_deg is None else {"prefer_deg": args.prefer_deg, "w_move": args.w_move}
    cases = [
        ("健全",                    None, 0.0, True),
        ("lf 故障 / 未知",           0,   0.0, False),
        ("lf 故障 / 既知",           0,   0.0, True),
        ("サーボ +5deg / 未補償",    None, 5.0, False),
        ("サーボ +5deg / 補償",      None, 5.0, True),
        ("lf 故障 + サーボ / 既知",  0,   5.0, True),
    ]
    print(f"{'cruise' if args.cruise else 'hold-station'}  episodes={args.episodes}  "
          f"DR={args.dr} disturb={args.disturb}  gains={gains}  alloc={alloc}  rl={args.rl_run}")
    print(f"{'条件':<26}{'制御':<10}{'median|esc|':>12}{'ori[rad]':>10}{'横流れ':>10}{'報酬/step':>11}")
    for label, dead, bias, aware in cases:
        r = run_classical(args.episodes, dead, bias, aware, args.dr, args.disturb,
                          gains=gains, cruise=args.cruise, alloc=alloc)
        print(f"{label:<26}{'古典':<10}{r['esc']:12.4f}{r['ori']:10.3f}{r['drift']:10.4f}"
              f"{r['rew']:11.3f}")
        if aware:                       # the policy has no "aware" variant — it cannot be TOLD
            e, o, d, rw = run_rl(args.episodes, dead, bias, args.dr, args.disturb,
                                 run=args.rl_run, cruise=args.cruise,
                                 action_mode=args.rl_action_mode)
            print(f"{'':<26}{'RL':<10}{e:12.4f}{o:10.3f}{d:10.4f}{rw:11.3f}")


if __name__ == "__main__":
    main()
