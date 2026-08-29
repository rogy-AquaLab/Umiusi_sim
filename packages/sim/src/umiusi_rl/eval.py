"""Evaluate / watch a trained go-to-pose policy on UmiusiPoseEnv.

Usage:
    python -m umiusi_rl.eval --model models/ppo/final.zip                 # headless metrics
    python -m umiusi_rl.eval --model models/ppo/final.zip --episodes 20   # more episodes
    python -m umiusi_rl.eval --model models/ppo/final.zip --render        # watch in the GUI viewer

The algorithm and config are read from the run's meta.yaml when present; override with
--algo / --config. Reports per-episode return, final position error, and the fraction of
steps spent within the goal tolerance (station-keeping quality).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from umiusi_rl.envs.umiusi_pose_env import VEL_PER_CAP, UmiusiPoseEnv, load_config

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="path to a trained policy .zip")
    ap.add_argument("--config", default=None, help="env config (default: from run meta.yaml)")
    ap.add_argument("--algo", choices=list(ALGOS), default=None, help="default: from run meta.yaml")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--domain-rand", action="store_true",
                    help="evaluate under domain randomization too (default: nominal model, disturbance from meta)")
    ap.add_argument("--no-disturb", action="store_true",
                    help="force disturbances OFF at eval (isolate the policy's own steadiness/wobble)")
    ap.add_argument("--legacy-hydro", action="store_true",
                    help="disable the higher-fidelity hydro (lift + CoP offset + coupling) so the sim uses "
                         "the old diagonal-drag model — for A/B testing a policy against the new physics")
    ap.add_argument("--max-duty", type=float, default=None,
                    help="pin the plant's esc cap for this eval (default: configs value). Use to "
                         "check a cap-conditioned (observe_max_duty) policy ACTUALLY adapts: run "
                         "at 0.2 / 0.25 / 0.4 and compare — identical behaviour means the cap "
                         "input was never learned")
    ap.add_argument("--render", action="store_true", help="watch in the MuJoCo GUI viewer")
    ap.add_argument("--record", default=None,
                    help="write an mp4 of the rollout (headless; run with MUJOCO_GL=egl)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--vecnormalize", default=None,
                    help="explicit VecNormalize stats .pkl (default: <model dir>/vecnormalize.pkl). "
                         "For mid-run checkpoints pass checkpoints/ppo_vecnormalize_<N>_steps.pkl")
    args = ap.parse_args()

    model_path = Path(args.model)
    meta_path = model_path.parent / "meta.yaml"
    if not meta_path.exists():  # mid-run checkpoint: checkpoints/ppo_<N>_steps.zip -> run dir meta
        meta_path = model_path.parent.parent / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    algo = args.algo or meta.get("algo", "ppo")
    config = args.config or meta.get("config", "configs/train_ppo.yaml")

    cfg = load_config(config)
    # match the task + sensor suite + curriculum condition the policy was trained with
    for k in ("task", "obs_mode", "proprio_mode", "obs_frame", "action_mode",
              "vel_cmd_cone_deg", "yaw_target_deg", "tilt_target_deg"):
        if meta.get(k) is not None:
            cfg["env"][k] = meta[k]
    if meta:  # obs-contract key: absent in old runs = trained WITHOUT the cap dim — never let the
        # (newer) config file grow the obs vector under an old policy
        cfg["env"]["observe_max_duty"] = bool(meta.get("observe_max_duty", False))
    if meta.get("disturbance") and not args.no_disturb:  # evaluate under the same disturbances it trained with
        cfg.setdefault("disturbance", {})["enabled"] = True
    elif args.no_disturb:  # isolate the policy's own steadiness (no current/impulses)
        cfg.setdefault("disturbance", {})["enabled"] = False
    # Domain randomization is OFF at eval by default (measure clean performance); --domain-rand tests
    # robustness to model mismatch (randomized buoyancy/thrust/drag + obs noise + action latency).
    cfg.setdefault("domain_rand", {})["enabled"] = bool(args.domain_rand)
    if args.max_duty is not None:
        cfg.setdefault("sim_config", "configs/umiusi.yaml")
        # per-episode DR must not overwrite the pinned cap
        cfg.setdefault("domain_rand", {}).pop("max_duty_range", None)
    env = UmiusiPoseEnv(cfg, render_mode="human" if args.render else None)
    if args.max_duty is not None:
        env.sim.max_duty = float(args.max_duty)
        env._base["max_duty"] = float(args.max_duty)
        print(f"[eval] esc cap pinned: max_duty = {args.max_duty}")
    if args.legacy_hydro:  # revert to the old diagonal-drag model (no lift / CoP moment / coupling)
        env.sim.lift_coef = 0.0
        env.sim.cop_offset[:] = 0.0
        env.sim.coupling_sway_yaw[:] = 0.0
        env.sim.coupling_heave_pitch[:] = 0.0
        print("[eval] legacy hydro: lift + CoP offset + coupling DISABLED (old diagonal-drag model)")
    else:
        print(f"[eval] new hydro: lift coef={env.sim.lift_coef}, cop_offset={list(env.sim.cop_offset)}")
    control_dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]

    recorder, frames = None, []
    if args.record:
        import mujoco  # local import; only needed when recording

        recorder = mujoco.Renderer(env.sim.model, 480, 640)
    model = ALGOS[algo].load(str(model_path), device="cpu")

    # Reapply the training-time observation normalization (VecNormalize stats), if any.
    stats_path = Path(args.vecnormalize) if args.vecnormalize else model_path.parent / "vecnormalize.pkl"
    if stats_path.exists():
        _dummy = DummyVecEnv([lambda: UmiusiPoseEnv(cfg)])
        vn = VecNormalize.load(str(stats_path), _dummy)
        _dummy.close()
        rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon

        def norm_obs(o):
            return np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
    else:
        def norm_obs(o):
            return o

    returns, pos_errs, ori_errs, depth_errs, vel_errs, hold_fracs, successes = [], [], [], [], [], [], []
    vel_alongs, vel_cmds, vel_reach = [], [], []
    thrust_uses, servo_motions, thrust_changes, ang_speeds = [], [], [], []
    null_fracs, roll_uses, esc_all = [], [], []   # allocation acceptance metrics (Umiusi_sim#3)
    mode_rates = []                               # mean |mode rate action| (rate-action runs)
    vert_powers = []                              # per-step vertical mode power (null weighting)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_ret, steps, in_tol = 0.0, 0, 0
        thrust_sum, servo_mot_sum, thrust_chg_sum, ang_speed_sum = 0.0, 0.0, 0.0, 0.0
        prev_servo, prev_esc = None, None
        done = False
        while not done:
            action, _ = model.predict(norm_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            steps += 1
            in_tol += int(info.get("is_success", False))
            esc, servo = info["esc_applied"], info["servo"]  # slew-limited actual thrust
            ang_speed_sum += float(info.get("ang_speed", 0.0))
            thrust_sum += float(np.mean(np.abs(esc)))
            null_fracs.append(float(info.get("null_frac", 0.0)))
            mode_rates.append(float(info.get("mode_rate_mag", 0.0)))
            roll_uses.append(float(info.get("roll_use", 0.0)))
            vert_powers.append(float(info.get("vert_power", 0.0)))
            esc_all.extend(np.abs(esc).tolist())
            if prev_esc is not None:
                thrust_chg_sum += float(np.mean(np.abs(esc - prev_esc)))
                servo_mot_sum += float(np.mean(np.abs(servo - prev_servo)))
            prev_esc, prev_servo = esc, servo
            if args.render:
                env.render()
                time.sleep(control_dt)
            if recorder is not None:
                recorder.update_scene(env.sim.data, camera="track")
                frames.append(recorder.render())
            done = terminated or truncated
        returns.append(ep_ret)
        pos_errs.append(info["pos_err"])
        ori_errs.append(info["ori_err"])
        depth_errs.append(abs(info["depth_err"]))
        vel_errs.append(info.get("vel_err", 0.0))
        vel_alongs.append(info.get("vel_along", 0.0))
        vel_cmds.append(info.get("vel_cmd_speed", 0.0))
        vel_reach.append(min(info.get("vel_cmd_speed", 0.0), VEL_PER_CAP * env.sim.max_duty))
        hold_fracs.append(in_tol / max(steps, 1))
        successes.append(info.get("is_success", False))
        thrust_uses.append(thrust_sum / max(steps, 1))
        ang_speeds.append(ang_speed_sum / max(steps, 1))  # mean ||angular velocity|| [rad/s]
        servo_motions.append(np.degrees(servo_mot_sum / max(steps - 1, 1)))  # deg/step
        thrust_changes.append(thrust_chg_sum / max(steps - 1, 1))
        print(f"ep {ep:2d}: return={ep_ret:8.1f}  ori_err={info['ori_err']:.3f} rad  "
              f"pos_err={info['pos_err']:.3f} m  depth_err={abs(info['depth_err']):.3f} m  "
              f"hold={hold_fracs[-1] * 100:4.0f}%")

    env.close()
    if recorder is not None:
        import imageio

        imageio.mimsave(args.record, frames, fps=round(1.0 / control_dt))
        recorder.close()
        print(f"wrote {args.record}  ({len(frames)} frames)")
    print("-" * 64)
    print(f"episodes={args.episodes}  task={meta.get('task', '?')}  algo={algo}  obs_mode={meta.get('obs_mode', '?')}")
    print(f"mean return        : {np.mean(returns):8.1f} +/- {np.std(returns):.1f}")
    print(f"mean final ori err : {np.mean(ori_errs):.3f} rad")
    print(f"mean final pos err : {np.mean(pos_errs):.3f} m")
    print(f"mean final depth err: {np.mean(depth_errs):.3f} m")
    print(f"attitude_velocity  : speed along cmd {np.mean(vel_alongs):.3f} / desired {np.mean(vel_cmds):.3f} m/s"
          f"   sideways drift {np.mean(vel_errs):.3f} m/s")
    # "Cruise formed" acceptance is judged against the PHYSICALLY REACHABLE speed at the episode's
    # cap (open-loop ceiling ~ VEL_PER_CAP * max_duty), not the raw command — commands above the
    # ceiling are unsatisfiable by any policy (measured 2026-08-26, Umiusi_sim#3).
    if np.mean(vel_reach) > 1e-9:
        print(f"cruise vs reachable: {np.mean(vel_alongs) / np.mean(vel_reach) * 100:.0f}%   "
              f"(along / min(desired, {VEL_PER_CAP:.2f}*cap); ACCEPT >= 70%)")
    print(f"mean hold fraction : {np.mean(hold_fracs) * 100:.0f}%   (steps within tolerance)")
    print(f"final-step success : {np.mean(successes) * 100:.0f}%")
    print(f"mean thrust use    : {np.mean(thrust_uses):.3f}   (mean |esc|, 0..1 -> minimize)")
    print(f"median |esc|       : {np.median(esc_all):.3f}   (accept: <= the deploy max_duty)")
    print(f"mean null share    : {np.mean(null_fracs) * 100:.1f}%   (null mode / vertical power; accept <= 5%,"
          f" real 8/25 run: 41.2%)")
    # Power-weighted null share = the ACCEPTANCE metric (Umiusi_sim#3): the per-step mean above
    # over-counts near-zero-power steps whose ratio is numerical noise.
    wp = np.array(vert_powers)
    nf = np.array(null_fracs)
    if wp.sum() > 1e-12:
        print(f"null share (pw)    : {float((nf * wp).sum() / wp.sum()) * 100:.1f}%   "
              f"(power-weighted; ACCEPT <= 5%)")
    print(f"mean roll authority: {np.mean(roll_uses) * 100:.1f}%   (roll mode / cap max; accept >= 50%,"
          f" real 8/25 run: 19%)")
    print(f"mean angular vel   : {np.mean(ang_speeds):.3f} rad/s   (wobble -> minimize)")
    print(f"mean servo motion  : {np.mean(servo_motions):.2f} deg/step   (vibration -> minimize)")
    if np.any(np.array(mode_rates) > 0.0):
        print(f"mean |mode rate|   : {np.mean(mode_rates):.3f}   (rate action; 1.0 = riding the slew limit)")
    print(f"mean thrust change : {np.mean(thrust_changes):.3f} /step     (|Δesc| -> minimize)")


if __name__ == "__main__":
    main()
