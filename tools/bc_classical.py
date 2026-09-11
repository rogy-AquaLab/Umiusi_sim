"""Behaviour-clone the CLASSICAL controller into an esc-action policy, as an RL warm start.

The question this exists to answer fairly: can a learned policy beat the classical controller once
it starts FROM it, instead of from scratch? Every RL-vs-classical number so far compared a tuned
classical controller against av_mode13, a policy trained in August with a reward bug since fixed
(the effort penalty was not cap-normalised), with no disturbances and — because domain_rand had no
failure knob until now — never having met a dead thruster, while the allocator was simply TOLD
which unit had died. That is not a fair fight, and it is not evidence about the architecture.

    ESC ACTION SPACE, NOT MODES. umiusi_rl/distill.py clones into the 6-D wrench-mode space, where
    ModeMixer expands the action. That caps the student at the mixer's behaviour, and the mixer is
    exactly where the measured problem is: holding station, the required per-unit force is nearly
    vertical, every servo sits on the +-90 deg fold, and the commanded servo rate exceeds the slew
    limit 15 % of the time. A mode-space student structurally cannot express the fix (the null
    space lives in the 8-D allocation, and the mixer chooses it for you). So clone into esc.

The teacher is the full classical stack: ClassicalController (cap-normalised PID + buoyancy trim +
thrust-model velocity observer) through GeneralAllocator with singularity avoidance. Fine-tune the
result with the failures and disturbances the teacher cannot adapt to:

    python tools/bc_classical.py --out bc_esc1 --steps 200000
    python -m umiusi_rl.train --action-mode esc --obs-frame rep103 --domain-rand --disturb \\
        --init-from models/bc_esc1 --run-name av_esc_ft1 --timesteps 10000000 --n-envs 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from classical_control import GeneralAllocator, _rep103, build_controller  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config  # noqa: E402
from umiusi_rl.train import build_model  # noqa: E402


def make_cfg(dr, disturb, dead_prob, action_mode="esc"):
    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"].update(task="attitude_velocity", action_mode=action_mode, obs_frame="rep103",
                      observe_max_duty=True)
    cfg.setdefault("domain_rand", {})["enabled"] = dr
    cfg["domain_rand"]["thrust_dead_prob"] = dead_prob
    cfg.setdefault("disturbance", {})["enabled"] = disturb
    return cfg


def rollout(cfg, steps, seed, gains, alloc_kw, label="esc", student=None, obs_rms=None, beta=1.0):
    """Roll out and record (obs, TEACHER action, reward, done).

    student=None is plain behaviour cloning: the teacher drives, so the states are the teacher's.
    With a student, this is one DAgger round — the STUDENT drives (with probability 1-beta per
    episode) while the teacher still labels every state it visits. That is the whole point: a clone
    fit only on the teacher's own trajectory has no data for the states its own small errors take
    it to, which is why better imitation loss stopped buying better flight (MSE 0.0261 -> 0.0055
    across parameterisation and capacity, closed-loop ori stuck at 0.16-0.26 against a teacher's
    0.083). The teacher's internal state (observer, allocator warm start, cap filter) runs along
    the visited trajectory, which is the only consistent way to query a stateful expert.
    """
    env = UmiusiPoseEnv(cfg)
    ctl = build_controller(env, **gains)
    alloc = GeneralAllocator(env.sim, **alloc_kw)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    n_obs = env.observation_space.shape[0]
    obs_buf = np.empty((steps, n_obs), dtype=np.float32)
    act_buf = np.empty((steps, 8), dtype=np.float32)
    rew_buf = np.empty(steps, dtype=np.float32)
    done_buf = np.zeros(steps, dtype=bool)
    obs, _ = env.reset(seed=seed)
    ctl.reset()
    alloc.reset()
    rng = np.random.default_rng(seed)
    teacher_drives = student is None or rng.random() < beta
    w, ep = np.zeros(6), 0
    for t in range(steps):
        v_hat = ctl.obs.update(obs[9:17], env.sim.get_state()["quat"], dt)
        m = ctl.wrench(obs[0:3], obs[3:6], obs[6:9], _rep103(v_hat), float(obs[17]))
        f_max_tot = ctl.f_max_total(ctl.cap)
        w_des = np.array([m[0], m[2], -m[1], m[3], m[5], -m[4]]) * f_max_tot
        w += np.clip(w_des - w, -0.25 * f_max_tot, 0.25 * f_max_tot)
        a_teacher = alloc.allocate(w, ctl.cap)
        label_a = np.clip(alloc.hv_prev, -1.0, 1.0) if label == "forces" else a_teacher
        obs_buf[t], act_buf[t] = obs, label_a
        if student is not None and not teacher_drives:
            o = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)
            act, _ = student.predict(o.astype(np.float32), deterministic=True)
        else:
            act = label_a if label == "forces" else a_teacher
        obs, r, term, trunc, _i = env.step(np.clip(act, -1.0, 1.0))
        rew_buf[t] = r
        if term or trunc:
            done_buf[t] = True
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
            ctl.reset()
            alloc.reset()
            teacher_drives = student is None or rng.random() < beta
            w = np.zeros(6)
        if (t + 1) % 20_000 == 0:
            print(f"[bc]   {t + 1}/{steps} steps, {ep} episodes")
    env.close()
    return obs_buf, act_buf, rew_buf, done_buf, ep


def mc_returns(rew, done, gamma):
    """Discounted return per step, cut at episode ends. The value head has to start somewhere and
    the teacher has no critic to copy; bootstrapping from its own reward is closer than zeros."""
    out = np.zeros_like(rew)
    run = 0.0
    for t in range(len(rew) - 1, -1, -1):
        run = rew[t] + (0.0 if done[t] else gamma * run)
        out[t] = run
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="student run name (models/<out>)")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-std", type=float, default=-1.0,
                    help="policy log_std after BC (SB3's 0.0 would swamp the clone with noise)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dr", action="store_true", help="collect without domain randomization")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--dead-prob", type=float, default=0.0,
                    help="P(a unit is dead) while COLLECTING. The teacher is not fault-aware here, "
                         "so this only teaches the student what a failure looks like, not how to "
                         "handle it — that is the fine-tune's job")
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.35)
    ap.add_argument("--prefer-deg", type=float, default=60.0)
    ap.add_argument("--w-move", type=float, default=3.0)
    ap.add_argument("--action-mode", default="forces", choices=["esc", "forces"],
                    help="forces = per-unit (h, v), continuous; esc = servo angle, discontinuous")
    ap.add_argument("--dagger-rounds", type=int, default=3,
                    help="extra rounds collected under the STUDENT's own state distribution "
                         "(0 = plain BC). Each round adds --steps//2 samples and refits.")
    args = ap.parse_args()

    cfg = make_cfg(not args.no_dr, args.disturb, args.dead_prob, args.action_mode)
    gains = {"kp": args.kp, "kd": args.kd}
    alloc_kw = {"prefer_deg": args.prefer_deg, "w_move": args.w_move}
    print(f"[bc] teacher rollout: {args.steps} steps  action_mode={args.action_mode} "
          f"gains={gains} alloc={alloc_kw} DR={not args.no_dr} disturb={args.disturb} "
          f"dead_prob={args.dead_prob} dagger_rounds={args.dagger_rounds}")
    obs_buf, act_buf, rew, done, ep = rollout(cfg, args.steps, args.seed, gains, alloc_kw,
                                              label=args.action_mode)
    print(f"[bc] collected {ep} episodes; |action| mean {np.abs(act_buf).mean():.3f}")

    # Normalization stats from the teacher's own state distribution — this is what the student
    # will see, and train.py --init-from loads and FREEZES these.
    obs_rms = RunningMeanStd(shape=obs_buf.shape[1:])
    obs_rms.update(obs_buf)
    ret = mc_returns(rew, done, cfg["ppo"]["gamma"])
    ret_rms = RunningMeanStd(shape=())
    ret_rms.update(ret)
    norm = np.clip((obs_buf - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0).astype(np.float32)
    val = (ret / np.sqrt(ret_rms.var + 1e-8)).astype(np.float32)   # PPO trains on normalized return

    run_dir = _ROOT / "models" / args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    venv = DummyVecEnv([lambda: UmiusiPoseEnv(cfg)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    venv.obs_rms, venv.ret_rms = obs_rms, ret_rms
    student = build_model("ppo", cfg, venv, args.seed, run_dir / "tb")

    print(f"[bc] BC: {args.epochs} epochs x {args.steps} samples (batch {args.batch_size})")
    policy = student.policy
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    obs_t, act_t, val_t = torch.as_tensor(norm), torch.as_tensor(act_buf), torch.as_tensor(val)
    for epoch in range(args.epochs):
        perm = torch.randperm(args.steps)
        a_sum = v_sum = nb = 0.0
        for i in range(0, args.steps, args.batch_size):
            idx = perm[i:i + args.batch_size]
            feat = policy.extract_features(obs_t[idx])
            lat_pi, lat_vf = policy.mlp_extractor(feat)
            a_loss = torch.nn.functional.mse_loss(policy.action_net(lat_pi), act_t[idx])
            v_loss = torch.nn.functional.mse_loss(policy.value_net(lat_vf).squeeze(-1), val_t[idx])
            opt.zero_grad()
            (a_loss + 0.5 * v_loss).backward()
            opt.step()
            a_sum, v_sum, nb = a_sum + a_loss.item(), v_sum + v_loss.item(), nb + 1
        print(f"[bc]   epoch {epoch + 1:2d}/{args.epochs}  action mse {a_sum / nb:.5f}  "
              f"value mse {v_sum / nb:.4f}")
    with torch.no_grad():
        policy.log_std.fill_(args.log_std)

    # ---- DAgger ---------------------------------------------------------------------------
    # Refit on the aggregated set after each round. beta is the probability the TEACHER drives an
    # episode; it decays so later rounds sample mostly the student's own states, which is where the
    # clone's errors actually live.
    for rnd in range(args.dagger_rounds):
        beta = 0.5 ** (rnd + 1)
        add = max(args.steps // 2, 1)
        print(f"[dagger] round {rnd + 1}/{args.dagger_rounds}  beta={beta:.2f}  +{add} steps")
        o2, a2, r2, d2, ep2 = rollout(cfg, add, args.seed + 1000 * (rnd + 1), gains, alloc_kw,
                                      label=args.action_mode, student=student, obs_rms=obs_rms,
                                      beta=beta)
        obs_buf = np.concatenate([obs_buf, o2])
        act_buf = np.concatenate([act_buf, a2])
        rew = np.concatenate([rew, r2])
        done = np.concatenate([done, d2])
        # Stats stay FROZEN at the round-0 estimate: train.py --init-from loads and freezes them,
        # so the student must be fit against the same normalisation it will be fine-tuned under.
        norm = np.clip((obs_buf - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0).astype(np.float32)
        ret = mc_returns(rew, done, cfg["ppo"]["gamma"])
        val = (ret / np.sqrt(ret_rms.var + 1e-8)).astype(np.float32)
        obs_t, act_t, val_t = torch.as_tensor(norm), torch.as_tensor(act_buf), torch.as_tensor(val)
        n = len(obs_t)
        for epoch in range(args.epochs):
            perm = torch.randperm(n)
            a_sum = nb = 0.0
            for i in range(0, n, args.batch_size):
                idx = perm[i:i + args.batch_size]
                feat = policy.extract_features(obs_t[idx])
                lat_pi, lat_vf = policy.mlp_extractor(feat)
                a_loss = torch.nn.functional.mse_loss(policy.action_net(lat_pi), act_t[idx])
                v_loss = torch.nn.functional.mse_loss(policy.value_net(lat_vf).squeeze(-1), val_t[idx])
                opt.zero_grad()
                (a_loss + 0.5 * v_loss).backward()
                opt.step()
                a_sum, nb = a_sum + a_loss.item(), nb + 1
        with torch.no_grad():
            policy.log_std.fill_(args.log_std)
        print(f"[dagger]   {ep2} episodes added, dataset {n}, action mse {a_sum / nb:.5f}")

    student.save(str(run_dir / "final.zip"))
    venv.save(str(run_dir / "vecnormalize.pkl"))
    with open(run_dir / "meta.yaml", "w") as f:
        yaml.safe_dump({
            "algo": "ppo", "config": "configs/train_ppo_mode_ft.yaml", "task": "attitude_velocity",
            "action_mode": args.action_mode, "obs_frame": "rep103", "obs_mode": "imu",
            "proprio_mode": "action", "observe_max_duty": True, "vecnormalize": True,
            "domain_rand": not args.no_dr, "disturbance": args.disturb,
            "tilt_target_deg": cfg["env"]["tilt_target_deg"],
            "yaw_target_deg": cfg["env"]["yaw_target_deg"],
            "vel_cmd_cone_deg": cfg["env"].get("vel_cmd_cone_deg"),
            "cloned_from": f"classical kp={args.kp} kd={args.kd} prefer_deg={args.prefer_deg}",
            "bc_steps": args.steps, "bc_epochs": args.epochs,
            "dagger_rounds": args.dagger_rounds,
        }, f, sort_keys=True)
    venv.close()
    print(f"[bc] done -> {run_dir}. Fine-tune with:\n"
          f"  python -m umiusi_rl.train --action-mode {args.action_mode} --obs-frame rep103 --domain-rand "
          f"--disturb --init-from models/{args.out} --run-name <name> "
          f"--timesteps 10000000 --n-envs 12")


if __name__ == "__main__":
    main()
