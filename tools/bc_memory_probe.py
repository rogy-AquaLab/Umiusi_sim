"""Does the student need MEMORY to reproduce the classical teacher at all?

Behaviour cloning into the deployed 18-D observation plateaued at action MSE 0.0223 and the clone
flew at ori 0.346 where its teacher flies at 0.091. Fine-tuning that start with PPO for 10M steps
reached 0.165 — better, still far off, and it collapsed into servo bang-bang (68 deg per control
step, 13x the slew limit). Both failures point at the same thing rather than at the algorithm:

    THE TEACHER IS NOT A FUNCTION OF THE OBSERVATION.

It carries state the observation does not expose — the velocity observer's v-hat, the allocator's
previous null-space offset, the cap low-pass, the attitude integral — and its map is discontinuous
(a grid search, the 180 deg fold, the dead zone). A memoryless MLP cannot represent that, so no
amount of RL on top of it is measuring what we wanted to measure.

This settles that question WITHOUT touching SB3, the deploy contract or a 4-hour training run: fit
a plain MLP on a stack of the last N observations and check, in CLOSED LOOP, how close it gets to
the teacher. Action MSE alone is not the test — the loop can amplify a small imitation error — so
the reported number is the rollout the clone actually flies.

    python tools/bc_memory_probe.py --steps 60000 --stacks 1,2,4,8

If ori falls toward the teacher as N grows, memory is the missing ingredient and a frame-stacked
(or recurrent) policy is the next RL attempt. If it does not, the gap is elsewhere and RL should
not be retried on this observation at all.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from bc_classical import make_cfg  # noqa: E402
from classical_control import GeneralAllocator, _rep103, build_controller  # noqa: E402
from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv  # noqa: E402


def collect(cfg, steps, seed, gains, alloc_kw, label):
    """Teacher rollout recording BOTH action parameterisations of the same behaviour.

    label="esc"    the 8-D [servo, esc] command — what the deployed actuator takes, and what the
                   earlier clone was fit to. DISCONTINUOUS: the +-90 deg fold means two nearly
                   identical states can want servo angles 180 deg apart.
    label="forces" the per-unit (h, v) force the allocator solved for, normalised by the cap force
                   — the same behaviour with the fold removed, and the action space of
                   `action_mode: "forces"`. Continuous, so it is fittable.
    """
    env = UmiusiPoseEnv(cfg)
    ctl = build_controller(env, **gains)
    alloc = GeneralAllocator(env.sim, **alloc_kw)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    n_obs = env.observation_space.shape[0]
    obs_buf = np.empty((steps, n_obs), dtype=np.float32)
    act_buf = np.empty((steps, 8), dtype=np.float32)
    done_buf = np.zeros(steps, dtype=bool)
    obs, _ = env.reset(seed=seed)
    ctl.reset()
    alloc.reset()
    w, ep = np.zeros(6), 0
    for t in range(steps):
        v_hat = ctl.obs.update(obs[9:17], env.sim.get_state()["quat"], dt)
        m = ctl.wrench(obs[0:3], obs[3:6], obs[6:9], _rep103(v_hat), float(obs[17]))
        f = ctl.f_max_total(ctl.cap)
        w += np.clip(np.array([m[0], m[2], -m[1], m[3], m[5], -m[4]]) * f - w, -0.25 * f, 0.25 * f)
        a = alloc.allocate(w, ctl.cap)
        obs_buf[t] = obs
        act_buf[t] = np.clip(alloc.hv_prev, -1.0, 1.0) if label == "forces" else a
        obs, _r, term, trunc, _i = env.step(a)
        if term or trunc:
            done_buf[t] = True
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
            ctl.reset()
            alloc.reset()
            w = np.zeros(6)
    env.close()
    return obs_buf, act_buf, done_buf, ep


def stack_dataset(obs, done, n):
    """[T, D] -> [T, n*D]: the last n observations, oldest first, zero-padded at episode starts."""
    t, d = obs.shape
    out = np.zeros((t, n * d), dtype=np.float32)
    start = 0
    for i in range(t):
        if i > 0 and done[i - 1]:
            start = i
        for k in range(n):
            j = i - (n - 1 - k)
            if j >= start:
                out[i, k * d:(k + 1) * d] = obs[j]
    return out


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_out, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.Tanh()]
            prev = h
        layers += [torch.nn.Linear(prev, d_out)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit(x, y, epochs, batch, lr, hidden=(256, 256)):
    model = MLP(x.shape[1], y.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    last = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        tot = nb = 0.0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            loss = torch.nn.functional.mse_loss(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        last = tot / nb
    return model, last


def closed_loop(model, cfg, n, mean, std, episodes=4, seed0=5000, label="esc"):
    """Fly the clone. This, not the imitation loss, is the number that matters."""
    cfg = {**cfg, "env": {**cfg["env"], "action_mode": "forces" if label == "forces" else "esc"}}
    env = UmiusiPoseEnv(cfg)
    env.vel_cmd_zero_prob = 1.0
    d = env.observation_space.shape[0]
    ori, drift, esc = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        hist = [np.zeros(d, dtype=np.float32) for _ in range(n)]
        done = False
        while not done:
            hist = hist[1:] + [obs.astype(np.float32)]
            x = (np.concatenate(hist) - mean) / std
            with torch.no_grad():
                a = model(torch.as_tensor(x, dtype=torch.float32)[None]).numpy()[0]
            obs, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
            esc.append(np.median(np.abs(info["esc_applied"])))
            if info.get("step_idx", 0) > 150:
                ori.append(float(info["ori_err"]))
            drift.append(float(info.get("vel_err", 0.0)))
            done = term or trunc
    env.close()
    return np.mean(ori), np.mean(drift), np.mean(esc)


def teacher_reference(cfg, gains, alloc_kw, episodes=4, seed0=5000):
    env = UmiusiPoseEnv(cfg)
    env.vel_cmd_zero_prob = 1.0
    ctl = build_controller(env, **gains)
    alloc = GeneralAllocator(env.sim, **alloc_kw)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    ori, drift, esc = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        ctl.reset()
        alloc.reset()
        w = np.zeros(6)
        done = False
        while not done:
            v_hat = ctl.obs.update(obs[9:17], env.sim.get_state()["quat"], dt)
            m = ctl.wrench(obs[0:3], obs[3:6], obs[6:9], _rep103(v_hat), float(obs[17]))
            f = ctl.f_max_total(ctl.cap)
            w += np.clip(np.array([m[0], m[2], -m[1], m[3], m[5], -m[4]]) * f - w, -0.25 * f, 0.25 * f)
            obs, _r, term, trunc, info = env.step(alloc.allocate(w, ctl.cap))
            esc.append(np.median(np.abs(info["esc_applied"])))
            if info.get("step_idx", 0) > 150:
                ori.append(float(info["ori_err"]))
            drift.append(float(info.get("vel_err", 0.0)))
            done = term or trunc
    env.close()
    return np.mean(ori), np.mean(drift), np.mean(esc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--stacks", default="1,2,4,8")
    ap.add_argument("--label", default="esc", choices=["esc", "forces"],
                    help="action parameterisation to clone (see `collect`)")
    ap.add_argument("--hidden", default="256x256",
                    help="comma-separated MLP sizes to try, e.g. 256x256,512x512x512")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.35)
    ap.add_argument("--prefer-deg", type=float, default=60.0)
    ap.add_argument("--w-move", type=float, default=3.0)
    args = ap.parse_args()

    cfg = make_cfg(not args.no_dr, args.disturb, 0.0)
    gains = {"kp": args.kp, "kd": args.kd}
    alloc_kw = {"prefer_deg": args.prefer_deg, "w_move": args.w_move}
    print(f"[probe] teacher rollout {args.steps} steps  label={args.label} "
          f"DR={not args.no_dr} disturb={args.disturb}")
    obs, act, done, ep = collect(cfg, args.steps, 0, gains, alloc_kw, args.label)
    print(f"[probe] {ep} episodes collected")
    t_ori, t_drift, t_esc = teacher_reference(cfg, gains, alloc_kw, args.episodes)

    print(f"\n{'stack':<7}{'hidden':<16}{'模倣MSE':>10}{'ori':>9}{'横流れ':>10}{'esc':>8}"
          f"   (教師 ori {t_ori:.3f} / 横流れ {t_drift:.4f} / esc {t_esc:.3f})")
    for n in [int(v) for v in args.stacks.split(",")]:
        x = stack_dataset(obs, done, n)
        mean, std = x.mean(0), x.std(0) + 1e-6
        for hs in args.hidden.split(","):
            hidden = tuple(int(h) for h in hs.split("x"))
            model, mse = fit((x - mean) / std, act, args.epochs, args.batch_size, args.lr, hidden)
            o, dr, e = closed_loop(model, cfg, n, mean, std, args.episodes, label=args.label)
            print(f"{n:<7}{hs:<16}{mse:10.5f}{o:9.3f}{dr:10.4f}{e:8.3f}")


if __name__ == "__main__":
    main()
