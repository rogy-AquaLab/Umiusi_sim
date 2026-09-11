"""Does the allocator ask the servos for motion they cannot deliver?

An azimuth-x4 vehicle has EIGHT actuator DOF for a SIX-DOF wrench, so every wrench has a 2-D
family of solutions. `GeneralAllocator` picks one of them by minimum norm, MEMORYLESSLY — it
re-solves from scratch each step with no knowledge of where the servos currently are or how fast
they can move (the plant tracks with rate = clip((target-angle)/tau, +-slew), slew 250 deg/s
nominal and 100-500 under DR, simulator.py:157). Two consequences the min-norm choice cannot see:

  * RATE. A small change in the desired wrench can move the min-norm solution a long way in servo
    angle. If the demanded rate exceeds the slew limit the servo lags, and the wrench the vehicle
    actually produces is not the one that was solved for.
  * THE FOLD. `allocate` folds |phi| > 90 deg to the rear by negating the esc. Near that boundary
    an arbitrarily small wrench change flips the servo by ~180 deg. That discontinuity is a real
    suspect for the measured anomaly that a COMMON +5 deg servo bias IMPROVES attitude hold
    (0.061 -> 0.049 without DR, 0.265 -> 0.137 with) — a bias moves the operating point off it.

This is exactly the structure a learned policy could exploit and this allocator cannot: the policy
sees prev_action, so it can pick the null-space member that is CHEAP TO REACH from where the
servos already are. So measure the gap before arguing about it.

    python tools/alloc_rate_audit.py --episodes 8
    python tools/alloc_rate_audit.py --episodes 8 --dr --cruise
    python tools/alloc_rate_audit.py --episodes 8 --rl av_mode13     # the policy, same metrics
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
from fault_compare import _env  # noqa: E402


def _stats(name, servo_cmd, servo_act, slew_deg_s, dt, esc_cmd):
    """servo_cmd/servo_act: [T, 4] commanded and achieved servo angles [rad]."""
    cmd = np.asarray(servo_cmd)
    rate = np.abs(np.diff(cmd, axis=0)) / dt                      # demanded [rad/s]
    rate_deg = np.degrees(rate)
    lag = np.degrees(np.abs(cmd[1:] - np.asarray(servo_act)[1:]))  # command vs achieved [deg]
    # a fold shows up as a near-180 deg demand inside ONE control step: no servo delivers that
    folds = np.mean(np.degrees(np.abs(np.diff(cmd, axis=0))) > 90.0)
    over = np.mean(rate_deg > slew_deg_s)
    esc = np.asarray(esc_cmd)
    esc_rate = np.abs(np.diff(esc, axis=0)) / dt
    print(f"{name:<12}"
          f"{np.mean(rate_deg):9.1f}{np.percentile(rate_deg, 95):9.1f}"
          f"{over * 100:8.1f}%{folds * 100:8.2f}%"
          f"{np.mean(lag):9.2f}{np.percentile(lag, 95):9.2f}"
          f"{np.mean(esc_rate):9.2f}")


def run_classical(episodes, dr, disturb, cruise, gains, **alloc_kw):
    env = _env("esc", dr, disturb)
    if cruise:
        env.vel_cmd_zero_prob = 0.0
    ctl = build_controller(env, **gains)
    alloc = GeneralAllocator(env.sim, **alloc_kw)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    srange = env.sim.servo_range_rad
    cmd, act, esc = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=5000 + ep)
        ctl.reset()
        alloc.reset()
        w = np.zeros(6)
        done = False
        while not done:
            v_hat = ctl.obs.update(obs[9:17], env.sim.get_state()["quat"], dt)
            m = ctl.wrench(obs[0:3], obs[3:6], obs[6:9], _rep103(v_hat), float(obs[17]))
            f_max_tot = ctl.f_max_total(ctl.cap)
            w_des = cad_wrench_from_modes(m, f_max_tot)
            w += np.clip(w_des - w, -0.25 * f_max_tot, 0.25 * f_max_tot)
            a = alloc.allocate(w, ctl.cap)
            cmd.append(a[:4] * srange)
            esc.append(a[4:8])
            obs, _r, term, trunc, _i = env.step(a)
            act.append(env.sim.get_state()["servo"])
            done = term or trunc
    slew = np.degrees(env.sim.servo_slew_rad)
    env.close()
    return cmd, act, esc, dt, slew


def run_rl(episodes, dr, disturb, cruise, run):
    env = _env("modes", dr, disturb)
    if cruise:
        env.vel_cmd_zero_prob = 0.0
    d = _ROOT / "models" / run
    from stable_baselines3 import PPO
    model = PPO.load(str(d / "final.zip"), device="cpu")
    with open(d / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon
    srange = env.sim.servo_range_rad
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    raw_mix, cmd, esc = env._mixer.mix, [], []

    def mix(modes, cap, prev):          # the mixer IS the policy's allocator; tap its output
        a = raw_mix(modes, cap, prev)
        cmd.append(np.asarray(a[:4]) * srange)
        esc.append(np.asarray(a[4:8]))
        return a
    env._mixer.mix = mix
    act = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=5000 + ep)
        done = False
        while not done:
            o = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
            a, _ = model.predict(o, deterministic=True)
            obs, _r, term, trunc, _i = env.step(a)
            act.append(env.sim.get_state()["servo"])
            done = term or trunc
    slew = np.degrees(env.sim.servo_slew_rad)
    env.close()
    return cmd[:len(act)], act, esc[:len(act)], dt, slew


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--cruise", action="store_true")
    ap.add_argument("--rl", default=None, help="also audit this policy run (e.g. av_mode13)")
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.35)
    args = ap.parse_args()

    print(f"servo 指令レート監査  DR={args.dr} disturb={args.disturb} "
          f"{'cruise' if args.cruise else 'hold'}  episodes={args.episodes}")
    print(f"{'制御':<12}{'平均deg/s':>9}{'p95':>9}{'限界超':>9}{'折返し':>9}"
          f"{'追従誤差':>9}{'p95':>9}{'esc変化':>9}")
    g = {"kp": args.kp, "kd": args.kd}
    for label, kw in (("古典(現行)", {}),
                      ("回避45 w0.3", {"prefer_deg": 45.0, "w_move": 0.3}),
                      ("回避45 w1", {"prefer_deg": 45.0, "w_move": 1.0}),
                      ("回避45 w3", {"prefer_deg": 45.0, "w_move": 3.0}),
                      ("回避60 w3", {"prefer_deg": 60.0, "w_move": 3.0}),
                      ("回避75 w3", {"prefer_deg": 75.0, "w_move": 3.0})):
        c = run_classical(args.episodes, args.dr, args.disturb, args.cruise, g, **kw)
        _stats(label, c[0], c[1], c[4], c[3], c[2])
    if args.rl:
        r = run_rl(args.episodes, args.dr, args.disturb, args.cruise, args.rl)
        _stats(f"RL({args.rl})", r[0], r[1], r[4], r[3], r[2])
    print(f"\n  サーボ速度限界 {c[4]:.0f} deg/s (nominal 250, DR で 100-500)。"
          f"「限界超」= 1 step の要求レートが限界を超えた割合、"
          f"「折返し」= 1 step で 90deg 超を要求した割合。")


if __name__ == "__main__":
    main()
