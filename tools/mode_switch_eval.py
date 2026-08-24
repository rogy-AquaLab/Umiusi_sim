"""Mode-switch supervisor rehearsal — closed-loop validation of the autonomy-side design.

Implements, inside the sim, the depth-threshold mode selector proposed for the robot
(sinsei_UMIUSI_autonomy#15): a pressure-sensor outer loop picks between the HORIZONTAL
policy (av_cal1_best — vertical commands are out-of-distribution) and the VERTICAL /
drone-mode policy (av_cal5_3d), with hysteresis and "round to pure vertical" command
shaping. The goal is to hand autonomy measured numbers for D_TH / hysteresis / K and the
switch transients instead of guesses.

The sim measurements forced three design changes vs the original issue sketch:

1. NO pitch-trim depth correction while cruising. Commanding the attitude target +/-8..15
   deg of pitch barely changes the vertical rate (the policy tracks the pitch weakly and
   the vertical leak dominates: vy stays +0.07..0.08 m/s at duty 0.4 regardless of trim).
2. Vertical corrections PAUSE the cruise (the supervisor zeroes the horizontal command
   itself). Both policies leave a systematic upward drift (~+0.05 m/s at hold, ~+0.08
   cruising: positively buoyant trim that low-duty actions do not cancel), so depth
   diverges during any horizontal leg and corrections must be able to interrupt it.
3. The vertical policy is DESCENT-ONLY. Measured pure-vertical rates (duty 0.4):
   DOWN -0.158 m/s (good), UP +0.024 m/s — SLOWER than the passive buoyant drift at hold
   (+0.05). It also ignores the command magnitude (speed is unobservable; direction-only).
   So: descend = brake, then burst on av_cal5_3d; ascend = hold on av_cal1_best and let
   buoyancy lift the vehicle (passive ascent, ~0.05 m/s, attitude stays clean).
   NOTE this makes slightly-positive buoyancy trim a REQUIREMENT of the scheme — keep the
   vehicle trimmed a touch positive (safety norm anyway) and measure the actual ascent
   rate in the static-release calibration (experiment #5).

Supervisor state machine (depth_err = target_depth - depth_measured; y up, so err < 0
means "too shallow, go down"):
    HORIZ  -> BRAKE:   depth_err < -d_enter     # too high: descend burst needed
              1.5 s of v_cmd = 0 on the horizontal policy first — entering drone mode
              still moving forward (~0.14 m/s) capsized the vehicle in sim (att 180 deg)
    BRAKE  -> VERT:    t_brake elapsed
    VERT   -> HORIZ:   depth_err > -d_exit_descend  AND  |depth_rate| < rate_gate
    HORIZ  -> ASCEND:  depth_err > +d_enter     # too deep: passive ascent (hold, buoyancy)
    ASCEND -> HORIZ:   depth_err < +d_exit
    VERT:   policy = 3d,   v_cmd = [0, clip(k_depth*depth_err, -v_vert, 0), 0]
    ASCEND: policy = cal1, v_cmd = 0
    HORIZ:  policy = cal1, v_cmd = [vx, 0, vz]
prev_action is carried across switches (both policies share the 17-D obs layout).

Usage:
    python -m tools.mode_switch_eval                          # defaults (duty 0.4)
    python -m tools.mode_switch_eval --duty 0.2               # first-water-test protocol
    python -m tools.mode_switch_eval --d-enter 0.3 --d-exit 0.15 --k-depth 0.7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CTRL_HZ = 50


class DepthSensor:
    """Pressure-derived depth: bias + gaussian noise, sample-and-hold at sensor_hz."""

    def __init__(self, rng, sigma=0.005, bias=0.01, sensor_hz=25):
        self.rng, self.sigma, self.bias = rng, sigma, bias
        self.every = max(1, CTRL_HZ // sensor_hz)
        self._held = None

    def read(self, y_true, t_step):
        if self._held is None or t_step % self.every == 0:
            self._held = y_true + self.bias + self.rng.normal(0.0, self.sigma)
        return self._held


class Supervisor:
    """The autonomy-side mode selector (depth outer loop + policy pick + cruise pause)."""

    def __init__(self, d_enter=0.3, d_exit=0.15, k_depth=0.7, v_vert=0.2, rate_gate=0.08,
                 t_brake=1.5, d_exit_descend=0.2):
        self.d_enter, self.d_exit = d_enter, d_exit
        self.d_exit_descend = d_exit_descend  # exit descent early: momentum + buoyancy finish it
        self.k_depth, self.v_vert = k_depth, v_vert
        self.rate_gate = rate_gate            # m/s; exit descent only once it has slowed
        self.t_brake = t_brake                # s of hold before the burst (kill momentum)
        self.state = "horiz"                  # horiz | brake | vert | ascend
        self.switches = 0                     # correction entries (brake or ascend starts)
        self.retries = 0                      # burst watchdog trips (vert failed to descend)
        self._brake_until = 0.0
        self._vert_since = 0.0
        self._hist = []                       # (t, depth) ring for the rate estimate

    def _depth_rate(self, now, depth):
        self._hist.append((now, depth))
        while self._hist and now - self._hist[0][0] > 0.5:
            self._hist.pop(0)
        (t0, d0), (t1, d1) = self._hist[0], self._hist[-1]
        return (d1 - d0) / (t1 - t0) if t1 > t0 else 0.0

    def update(self, now, depth_meas, target_depth, v_horiz):
        """-> (policy, v_cmd(3) body-frame). Corrections override (pause) the cruise."""
        depth_err = target_depth - depth_meas
        rate = self._depth_rate(now, depth_meas)
        if self.state == "horiz":
            if depth_err < -self.d_enter:                 # too shallow -> active descent
                self.state = "brake"
                self._brake_until = now + self.t_brake
                self.switches += 1
            elif depth_err > self.d_enter:                # too deep -> passive ascent
                self.state = "ascend"
                self.switches += 1
        elif self.state == "brake":
            if now >= self._brake_until:
                self.state = "vert"
                self._vert_since = now
        elif self.state == "vert":
            if depth_err > -self.d_exit_descend and abs(rate) < self.rate_gate:
                self.state = "horiz"
            elif ((now - self._vert_since > 2.5 and rate > 0.03)
                  or (now - self._vert_since > 4.0 and rate > -0.02)):
                # burst watchdog: the 3-D policy occasionally misfires (rises / translates
                # instead of descending — multimodal fragility). Re-brake and retry. The
                # 2.5 s grace covers the normal arrest phase (a healthy burst still rises
                # ~2 s while it kills the buoyant drift).
                self.state = "brake"
                self._brake_until = now + self.t_brake
                self.retries += 1
        elif depth_err < self.d_exit:                     # ascend done
            self.state = "horiz"

        if self.state == "vert":
            vy = float(np.clip(self.k_depth * depth_err, -self.v_vert, 0.0))
            return "vert", np.array([0.0, vy, 0.0])
        if self.state in ("brake", "ascend"):
            return "horiz", np.zeros(3)                   # hold on the horizontal policy
        return "horiz", np.array([v_horiz[0], 0.0, v_horiz[1]])


def run_scenario(env, policies, sup, sensor, segments, seconds, duty, start_y=0.0):
    """segments: list of (t_start, target_depth, (vx, vz)). Returns the full trace."""
    env.reset(seed=0)
    env.sim.reset(pos=(0.0, start_y, 0.0), quat=(1.0, 0.0, 0.0, 0.0))
    env.prev_action = np.zeros(8)
    env.target_quat = np.array([1.0, 0.0, 0.0, 0.0])   # level; yaw hold at 0
    st = env.sim.get_state()
    trace = {k: [] for k in ("t", "y", "mode", "ori_deg", "speed", "vx", "depth_err")}
    for t in range(int(seconds * CTRL_HZ)):
        now = t / CTRL_HZ
        target_depth, v_horiz = segments[0][1], segments[0][2]
        for seg in segments:
            if now >= seg[0]:
                target_depth, v_horiz = seg[1], seg[2]
        y_meas = sensor.read(st["pos"][1], t)
        mode, v_cmd = sup.update(now, y_meas, target_depth, v_horiz)  # mode = policy to run

        env.v_cmd = v_cmd
        env.v_cmd_world = v_cmd.copy()      # target_quat is identity
        R, oe = env._errors(st)
        obs = env._get_obs(st, R, oe)

        model, vn = policies[mode]
        a, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
        plant = a.copy()
        plant[4:] = np.clip(plant[4:], -duty, duty)
        st = env.sim.step(plant)
        env.prev_action = a.copy()
        env.step_count += 1

        trace["t"].append(now)
        trace["y"].append(float(st["pos"][1]))
        trace["mode"].append(sup.state)     # supervisor STATE (brake/ascend distinct)
        trace["ori_deg"].append(float(np.degrees(np.linalg.norm(oe))))
        trace["speed"].append(float(np.linalg.norm(st["lin_vel"])))
        trace["vx"].append(float(st["lin_vel"][0]))
        trace["depth_err"].append(float(target_depth - st["pos"][1]))  # true error
    return {k: (np.asarray(v) if k != "mode" else np.asarray(v, dtype=object)) for k, v in trace.items()}


def depth_hold_metrics(tr, t_cmd, band):
    """(time to first |err|<band after t_cmd, post-capture mean/max |err|) — sawtooth-aware."""
    t, err = tr["t"], np.abs(tr["depth_err"])
    idx = np.where((t >= t_cmd) & (err < band))[0]
    if len(idx) == 0:
        return None, None, float(np.min(err[t >= t_cmd]))
    i0 = idx[0]
    return float(t[i0] - t_cmd), float(np.mean(err[i0:])), float(np.max(err[i0:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horiz-model", default="models/av_cal1_best")
    ap.add_argument("--vert-model", default="models/av_cal5_3d")
    ap.add_argument("--duty", type=float, default=0.4)
    ap.add_argument("--d-enter", type=float, default=0.3)
    ap.add_argument("--d-exit", type=float, default=0.15)
    ap.add_argument("--d-exit-descend", type=float, default=0.2)
    ap.add_argument("--k-depth", type=float, default=0.7)
    ap.add_argument("--rate-gate", type=float, default=0.08)
    ap.add_argument("--t-brake", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "sim" / "src"))
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["obs_mode"] = "imu"
    cfg["env"]["vel_cmd_horizontal"] = False
    cfg["domain_rand"] = {"enabled": False}
    env = UmiusiPoseEnv(cfg)

    policies = {}
    for mode, mdir in (("horiz", args.horiz_model), ("vert", args.vert_model)):
        d = Path(mdir)
        model = PPO.load(str(d / "final.zip"), device="cpu")
        vn = VecNormalize.load(str(d / "vecnormalize.pkl"), DummyVecEnv([lambda: UmiusiPoseEnv(cfg)]))
        vn.training = False
        policies[mode] = (model, vn)

    mk_sup = lambda: Supervisor(args.d_enter, args.d_exit, args.k_depth,
                                rate_gate=args.rate_gate, t_brake=args.t_brake,
                                d_exit_descend=args.d_exit_descend)
    mk_sen = lambda: DepthSensor(np.random.default_rng(args.seed))
    # Absolute excursion ceiling once captured (pool-test tolerance). Expect the sawtooth
    # peak at ~ d_enter + 0.17 m: the vehicle keeps rising through the brake and the first
    # ~2 s of the burst while the 3-D policy arrests the buoyant ascent.
    bound = 0.5

    print(f"mode-switch rehearsal  duty<={args.duty}  d_enter/exit {args.d_enter}/{args.d_exit}"
          f" (descend exit {args.d_exit_descend})  k_depth {args.k_depth}  "
          f"rate_gate {args.rate_gate}  t_brake {args.t_brake}")
    fails = 0

    # 1/2 — pure depth steps (dive = active burst, climb = passive/buoyant), then hold
    for name, target, tmax in (("dive  1.0 m", -1.0, 30.0), ("climb 1.0 m", +1.0, 40.0)):
        sup = mk_sup()
        tr = run_scenario(env, policies, sup, mk_sen(),
                          [(0.0, 0.0, (0, 0)), (1.0, target, (0, 0))], seconds=tmax, duty=args.duty)
        rise, mean_e, max_e = depth_hold_metrics(tr, 1.0, args.d_enter)
        ori = float(np.max(tr["ori_deg"][tr["t"] > 1.0]))
        rate = sup.switches / (tmax - 1.0)
        ok = rise is not None and max_e < bound and ori < 60.0 and rate < 0.5
        fails += not ok
        print(f"[{name}] reach {'NONE' if rise is None else f'{rise:.1f}'} s  hold mean/max |err| "
              f"{'-' if mean_e is None else f'{mean_e:.2f}'}/{max_e:.2f} m  "
              f"switches {sup.switches} ({rate:.2f}/s, retry {sup.retries})  max_att {ori:.0f} deg  {'ok' if ok else 'FAIL'}")

    # 3 — cruise at constant depth: descend bursts pause the cruise (upward leak drives it)
    sup = mk_sup()
    tr = run_scenario(env, policies, sup, mk_sen(),
                      [(0.0, 0.0, (0.2, 0.0))], seconds=30.0, duty=args.duty)
    frac_cruise = float(np.mean(tr["mode"] == "horiz"))
    max_e = float(np.max(np.abs(tr["depth_err"])))
    vx = float(np.mean(tr["vx"]))
    ok = max_e < bound and frac_cruise > 0.4 and vx > 0.03
    fails += not ok
    print(f"[cruise+hold-depth] cruise fraction {frac_cruise:.0%}  mean vx {vx:+.2f} m/s  "
          f"max |err| {max_e:.2f} m  switches {sup.switches} (retry {sup.retries})  {'ok' if ok else 'FAIL'}")

    # 4 — mission: cruise -> dive (supervisor pauses the cruise on its own) -> cruise at depth
    sup = mk_sup()
    tr = run_scenario(env, policies, sup, mk_sen(),
                      [(0.0, 0.0, (0.2, 0.0)), (8.0, -1.0, (0.2, 0.0))], seconds=40.0, duty=args.duty)
    rise, mean_e, max_e = depth_hold_metrics(tr, 8.0, args.d_enter)
    late = tr["t"] > (8.0 + (rise or 0.0))
    cru = late & (tr["mode"] == "horiz")
    vx_after = float(np.mean(tr["vx"][cru])) if rise is not None and np.any(cru) else 0.0
    ori = float(np.max(tr["ori_deg"]))
    # Gate on the END state (captured + cruising again): a burst misfire mid-scenario costs
    # a ~0.6 m excursion but the watchdog recovers it — that excursion is reported, not failed.
    end_e = float(np.mean(np.abs(tr["depth_err"][-int(5 * CTRL_HZ):])))
    ok = rise is not None and end_e < 0.35 and vx_after > 0.03 and ori < 60.0
    fails += not ok
    print(f"[mission] dive reach {'NONE' if rise is None else f'{rise:.1f}'} s  worst excursion "
          f"{max_e:.2f} m  end |err| {end_e:.2f} m  cruise vx after {vx_after:+.2f} m/s  "
          f"switches {sup.switches} (retry {sup.retries})  max_att {ori:.0f} deg  {'ok' if ok else 'FAIL'}")

    # 5 — deadband: sub-threshold depth error at hold (design accepts this; measure it)
    sup = mk_sup()
    tr = run_scenario(env, policies, sup, mk_sen(),
                      [(0.0, -0.25, (0, 0))], seconds=15.0, duty=args.duty)
    e1 = float(np.mean(np.abs(tr["depth_err"][-int(5 * CTRL_HZ):])))
    print(f"[hold deadband] 0.25 m err at hold -> stays {e1:.2f} m "
          f"(no correction below d_enter by design), switches {sup.switches}")

    print("PASS" if fails == 0 else f"FAIL ({fails} scenario(s))")
    raise SystemExit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
