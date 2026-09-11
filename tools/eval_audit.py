"""Is the evaluation itself measuring what we think it is?

This project has a history of numbers that looked good and then did not survive contact with the
hardware, so before another architecture decision rests on `tools/fault_compare.py`, audit the
measurement. Four ways a mean attitude error can be a lie:

  * SATURATION. If the esc sits at the duty cap or the servo at its travel limit, the controller
    is not tracking, it is doing the most it can — and the error number then describes the plant,
    not the controller. A cap that is too generous hides it the other way: everything looks easy.
  * EARLY TERMINATION. `ori` is only accumulated after step 150. An episode that ends at step 200
    (out of bounds) contributes 50 samples of whatever it was doing on the way out, and a
    controller that leaves the workspace often is scored on a short, unrepresentative window.
  * COVERAGE. "hold" forces v_cmd = 0 and "cruise" samples it, but the mission is to hold an
    ARBITRARY attitude WHILE pushing in an ARBITRARY direction. If tilt and v_cmd never co-occur,
    or the tilt distribution collapses to near-upright, the hard case is simply never measured.
  * DIRECTION. Cruise is scored by perpendicular drift, which says nothing about whether the
    vehicle went the commanded way at all. A controller that stands still scores a perfect drift.

    python tools/eval_audit.py --episodes 8
    python tools/eval_audit.py --episodes 8 --cruise
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "sim" / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from umiusi_perception.classical import cad_wrench_from_modes  # noqa: E402
from classical_control import GeneralAllocator, _rep103, build_controller  # noqa: E402
from fault_compare import _env  # noqa: E402


def audit(episodes, dr, disturb, cruise, gains, alloc_kw, seed0=5000):
    env = _env("esc", dr, disturb)
    if cruise:
        env.vel_cmd_zero_prob = 0.0
    ctl = build_controller(env, **gains)
    alloc = GeneralAllocator(env.sim, **alloc_kw)
    dt = 1.0 / env.sim.cfg["sim"]["control_rate_hz"]
    m = {k: [] for k in ("esc_sat", "servo_sat", "ori", "tilt", "along", "speed", "cmd_speed")}
    ends, lens, scored = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        ctl.reset()
        alloc.reset()
        w, n_scored, step = np.zeros(6), 0, 0
        done = False
        while not done:
            v_hat = ctl.obs.update(obs[9:17], env.sim.get_state()["quat"], dt)
            mo = ctl.wrench(obs[0:3], obs[3:6], obs[6:9], _rep103(v_hat), float(obs[17]))
            f = ctl.f_max_total(ctl.cap)
            w += np.clip(cad_wrench_from_modes(mo, f) - w,
                         -0.25 * f, 0.25 * f)
            a = alloc.allocate(w, ctl.cap)
            obs, _r, term, trunc, info = env.step(a)
            st = env.sim.get_state()
            cap = env.sim.max_duty
            # saturation of the PLANT, not of the normalised command: |esc| at the duty cap
            m["esc_sat"].append(float(np.mean(np.abs(info["esc_applied"]) >= 0.995 * cap)))
            m["servo_sat"].append(float(np.mean(np.abs(a[:4]) >= 0.995)))
            # how far from upright the TARGET is this episode (the difficulty actually sampled)
            q = env.target_quat if hasattr(env, "target_quat") else None
            if q is not None:
                up = np.array([0.0, 1.0, 0.0])
                R = np.zeros(9)
                import mujoco
                mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
                m["tilt"].append(float(np.degrees(np.arccos(np.clip(R.reshape(3, 3) @ up @ up,
                                                                    -1.0, 1.0)))))
            v = st["lin_vel"]                      # WORLD frame (mj_objectVelocity flg_local=0)
            # env.v_cmd is in the TARGET-BODY frame; v_cmd_world is the same command rotated to
            # world, which is the only one comparable with lin_vel. Dotting the body-frame command
            # with a world-frame velocity gave 0.007 m/s and looked like "the vehicle never moves".
            vc = env.v_cmd_world
            nc = float(np.linalg.norm(vc))
            if nc > 1e-6:
                m["along"].append(float(v @ (vc / nc)))
                m["cmd_speed"].append(nc)
            m["speed"].append(float(np.linalg.norm(v)))
            if info.get("step_idx", 0) > 150:
                m["ori"].append(float(info["ori_err"]))
                n_scored += 1
            step += 1
            done = term or trunc
        ends.append("term" if term else "trunc")
        lens.append(step)
        scored.append(n_scored)
    env.close()
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in m.items()}
    out["early"] = float(np.mean([e == "term" for e in ends]))
    out["ep_len"] = float(np.mean(lens))
    out["scored"] = float(np.mean(scored))
    out["ori_p95"] = float(np.percentile(m["ori"], 95)) if m["ori"] else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--cruise", action="store_true")
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.35)
    args = ap.parse_args()
    gains = {"kp": args.kp, "kd": args.kd}

    cases = [("DR無 外乱無 minnorm", False, False, None),
             ("DR無 外乱無 回避60", False, False, {"prefer_deg": 60.0, "w_move": 3.0}),
             ("DR+外乱 minnorm", True, True, None),
             ("DR+外乱 回避60", True, True, {"prefer_deg": 60.0, "w_move": 3.0})]
    print(f"評価の健全性監査  {'cruise' if args.cruise else 'hold'}  episodes={args.episodes}")
    print(f"{'条件':<20}{'ori':>7}{'p95':>7}{'esc飽和':>8}{'servo飽和':>9}"
          f"{'早期終了':>8}{'採点数':>7}{'目標傾斜':>8}{'指令速度':>9}{'達成速度':>9}{'達成率':>8}")
    for label, dr, dist, ak in cases:
        r = audit(args.episodes, dr, dist, args.cruise, gains, ak or {})
        print(f"{label:<20}{r['ori']:7.3f}{r['ori_p95']:7.3f}{r['esc_sat'] * 100:7.1f}%"
              f"{r['servo_sat'] * 100:8.1f}%{r['early'] * 100:7.0f}%"
              f"{r['scored']:7.0f}{r['tilt']:8.1f}{r['cmd_speed']:9.3f}{r['along']:9.3f}"
              f"{r['along'] / max(r['cmd_speed'], 1e-9) * 100:7.0f}%")
    print("\n  esc飽和 = 適用 |esc| が duty cap の 99.5% 以上の step 割合 (制御していない印)。"
          "\n  早期終了 = 範囲外で terminate した割合。採点数 = ori に使われた step 数 (満点 450)。"
          "\n  達成率 = 指令方向の実速度 / 指令速度。100% 未満は「指令ほど進んでいない」、"
          "0 に近ければ「止まっているだけ」で横流れ指標が無意味になる。")


if __name__ == "__main__":
    main()
