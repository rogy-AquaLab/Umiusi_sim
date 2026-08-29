#!/usr/bin/env python3
"""学習済みポリシー bundle を SB3 非依存の素形式へ書き出し、SB3 と出力一致を検証する。

書き出し対象は下の `POL` 定数で指すバンドル(使うときに対象バンドルへ向ける)。
現在の配備バンドルは REP-103 変換済みの av_cal1_best_rep103 / att_cal1_best_rep103 /
av_sim2real2_rep103 / av_cal5_3d_rep103(降下専用・EXPERIMENTAL)である。

出力: <bundle>/export/{weights.pt, obs_norm.npz, meta.json}
実機側は torch だけで推論できる(SB3/gymnasium/cloudpickle 不要)。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from umiusi_rl.envs.mode_mixer import DEADBAND_FRAC, MODE_NAMES, _MODE_SIGNS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces

# 実機側で動く推論実装は sinsei_UMIUSI_autonomy/tools/policy_infer.py に一本化してある
# (重複を避けるため、ここでは検証時にそこから import する)。
_ROOT = Path(__file__).resolve().parents[1]          # umiusi_sim repo root (config paths)
AUTONOMY = Path("../ros2_ws/src/sinsei_UMIUSI_autonomy").resolve()
# POL = 書き出す対象バンドル。argv[1] で上書きできる (既定は最初に書き出した cruise_policy)。
POL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else AUTONOMY / "umiusi_rl_control/models/cruise_policy"
OUT = POL / "export"
# 25 = 旧 proprio_mode "full" (imu 6 + v_cmd 3 + servo 4 + thrust 4 + prev_action 8)。
# proprio_mode "action" で学習したポリシーは 17 (imu 6 + v_cmd 3 + prev_action 8)。
# final.zip から実次元を読むので、この既定値は fallback にすぎない。
OBS_DIM, ACT_DIM = 25, 8

def obs_fields(bundle_dir, obs_dim):
    """[[name, width], ...] describing the OBSERVATION LAYOUT, derived from the bundle's training
    meta.yaml (task / obs_mode / proprio_mode / observe_max_duty) — mirrors UmiusiPoseEnv._get_obs.
    The deploy node (rl_attitude_node) cross-checks this against its own assembly order by NAME and
    WIDTH and refuses to start on a mismatch — the only guard against a silent field reorder, which
    golden vectors cannot catch (they replay pre-built obs, they don't test the assembly).
    Returns None (and warns) if the widths don't add up to the model's obs dim — never write a
    wrong table."""
    meta_p = Path(bundle_dir) / "meta.yaml"
    m = yaml.safe_load(meta_p.read_text()) if meta_p.exists() else {}
    obs_mode = m.get("obs_mode", "imu")
    fields = [["ori_err", 3], ["gyro", 3]]
    if obs_mode == "imu_depth":
        fields.append(["depth_err", 1])
    elif obs_mode == "imu_depth_dvl":
        fields += [["depth_err", 1], ["lin_vel", 3]]
    elif obs_mode == "full":
        fields = [["pos_err", 3], ["ori_err", 3], ["lin_vel", 3], ["gyro", 3]]
    if m.get("task") == "attitude_velocity":
        fields.append(["v_cmd", 3])
    if m.get("proprio_mode", "action") == "full":
        fields += [["servo", 4], ["thrust", 4]]
    fields.append(["prev_action", 8])
    if m.get("observe_max_duty"):
        fields.append(["max_duty", 1])
    if sum(w for _, w in fields) != obs_dim:
        print(f"⚠ obs_fields の合計 {sum(w for _, w in fields)} != obs_dim {obs_dim} — "
              f"meta.yaml が不完全 (task/obs_mode/proprio_mode を確認)。obs_fields は書き出さない")
        return None
    return fields


def action_contract(tm, env, sim_cfg):
    """Deploy-side contract for a wrench-mode policy (action_mode: modes).

    A modes policy does NOT output [servo x4, esc x4]: it outputs 6 wrench-mode RATES, and the
    deploy node must reproduce the same three stages the sim ran, in order, or the robot gets a
    different plant than the one that was trained (the A-11 failure mode). The policy rides the
    slew limiter (measured: 100 % of steps), so none of this is optional.

    The plant constants are READ FROM THE SIM CONFIG this run trained against, never hardcoded:
    thrust_per_cmd is explicitly UNMEASURED (configs/umiusi.yaml — bench calibration pending), so
    a literal here would silently ship a stale contract the first time it is retuned, and the
    deployed plant would diverge from the trained one — the A-11 failure this contract prevents.
    """
    if tm.get("action_mode") != "modes":
        return None
    thr = sim_cfg["thrusters"]
    return {
        "action_mode": "modes",
        "mode_names": list(MODE_NAMES),
        "note": "action = mode RATES in [-1,1] (REP-103 body wrench), NOT servo/esc",
        "stages": [
            {"1_integrate": "m += a * mode_slew_per_s * dt, clipped to [-1, 1]; m persists "
                            "across steps and resets to 0 on disarm"},
            {"2_mix": "per unit: h = Sh @ (m.fx, m.fy, m.tz) * f_max, v = Sv @ (m.fz, m.tx, m.ty) "
                      "* f_max, with f_max = thrust_per_cmd * max_duty**thrust_curve_exp"},
            {"3_fold": "servo = atan2(v, h) folded into +/-90 deg by reversing the esc sign; "
                       "esc = sign * (min(|f|, f_max)/thrust_per_cmd)**(1/thrust_curve_exp); "
                       "a unit inside the deadband holds its previous servo angle and zeroes esc"},
        ],
        "mode_signs": {n: list(s) for n, s in _MODE_SIGNS.items()},
        "mode_sign_columns": ["fx", "fy", "tz", "fz", "tx", "ty"],
        "mode_slew_per_s": float(env.get("mode_slew_per_s", 0.0)),
        "deadband_frac": float(DEADBAND_FRAC),
        "thrust_per_cmd": float(thr["thrust_per_cmd"]),
        "thrust_curve_exp": float(thr.get("thrust_curve_exp", 1.0)),
        "servo_range_deg": float(max(abs(v) for v in thr["servo_range_deg"])),
        "control_rate_hz": float(sim_cfg["sim"]["control_rate_hz"]),
    }


def stub():
    class S(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        def reset(self, *, seed=None, options=None): return np.zeros(OBS_DIM, np.float32), {}
        def step(self, a): return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}
    return S()

def main():
    global OBS_DIM
    OUT.mkdir(exist_ok=True)
    model = PPO.load(str(POL / "final.zip"), device="cpu")
    pol = model.policy
    # 実際の観測次元はモデルが知っている (proprio_mode "action" なら 17)。
    OBS_DIM = int(np.prod(model.observation_space.shape))
    global ACT_DIM
    ACT_DIM = int(np.prod(model.action_space.shape))   # 6 for action_mode "modes", 8 for "esc"

    # --- 重み: 素の tensor だけの state_dict にする ---
    sd = {k: v.detach().cpu().clone() for k, v in pol.state_dict().items()}
    torch.save(sd, OUT / "weights.pt")

    # --- 正規化統計 ---
    dummy = DummyVecEnv([stub])
    vn = VecNormalize.load(str(POL / "vecnormalize.pkl"), dummy); dummy.close()
    np.savez(OUT / "obs_norm.npz",
             mean=np.asarray(vn.obs_rms.mean, dtype=np.float64),
             var=np.asarray(vn.obs_rms.var, dtype=np.float64),
             clip=np.float64(vn.clip_obs), eps=np.float64(vn.epsilon))

    # --- 構造メタ ---
    arch = []
    for k, v in sd.items():
        if k.endswith("weight") and v.ndim == 2:
            arch.append([k, list(v.shape)])
    meta = {"obs_dim": OBS_DIM, "act_dim": ACT_DIM,
            "net_arch": model.policy.net_arch if hasattr(model.policy, "net_arch") else None,
            "layers": arch}
    # 学習側 meta.yaml の契約キーを転記する。特に obs_frame はノードのローダが**必須**で見る
    # (rep103 以外/欠落は起動拒否 — 2026-08-21 の軸取り違えの再発防止ゲート)。
    train_meta_p = POL / "meta.yaml"
    tm = yaml.safe_load(train_meta_p.read_text()) if train_meta_p.exists() else {}
    for k in ("obs_frame", "task", "obs_mode", "proprio_mode", "action_mode"):
        if tm.get(k) is not None:
            meta[k] = tm[k]
    # The env block and the plant constants live in the TRAINING config / the sim config it
    # points at, not the run meta: resolve both so the contract carries the values this run
    # actually trained with (never literals — see action_contract).
    if tm.get("action_mode") == "modes":
        train_cfg = yaml.safe_load((_ROOT / tm.get("config", "configs/train_ppo.yaml")).read_text())
        sim_cfg_p = Path(train_cfg.get("sim_config", "configs/umiusi.yaml"))
        sim_cfg_p = sim_cfg_p if sim_cfg_p.is_absolute() else _ROOT / sim_cfg_p
        contract = action_contract(tm, train_cfg.get("env", {}), yaml.safe_load(sim_cfg_p.read_text()))
        if contract is not None:
            if not contract["mode_slew_per_s"]:
                raise SystemExit("action_mode=modes but mode_slew_per_s is 0 in the training config "
                                 "— the deploy contract would be incomplete")
            meta["action_contract"] = contract
    meta["source"] = POL.name
    fields = obs_fields(POL, OBS_DIM)
    if fields is not None:
        meta["obs_fields"] = fields
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("書き出し先:", OUT)
    for k, v in sd.items():
        print(f"  {k:46s} {tuple(v.shape)}")

    # --- 素 torch 実装で SB3 と一致するか検証 ---
    # policy_infer は tools/ から umiusi_rl_control パッケージへ移動した (どちらでも動くよう両方通す)
    sys.path.insert(0, str(AUTONOMY / 'tools'))
    sys.path.insert(0, str(AUTONOMY / 'umiusi_rl_control' / 'umiusi_rl_control'))
    import policy_infer as pi
    runner = pi.PolicyRunner(OUT)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        o = rng.normal(0, 1, OBS_DIM).astype(np.float32) * rng.uniform(0.1, 3.0)
        a_ref, _ = model.predict(runner.normalize(o), deterministic=True)
        a_new = runner.act(o)
        worst = max(worst, float(np.abs(np.asarray(a_ref) - a_new).max()))
    print(f"\nSB3 との最大差: {worst:.3e}  ->  {'✅ 一致' if worst < 1e-5 else '❌ 不一致'}")

if __name__ == "__main__":
    main()
