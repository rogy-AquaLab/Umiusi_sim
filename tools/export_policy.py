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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces

# 実機側で動く推論実装は sinsei_UMIUSI_autonomy/tools/policy_infer.py に一本化してある
# (重複を避けるため、ここでは検証時にそこから import する)。
AUTONOMY = Path("../ros2_ws/src/sinsei_UMIUSI_autonomy").resolve()
# POL = 書き出す対象バンドル。使用時にこれを対象へ向ける(既定は最初に書き出した cruise_policy のまま)。
POL = AUTONOMY / "umiusi_rl_control/models/cruise_policy"
OUT = POL / "export"
# 25 = 旧 proprio_mode "full" (imu 6 + v_cmd 3 + servo 4 + thrust 4 + prev_action 8)。
# proprio_mode "action" で学習したポリシーは 17 (imu 6 + v_cmd 3 + prev_action 8)。
# final.zip から実次元を読むので、この既定値は fallback にすぎない。
OBS_DIM, ACT_DIM = 25, 8

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
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("書き出し先:", OUT)
    for k, v in sd.items():
        print(f"  {k:46s} {tuple(v.shape)}")

    # --- 素 torch 実装で SB3 と一致するか検証 ---
    sys.path.insert(0, str(AUTONOMY / 'tools'))
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
