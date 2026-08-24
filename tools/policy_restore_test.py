#!/usr/bin/env python3
"""ポリシー bundle の復元性・発散有無を閉ループで検証する。

対象は下の `POL` 定数で指すバンドル(使うときに対象バンドルへ向ける)。

機体を実際に傾けた状態から開始し、目標を水平に固定してポリシーを回す。
姿勢誤差 |ori_err| が減衰すれば「戻す方向」、増大し続ければ「発散」。
"""
from __future__ import annotations
import math, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages/sim/src"))

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config  # noqa: E402
from stable_baselines3 import PPO                                       # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

# POL = 検証する対象バンドル。使用時にこれを対象へ向ける。
POL = (ROOT / "../ros2_ws/src/sinsei_UMIUSI_autonomy/umiusi_rl_control/models/cruise_policy").resolve()

def axis_quat(axis, deg):
    """軸まわり deg 度の回転クォータニオン (w,x,y,z)"""
    a = np.array(axis, dtype=float); a /= np.linalg.norm(a)
    h = math.radians(deg) / 2.0
    return np.concatenate([[math.cos(h)], a * math.sin(h)])

def make_env():
    cfg = load_config(str(ROOT / "configs/train_ppo.yaml"))
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["obs_mode"] = "imu"
    cfg["env"]["yaw_target_deg"] = 180.0
    cfg["env"]["vel_cmd_cone_deg"] = 180.0
    cfg["domain_rand"]["enabled"] = False
    cfg["disturbance"]["enabled"] = False
    return UmiusiPoseEnv(cfg)

def load_policy(env):
    model = PPO.load(str(POL / "final.zip"), device="cpu")
    stats = POL / "vecnormalize.pkl"
    dummy = DummyVecEnv([lambda: make_env()])
    vn = VecNormalize.load(str(stats), dummy); dummy.close()
    rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon
    def norm(o):
        return np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
    return model, norm

def rollout(env, model, norm, axis, deg, steps=400, hold_still=True):
    env.reset(seed=0)
    env.target_quat = np.array([1.0, 0.0, 0.0, 0.0])   # 目標 = 水平・ヨー0
    if hold_still:
        env.v_cmd = np.zeros(3)                        # 前進指令なし = 純粋な姿勢保持
    env.sim.reset(pos=(0.0, 0.0, 0.0), quat=tuple(axis_quat(axis, deg)))  # 機体を傾けて配置
    errs, acts = [], []
    obs = None
    for i in range(steps):
        if obs is None:
            a = np.zeros(env.action_space.shape[0], dtype=np.float32)
        else:
            a, _ = model.predict(norm(obs), deterministic=True)
            a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        obs, r, term, trunc, info = env.step(a)
        obs = np.asarray(obs, dtype=float)
        errs.append(float(np.linalg.norm(obs[:3])))
        acts.append(np.abs(a).max())
        if term or trunc:
            env.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
    return np.array(errs), np.array(acts)

def main():
    env = make_env()
    model, norm = load_policy(env)
    print(f"policy: {POL/'final.zip'}")
    print(f"obs_dim={env.observation_space.shape}  act_dim={env.action_space.shape}\n")
    print(f"{'軸':>6} {'初期傾き':>8} {'初期誤差':>9} {'最小誤差':>9} {'最終誤差':>9} {'最大誤差':>9} {'|a|max':>7}  判定")
    print("-" * 84)
    cases = [((1,0,0), "X(roll)",  (10, 20, 30, 45, 60, 90)),
             ((0,0,1), "Z(pitch)", (10, 20, 30, 45, 60, 90)),
             ((0,1,0), "Y(yaw)",   (15, 30, 45, 90, 135, 179))]
    for axis, name, degs in cases:
        for deg in degs:
            e, a = rollout(env, model, norm, axis, deg, steps=600)
            e0, emin, efin, emax = e[0], e.min(), e[-50:].mean(), e.max()
            if efin < e0 * 0.5 and efin < 0.35:
                verdict = "✅ 復元"
            elif efin < e0:
                verdict = "△ 部分復元"
            elif emax > e0 * 1.5:
                verdict = "❌ 発散"
            else:
                verdict = "△ 停滞"
            print(f"{name:>8} {deg:6d}° {math.degrees(e0):8.1f}° {math.degrees(emin):8.1f}° "
                  f"{math.degrees(efin):8.1f}° {math.degrees(emax):8.1f}° {a.max():7.2f}  {verdict}")
    print("\n(誤差は目標姿勢への回転ベクトルのノルム。最終誤差=末尾50ステップ平均=1秒)")


if __name__ == "__main__":
    main()
