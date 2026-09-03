#!/usr/bin/env python3
"""SB3 非依存の PPO(MlpPolicy) 推論。torch と numpy だけで動く(numpy 1.x でも可)。

配備物 (deploy.pt) を作るときと、SB3 との一致を検証するときに使う。
`export_policy.py` / `export_deploy_bundle.py` の両方がここを通ることで、
ネットの組み立て方が 1 箇所に固まる。

C++ 側 (sinsei_umiusi_control の logic::attitude::Rl) はこの手順を移植している。
正規化を float64 で計算してから float32 に落とす順序まで合わせること —
float32 で計算すると golden がわずかにずれる。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class PolicyRunner:
    def __init__(self, export_dir):
        d = Path(export_dir)
        sd = torch.load(d / "weights.pt", map_location="cpu", weights_only=True)
        z = np.load(d / "obs_norm.npz")
        self.mean, self.var = z["mean"], z["var"]
        self.clip, self.eps = float(z["clip"]), float(z["eps"])
        # meta.json: obs_frame (frame 契約の検証用) / task など。無い旧 export は空 dict
        mj = d / "meta.json"
        self.meta = json.loads(mj.read_text()) if mj.exists() else {}

        # mlp_extractor.policy_net.{0,2,...} と action_net を取り出す
        layers, i = [], 0
        while f"mlp_extractor.policy_net.{i}.weight" in sd:
            w = sd[f"mlp_extractor.policy_net.{i}.weight"]
            b = sd[f"mlp_extractor.policy_net.{i}.bias"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data, lin.bias.data = w, b
            layers += [lin, nn.Tanh()]
            i += 2
        w, b = sd["action_net.weight"], sd["action_net.bias"]
        act = nn.Linear(w.shape[1], w.shape[0])
        act.weight.data, act.bias.data = w, b
        layers.append(act)
        self.net = nn.Sequential(*layers).eval()
        self.obs_dim = self.mean.shape[0]

    def normalize(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        return np.clip((o - self.mean) / np.sqrt(self.var + self.eps),
                       -self.clip, self.clip).astype(np.float32)

    @torch.no_grad()
    def act(self, obs, already_normalized=False):
        o = np.asarray(obs, dtype=np.float32) if already_normalized else self.normalize(obs)
        a = self.net(torch.from_numpy(o).float().unsqueeze(0)).squeeze(0).numpy()
        return np.clip(a, -1.0, 1.0)
