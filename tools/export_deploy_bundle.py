#!/usr/bin/env python3
"""バンドル (export/) を C++ から読める 1 ファイルの TorchScript に固める。

C++ 側 (sinsei_umiusi_control の logic::attitude::Rl) には JSON パーサも npz リーダも
無い。ファイルを 3 つ読ませる代わりに、ネットと正規化パラメータと action_contract を
1 つの TorchScript に入れる。

    uv run python tools/export_deploy_bundle.py models/av_mode13

-> そのバンドルの直下に deploy.pt と golden.pt ができる (models/ は .gitignore 済み)。

元のバンドル (export/weights.pt, obs_norm.npz, meta.json) は変えない。これは
export_policy.py の出力を配備用に詰め直すだけの後段。バンドルを差し替えたら流し直すこと。

**配備の経路に autonomy を挟まない。** 方策を作るのは sim なので、配備物を作るのも sim。
control は umiusi_sim の models/<name>/deploy.pt を直接指す。

入っているもの:
  forward(x)       ネット本体 (正規化前の観測を渡す。正規化はこの中でやらない)
  obs_mean/var     正規化パラメータ (obs_norm.npz と同じ値)
  obs_clip/eps
  obs_dim / act_dim
  obs_field_names / obs_field_widths    観測レイアウト (meta.json の obs_fields。
                   古いバンドルで欠けていれば学習時 meta.yaml から導出する)
  action_mode      "direct" か "modes"
  action_contract の全項目 (レンチモードのとき):
    thrust_per_cmd / thrust_curve_exp / servo_range_deg / control_rate_hz
    mode_slew_per_s / deadband_frac
    mode_names / mode_sign_columns / mode_positions / mode_signs

**meta.json の情報を落とさないこと。** 落とした項目は C++ 側で再構成できず、
「その次元・その action_mode は未対応」として起動を拒否するしかなくなる。
observation の並び (obs_fields) も同じ: 運ばないと並びの取り違えを golden が素通りさせる。

golden.npz はここには入れない。配備前検証は別に流す (検証用のデータを配備物に混ぜない)。
ただし C++ 側は npz を読めないので、golden.npz があれば **別ファイル** golden.pt に
書き出す (obs / act の 2 テンソルだけ)。配備物と検証データは分けたまま。
"""

# `from __future__ import annotations` は使わない。TorchScript は _Deploy のクラス注釈を
# 実物の型として解決するので、文字列化されると List[str] を読めずに落ちる。

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]          # umiusi_sim repo root (config paths)


# mixed 軌跡を作るときの条件。C++ 側 (test_attitude_rl) は golden.pt が運ぶこの値で再生する
MIXED_MAX_DUTY = 0.25
MIXED_DT = 0.02


def _load_net(export: Path):
    """policy_infer と同じ手順でネットを組む。ここで独自に組み直すと静かにずれる。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from policy_infer import PolicyRunner

    return PolicyRunner(str(export))


class _Deploy(torch.nn.Module):
    """属性は __init__ で登録し、torch.jit.script で固めること。
    trace はネットの実行だけを写すので、属性が落ちる。

    空のリストは型注釈が要る。TorchScript は `[]` を List[Tensor] と推論するので、
    contract が無いバンドルで属性の型が変わってしまう (C++ 側の toStringList() が落ちる)。
    """

    obs_field_names: List[str]
    obs_field_widths: List[int]
    mode_names: List[str]
    mode_sign_columns: List[str]
    mode_positions: List[str]

    def __init__(self, net, norm, meta, obs_dim, obs_fields):
        super().__init__()
        self.net = net
        self.register_buffer("obs_mean", torch.tensor(norm["mean"], dtype=torch.float64))
        self.register_buffer("obs_var", torch.tensor(norm["var"], dtype=torch.float64))
        self.obs_clip = float(norm["clip"])
        self.obs_eps = float(norm["eps"])
        self.obs_dim = int(obs_dim)
        self.act_dim = int(meta.get("act_dim", 0))
        self.action_mode = str(meta.get("action_mode", "direct"))
        # 観測の座標系。読む側は rep103 以外を拒否する — 2026-08-21 のプール試験で
        # pitch/yaw が入れ替わった観測を食わせて姿勢制御不能になった再発防止ゲート
        self.obs_frame = str(meta.get("obs_frame", "unknown"))

        # 観測レイアウト。導出もできなければ空 (読む側は「照合できない」として警告する)
        self.obs_field_names = [str(n) for n, _ in obs_fields]
        self.obs_field_widths = [int(w) for _, w in obs_fields]

        # contract が無いバンドル (direct 出力) では 0 / 空。読む側は action_mode で分岐する
        c = meta.get("action_contract") or {}
        self.thrust_per_cmd = float(c.get("thrust_per_cmd", 0.0))
        self.thrust_curve_exp = float(c.get("thrust_curve_exp", 0.0))
        self.servo_range_deg = float(c.get("servo_range_deg", 0.0))
        self.control_rate_hz = float(c.get("control_rate_hz", 0.0))
        self.mode_slew_per_s = float(c.get("mode_slew_per_s", 0.0))
        self.deadband_frac = float(c.get("deadband_frac", 0.0))
        self.mode_names = [str(x) for x in c.get("mode_names", [])]
        self.mode_sign_columns = [str(x) for x in c.get("mode_sign_columns", [])]
        # 符号表は dict なので、行の並びを別に運ぶ (ユニット名 -> 行)。
        # C++ 側は自分の配置順に並べ替えてから使う
        signs = c.get("mode_signs") or {}
        self.mode_positions = [str(p) for p in signs]
        rows = [[float(v) for v in signs[p]] for p in signs]
        self.register_buffer(
            "mode_signs",
            torch.tensor(rows, dtype=torch.float64) if rows
            else torch.zeros((0, 0), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Golden(torch.nn.Module):
    """golden.npz を C++ から読める形にしただけの入れ物 (obs / act)。

    実機の配備物ではない。C++ 側 (verify_golden) が起動時に再生して、重み・正規化統計・
    観測レイアウトが sim と食い違っていないかを見る。

    レンチモードでは `mixed` も入れる: act (6 次元のレート) を Python の ModeAction に
    通した [servo x4, esc x4]。**golden の obs/act だけでは 3 段の変換を検証できない** —
    ネットの生出力しか突き合わせないので、積分・ミキサ・折返しを取り違えても PASS する。
    `has_mixed` が false のときは direct 出力のバンドル。
    """

    def __init__(self, obs, act, mixed=None, max_duty=0.0, dt=0.0):
        super().__init__()
        self.register_buffer("obs", torch.tensor(obs, dtype=torch.float64))
        self.register_buffer("act", torch.tensor(act, dtype=torch.float64))
        self.has_mixed = mixed is not None
        self.mixed_max_duty = float(max_duty)
        self.mixed_dt = float(dt)
        self.register_buffer(
            "mixed",
            torch.tensor(mixed, dtype=torch.float64) if mixed is not None
            else torch.zeros((0, 0), dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        return self.obs


def _derive_obs_fields(bundle: Path, obs_dim: int):
    """meta.json に obs_fields が無い古いバンドル向けに、学習時 meta.yaml から組み直す。

    `tools/export_policy.py` の `obs_fields()` と同じ導出。あちらが正で、
    ここは既に出回っているバンドル (17 / 14 次元) を後追いで埋めるためだけにある。
    **新しいバンドルは sim 側が meta.json に書くので、ここには来ない。**

    導出元は学習時の meta.yaml であってノードの組み立て順ではないので、
    「並びが食い違っていないか」という照合の意味は失われない。
    幅の合計が合わなければ空を返す — 間違った表を書くくらいなら「照合できない」と言う。
    """
    meta_p = bundle / "meta.yaml"
    if not meta_p.exists():
        return []
    import yaml

    m = yaml.safe_load(meta_p.read_text()) or {}
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
        print(f"⚠ meta.yaml から導いた obs_fields の合計が obs_dim {obs_dim} と合わないので"
              "書き出しません (task/obs_mode/proprio_mode を確認)")
        return []
    return fields


def _mixed_trajectory(bundle: Path, meta, act, max_duty, dt):
    """act (N, 6) を **env が回したのと同じ ModeMixer** に通した (N, 8) を返す。

    契約から組み直した別実装ではなく sim の正本を通すので、ModeMixer に入れた変更が
    C++ 側と食い違えば golden 検証で落ちる。幾何は学習時の config が指す sim config
    から取る (action_contract の係数と同じ出どころ)。

    積分は env と同じ 2 段: m += a * mode_slew_per_s * dt を [-1, 1] にクリップし、
    ミキサに渡す (umiusi_pose_env.py の _mode_rate_action 経路)。積分器と前回サーボ角は
    行をまたいで持ち越す — その持ち越しごと固定したいので 1 行ずつ計算し直さないこと。
    """
    from umiusi_rl.envs.mode_mixer import MODE_DIM, ModeMixer
    from umiusi_sim.simulator import UmiusiSimulator

    tm = yaml.safe_load((bundle / "meta.yaml").read_text())
    train_cfg = yaml.safe_load((_ROOT / tm.get("config", "configs/train_ppo.yaml")).read_text())
    sim_cfg_p = Path(train_cfg.get("sim_config", "configs/umiusi.yaml"))
    sim_cfg_p = sim_cfg_p if sim_cfg_p.is_absolute() else _ROOT / sim_cfg_p
    sim = UmiusiSimulator(config_path=sim_cfg_p)
    mixer = ModeMixer(sim.unit_names, sim.thrust_axes, sim.unit_pivots, sim.servo_range_rad,
                      sim.thrust_per_cmd, sim.thrust_curve_exp)

    slew = float(meta["action_contract"]["mode_slew_per_s"])
    m = np.zeros(MODE_DIM)
    prev_servo = np.zeros(len(sim.unit_names))
    out = []
    for a in act:
        m = np.clip(m + np.clip(np.asarray(a, dtype=float), -1.0, 1.0) * slew * dt, -1.0, 1.0)
        y = mixer.mix(m, max_duty, prev_servo)
        prev_servo = y[:4].copy()
        out.append(y)
    return np.stack(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", help="models/<name> (export/ を含むディレクトリ)")
    p.add_argument("-o", "--out", default=None, help="出力先 (既定: <bundle>/deploy.pt)")
    a = p.parse_args(argv)

    bundle = Path(a.bundle)
    export = bundle / "export"
    if not export.is_dir():
        p.error(f"{export} がありません")

    torch.set_num_threads(1)
    runner = _load_net(export)
    meta = json.loads((export / "meta.json").read_text(encoding="utf-8"))
    norm = np.load(export / "obs_norm.npz")

    # レンチモードの係数は 1 つでも欠けたら書き出さない。0.0 で埋めて出すと、読む側は
    # 「欠けている」と「その値が 0」を区別できない (deadband_frac 0 = デッドバンド無効)
    if meta.get("action_mode") == "modes":
        c = meta.get("action_contract") or {}
        missing = [k for k in ("mode_names", "mode_signs", "mode_sign_columns", "mode_slew_per_s",
                               "deadband_frac", "thrust_per_cmd", "thrust_curve_exp",
                               "servo_range_deg") if k not in c]
        if missing:
            p.error(f"{export}/meta.json の action_contract に {missing} がありません "
                    "(sim 側の export が古い可能性)")

    # obs_fields は meta.json が正。無い古いバンドルだけ meta.yaml から後追いで埋める
    fields = meta.get("obs_fields") or _derive_obs_fields(bundle, runner.obs_dim)
    mod = _Deploy(runner.net, norm, meta, runner.obs_dim, fields)
    # trace ではなく script。trace はネットの実行だけを写し、属性を落とす
    m = torch.jit.script(mod)

    out = Path(a.out) if a.out else bundle / "deploy.pt"
    m.save(str(out))
    layout = "/".join(mod.obs_field_names) or "(照合できません)"
    print(f"{out}  obs {mod.obs_dim} -> act {mod.act_dim}  mode={mod.action_mode}  obs_fields={layout}")

    golden = bundle / "golden.npz"
    if golden.exists():
        g = np.load(golden)
        mixed = None
        if mod.action_mode == "modes":
            # 検証に使う max_duty / dt は 18 次元ポリシーの学習分布と制御周期の代表値。
            # C++ 側は同じ値で再生するので、ここを変えたら golden.pt を作り直すこと
            mixed = _mixed_trajectory(bundle, meta, g["act"], MIXED_MAX_DUTY, MIXED_DT)
        gout = out.parent / "golden.pt"
        torch.jit.script(_Golden(g["obs"], g["act"], mixed, MIXED_MAX_DUTY, MIXED_DT)).save(
            str(gout))
        extra = "" if mixed is None else f" + mixed {mixed.shape}"
        print(f"{gout}  {len(g['obs'])} vectors{extra}")
    else:
        print(f"{golden} が無いので golden.pt は作りません (配備前検証をスキップすることになる)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
