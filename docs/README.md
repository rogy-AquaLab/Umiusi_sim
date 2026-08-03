# umiusi_sim — docs

UMIUSI 水中ロボットシミュレータ + 自律スタックの設計・実装ノート。まず
[`architecture.md`](architecture.md) を読み、各要素がシムと実機をまたいでどう組み合わさるかを把握すること。

| doc | subsystem | covers |
|---|---|---|
| [architecture.md](architecture.md) | システム | シム ⇄ 実機の構成、物理の単一情報源ルール、ROS ブリッジ、1ライブラリ2フロントエンド |
| [physics.md](physics.md) | 物理 / 流体 | 浮力 · 抗力 · 揚力 · CoP モーメント · 付加質量 · スラスタ推力 · サーボスルー |
| [perception.md](perception.md) | perception (知覚) | 検出器（HSV / Hough / 学習済み `TinyBalloonNet`）、水中復元、トラッカ、学習、ベンチマーク |
| [rl.md](rl.md) | 本体挙動 RL (本体挙動) | `UmiusiPoseEnv` 上の PPO、観測/行動/報酬、タスク、sim2real DR、モデルファミリ、評価 |
| [autonomy.md](autonomy.md) | 上位制御 (上位行動制御) | 風船割り FSM `BalloonBehavior`、ターゲット選択、競技シナリオ、フィードフォワード配分、ROS デプロイ |

## 3層の概観

```
camera ─▶ perception  (学習済み TinyBalloonNet → Detection: 色 + 方位 + 距離)          → perception.md
              │  tracker (関連付け → 確定 → 保持)
              ▼
        autonomy FSM  (最近傍・到達可能・非青; SEARCH→APPROACH→ALIGN→RAM→CONFIRM)      → autonomy.md
              │  {surge, heave, yaw}  ──feedforward_allocation──▶  8次元スラスタ行動
              ▼
        low-level     ── 競技 = フィードフォワード (control.py) ── または ── RL ポリシー ─  → rl.md
                                                                                  (姿勢 / 巡航)
```

**Perception + autonomy** は 1 つの ROS-free ホイール（`umiusi_perception`）であり、シム
（`tools/autonomy_run.py`）と実機（`ros2_ws/src/umiusi_autonomy`）でビット単位で同一である。**RL** は独立した
低レベルトラックであり、現時点では風船ミッションには組み込まれていない（競技の自律制御はフィードフォワードのみ）。
