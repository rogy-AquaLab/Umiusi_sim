# umiusi_sim — ランタイムアーキテクチャ (シム ⇄ 実機)

**開発/シミュレーション**と**実機デプロイ**の両方で各コンポーネントがどのように組み合わさるか、そして両者が乖離しないように保つ唯一のルール: **物理は一度だけ記述する (Python)。それ以外はすべて、両方の世界で再利用される単一のライブラリである。**

## 3つの関心事

| 関心事 | 内容 | 所在 |
|---|---|---|
| **物理 / シム** | 浮力 · 抗力 · 揚力 · スラスタ推力 · MuJoCo ステップ | **`umiusi_sim` (Python) — 単一の実装** |
| **知覚 + ナビゲータ** (高レベル) | カメラ → 学習済み検出器 → 検出結果 → 挙動 FSM → 速度/姿勢コマンド | **単一の Python ライブラリ** (`umiusi_perception`、`umiusi_perception.autonomy` を含む)。シム (`tools/autonomy_run`) **と**実機 (`ros2_ws/src/umiusi_autonomy`) で再利用 |
| **低レベル制御** | Gate → Attitude → Thruster (または RL ポリシー — **実機で稼働中**) → スラスタごとのサーボ/ESC | ROS 2 `sinsei_umiusi_control` コントローラ (変更なし) + `rl_attitude_node` |
| **学習** | RL (巡航/姿勢) + 検出器の学習 | **Python、オフライン** → モデルを出力 (RL は**素 torch bundle**、検出器は `.pt`/`.onnx`) |

## 物理の単一ソース

物理実装は**厳密に1つだけ**でなければならない: `packages/sim/src/umiusi_sim/simulator.py` + `physics/` (解析的な流体) + MJCF。2つのコピーは乖離する — 以前の C++ 流体移植版はすでに遅れていた (揚力/CoP なし)。**シムは開発/テストでのみ実行され、実機では決して実行されない** (実機は実際のハードウェアである) ため、C++ 移植版の唯一の正当化理由 — 「Pi パリティ」 — は当てはまらない。したがって:

- **ROS ブリッジは物理を再実装しない。** これは、動作中の Python シムへの*薄いリレー*である (下記「IPC ブリッジ」を参照)。C++ 流体は削除/非推奨とする。
- 忠実度に関する作業 (揚力、CoP モーメント、キャリブレーション) は**1箇所**で行う: Python シム。

## ランタイム構成

```
(1) DEV — 純粋な Python (高速なイテレーション)
    tools/autonomy_run.py :  perception + navigator FSM  ──drive──▶  Python sim (umiusi_sim)
    tools/drive.py / umiusi_rl.eval : RL policy / manual ──────────▶  Python sim

(2) DEV — ROS-in-the-loop (実際の制御スタックをシムに対してテストする)
    sinsei Gate→Attitude→Thruster (ros2_control)
        │ command interfaces (esc duty, servo angle)
        ▼
    umiusi_sim_bridge  ── thin IPC relay (NO physics) ──▶  Python sim server (umiusi_sim)
        ▲ state interfaces (imu quat/gyro/accel, thruster servo/esc)
    (same perception_node + navigator_node as (3) can run on top)

(3) REAL ROBOT — NO sim
    [camera] ─▶ perception_node (loads examples/balloon_detector) ─▶ /detections
    /detections + /state/imu ─▶ navigator_node (umiusi_perception.autonomy FSM) ─▶ /cmd/target (or /cmd/direct)
    /cmd/target ─▶ sinsei Gate→Attitude→Thruster ─▶ CAN plugin ─▶ real thrusters

    低レベル姿勢/ベクタリングの実配備経路 (2026-08-21 のプール試験以降、稼働中):
    /state/imu ─▶ rl_attitude_node (素 torch bundle を policy_infer で推論) ─▶ /cmd/direct/… ─▶ CAN
      ▲ 深度しきい値のモード切替スーパバイザ (水平ポリシー ⇄ 降下ポリシー; 圧力センサ) が
        どの bundle を回すか / 指令をどう整形するかを決める
```

シム ↔ 実機は **ros2_control ハードウェアプラグインの入れ替え**である (IPC-ブリッジ-to-Python ↔ CAN)。コントローラ、launch、パラメータ、perception_node、navigator_node は (2) と (3) で**同一**である。

## IPC ブリッジ (構成 2) — 「接続部分だけ」

ros2_control ハードウェアコンポーネントは C++ でなければならないが、**物理は一切持たない** — 1サイクル分のコマンド/状態を Python シムとの間でマーシャリングするだけである:

- **Python シムサーバ** (`umiusi_sim`): `UmiusiSimulator` をラップし、ローカルの IPC チャネル (Unix domain socket / ZMQ) で待ち受ける。リクエストごとに、8次元のコマンド (スラスタごとの esc duty + サーボ角度 + 許可ビット) を受信し、シムを1制御周期分ステップさせ、状態 (IMU クォータニオン `[w,x,y,z]`、ジャイロ、加速度 = 比力、スラスタごとのサーボ角度 + esc rpm、およびビューア用の qpos) を返す。シム — すなわち浮力/抗力/**揚力/CoP**/推力 — は単一の Python 実装である。
- **C++ リレー** (`umiusi_sim_bridge`): `SystemInterface` は `on_activate` でソケットに接続する。`write()` がコマンドを送信し、`read()` が状態を受信してインタフェースハンドルを埋める (キャッシュ済み、サイクルごとのアロケーションなし)。CAN ハードウェアと同じインタフェース名なので、6つのコントローラは変更なしでスポーンする。
- 100 Hz はローカルソケット上で問題ない (往復時間はサブミリ秒)。シムは開発マシン上で動作する (Python が利用可能) ため、組み込みターゲットの制約は課されない。

## 知覚 + ナビゲータ = 単一のライブラリ、2つのフロントエンド

`umiusi_perception` (検出器)、`umiusi_perception.autonomy` (バルーン FSM)、`umiusi_perception.control` (フィードフォワードのスラスタ配分、純粋な numpy) は、**ROS もシムも依存しないプレーンな Python ライブラリコード**である — 検出結果/状態を受け取り、コマンド/アクションを返す。これらはリポジトリの**実機搭載ホイール** (`packages/perception`) である: `pip install ./packages/perception` はこれら*だけ*を導入する — シミュレータも学習コードも mujoco も含まない。(パッケージングの詳細: これは uv ワークスペース内の2つのホイールのうちの1つである。リポジトリの README と `ai/architecture.md` §2 を参照。)

- **シムフロントエンド**: `tools/autonomy_run.py` が劣化させたシムカメラをこれらに供給し、Python シムを駆動する。
- **実機フロントエンド**: 薄い `rclpy` ノード — `perception_node` (カメラトピック → 検出器 → 検出結果) と `navigator_node` (検出結果 + IMU → FSM → 配分 → `/cmd/direct`) — が**同じ**関数をラップする。

このように、検出器、ナビゲータ、配分は**一度だけ**記述され、シムで検証され、実機に変更なしでデプロイされる。Pi 上での推論は fps のために ONNX/int8 エクスポートを使用する (~12–30 fps @320、ベンチマーク済み)。

## 学習は Python にとどまる

RL (`umiusi_rl`、Python シム上の PPO) と検出器の学習 (`tools/perception_train`) は **Python でオフラインで**実行され、モデルを出力する。実機と ROS ブリッジは、それらのモデルを**ロード**するだけである (`examples/cruise_policy/`、`examples/balloon_detector/`) — 決して学習しない。実機上での学習はない。したがって `umiusi_rl` (sb3/gymnasium) は**開発/学習ホイール**の一部であり、実機には決してインストールされない。**RL ポリシーはすでにハードウェア上で動いている** (2026-08-21 のプール試験、`rl_attitude_node`。競技オートノミーの方は依然フィードフォワードのみ)。ただし実装は当初構想していた `umiusi_perception.policy` + onnxruntime ローダ**ではない** (作られなかった)。実際の経路は **素 torch エクスポート**である: `tools/export_policy.py` が SB3 の重み + VecNormalize 統計を `weights.pt` / `obs_norm.npz` / `meta.json` の bundle へ書き出し、実機側は `policy_infer.py` (torch + numpy のみ) で推論する — sb3/gymnasium/cloudpickle/シムなし。コピー後の同一性は `tools/preflight_policy.py` が生成した `golden.npz` を実機で再生して検証する。

## 帰結 / 決定事項

- 揚力/CoP (またはあらゆる物理) を C++ に**移植しない** — 代わりにブリッジが Python にリレーする。
- 両フロントエンドが再利用できるよう、`umiusi_perception.autonomy` + `umiusi_perception` を ROS/シムの import から解放しておく。
- 実機は3つのもの (perception_node、navigator_node、低レベルコントローラ) を実行する — **シムプロセスはない**。
- 配備するポリシーは **REP-103 (x 前 / y 左 / z 上) の body-frame 観測**を食う (`obs_frame` 契約)。実機側で手書きの軸入れ替えは**しない** — frame 変換はオフラインで `tools/convert_policy_frame.py` が厳密に行い、`golden.npz` の検証がその契約ごと固定する ([`rl.md`](rl.md) 参照)。
