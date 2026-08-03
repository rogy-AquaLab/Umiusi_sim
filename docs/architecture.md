# umiusi_sim — ランタイムアーキテクチャ (シム ⇄ 実機)

**開発/シミュレーション**と**実機デプロイ**の両方で各コンポーネントがどのように組み合わさるか、そして両者が乖離しないように保つ唯一のルール: **物理は一度だけ記述する (Python)。それ以外はすべて、両方の世界で再利用される単一のライブラリである。**

## 3つの関心事

| 関心事 | 内容 | 所在 |
|---|---|---|
| **物理 / シム** | 浮力 · 抗力 · 揚力 · スラスタ推力 · MuJoCo ステップ | **`umiusi_sim` (Python) — 単一の実装** |
| **知覚 + ナビゲータ** (高レベル) | カメラ → 学習済み検出器 → 検出結果 → 挙動 FSM → 速度/姿勢コマンド | **単一の Python ライブラリ** (`umiusi_perception`、`umiusi_perception.autonomy` を含む)。シム (`tools/autonomy_run`) **と**実機 (`ros2_ws/src/umiusi_autonomy`) で再利用 |
| **低レベル制御** | Gate → Attitude → Thruster (または RL ポリシー) → スラスタごとのサーボ/ESC | ROS 2 `sinsei_umiusi_control` コントローラ (変更なし) |
| **学習** | RL (巡航/姿勢) + 検出器の学習 | **Python、オフライン** → モデルを出力 (`.zip`、`.pt`/`.onnx`) |

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
    (optional: RL policy node replaces AttitudeController via /cmd/direct/…)
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

RL (`umiusi_rl`、Python シム上の PPO) と検出器の学習 (`tools/perception_train`) は **Python でオフラインで**実行され、モデルを出力する。実機と ROS ブリッジは、それらのモデルを**ロード**するだけである (`examples/cruise_policy/`、`examples/balloon_detector/`) — 決して学習しない。実機上での学習はない。したがって `umiusi_rl` (sb3/gymnasium) は**開発/学習ホイール**の一部であり、実機には決してインストールされない。RL ポリシーがハードウェア上で動作する場合 (将来 — 現在の競技オートノミーはフィードフォワードのみ)、それは検出器のパターンに従う: `umiusi_rl` で学習 + **エクスポート** (onnx/torchscript) を行い、その後、薄い `umiusi_perception.policy` ローダ (onnxruntime、予約済み) が実機上でそれを実行する — sb3/gymnasium/シムなし。

## 帰結 / 決定事項

- 揚力/CoP (またはあらゆる物理) を C++ に**移植しない** — 代わりにブリッジが Python にリレーする。
- 両フロントエンドが再利用できるよう、`umiusi_perception.autonomy` + `umiusi_perception` を ROS/シムの import から解放しておく。
- 実機は3つのもの (perception_node、navigator_node、低レベルコントローラ) を実行する — **シムプロセスはない**。
