# パラメータ同定計画 — 次回水中実験でのデータ収集と較正手順

`configs/umiusi.yaml` の流体・アクチュエータ係数は現在 **CAD + BlueROV2 文献からの推定値**
（`python -m tools.estimate_hydro` で再現可能。誤差帯: 抗力 ±30 %、付加質量 並進 ±30 % /
回転 ×2、サーボレート 100–500 deg/s）。この文書は、**次回の実機テストで取るべきデータと、
それを係数にどう落とすか**をまとめたもの。issue #3 の 2b を実行計画にしたもの。

実験はすべて `record_run.sh --bag-only` の bag だけで完結する（追加の計測器は
スラスタ推力測定の秤のみ）。**順番も重要**: 1 が合っていないと 2 以降の符号が狂う。

---

## 0. 前提: bag に必要なトピック

`/state/imu`（姿勢・角速度）、`/cmd/direct/thruster_controller/output_*`（指令）、
`/state/thruster_state_all`（テレメトリ。angle はエコーと確定済みだが記録は継続）、
可能なら圧力（深度）。**各実験の開始/終了に静止区間を 5 秒**入れると解析が楽。

## 1. スラスタ ID ↔ 物理位置の確認（最優先・ドライで可）

**手順**: 1 基ずつ `duty=0.2` を 2 秒、順に 4 基。どの物理位置（前左/後左/前右/後右）が
回ったかを目視記録。サーボも 1 基ずつ +45° を指令し、動いた個体を記録。

**反映**: `configs/umiusi.yaml` の `units[].name` を実機の対応に合わせる（action
順序 `action_order: [lf, lb, rb, rf]` は autonomy の POSITIONS 契約なので触らない）。
sim 側の幾何命名は id1=lf, id2=lb, **id3=rf, id4=rb**（+Z=右舷）である点に注意 —
旧 "3=rb, 4=rf" は右舷の前後が入れ替わっていた。

## 2. サーボ応答（ドライで可）

**手順**: 大角度ステップ（0→80°）と小角度ステップ（0→10°）を各 3 回、動画（スマホ 60fps
スローモーションで可）または到達時間を計測。

**反映**:
- 到達時間が角度に比例 → レート制限支配。`servo_slew_deg_per_s` = 角度/時間。
- 小角度で指数的に収束 → `servo_tau_s` を時定数から。
- 現行モデルは `rate = clip(err/tau, ±slew)`（`physics/thruster.py::track`）。両方の実験から
  2 パラメータがそのまま決まる。

## 3. スラスタ推力マップ（秤、ドライ〜バケツ）

**手順**: スラスタを秤に固定し（水没させて）、duty = ±0.2, ±0.4, ±0.6, ±0.8, ±1.0 で推力 [N]
を記録。1 基で良い（`thrust_unit_frac` の DR が個体差を吸収）。

**反映**: `thrust_per_cmd`（現在 30 N は出典不明）。線形でなければ
BlueROV2 同様の多項式マップを `thruster.py` に足す（正負非対称も分かる）。

## 4. 浮力と復元（水中・無指令）

**手順**: 水中で全脱力し 20 秒静置 ×3。浮上/沈降の終端速度と、静定時の傾き（IMU）を記録。

**反映**:
- 終端鉛直速度 v_t → `displaced_volume`: net force = ρg(V−V_neutral) = D_l·v_t + D_q·v_t²。
- 静定傾きが 0 でなければ CoB の水平オフセット（現在は CoM 直上と仮定）。
- 傾けて手を離す実験（±20° ロール/ピッチ）の復元の速さ → `buoyancy_offset_above_com`
  （復元モーメント = ρgV·offset·sinθ。振動周期と減衰から offset と回転抗力が同時に出る）。

## 5. 並進抗力（水中・定常前進）

**手順**: 一定 duty（0.3 / 0.5 / 0.7）で 10 秒ずつ直進。距離をプールサイドからメジャー+
動画、または既知長を横切る時間で終端速度を測る。

**反映**: 終端で推力 = 抗力なので F(duty) [3 で較正済み] = D_l·v + D_q·v² を
2 点以上でフィットして surge の `drag.linear[0]` / `drag.quadratic[0]`。
推定では終端 0.80 m/s @ full（BlueROV2 実測 0.72 m/s @ 85 N が規模感）。

## 6. 回転抗力（水中・定常旋回）

**手順**: 左右スラスタ逆転（yaw トルク一定）で 10 秒旋回 ×2 トルクレベル。IMU の
ヨーレートが定常に達した値を使う。

**反映**: 終端 ω で トルク = D_l·ω + D_q·ω²。トルクは推力マップと幾何（腕 ~0.19 m）から計算。
yaw の `drag.linear[4]` / `drag.quadratic[4]`（軸順注意: index 4 = +Y = yaw）。

## 7. 付加質量（水中・ステップ応答）

**手順**: 静止からステップ状に duty 0.5 を入れ、立ち上がりの IMU 加速度/速度プロファイルを
記録 ×3。（4 の傾き解放実験の振動周期も回転付加慣性の情報を持つ。）

**反映**: 時定数 τ = (m + m_a)/D_l|_effective から m_a。
sim 推定は surge 2.7 kg / heave 7.1 kg / sway 3.2 kg。倍半分ズレていないかの確認が主目的。

## 8. 検証ループ

較正した yaml で `python -m tools.validate_sim -v` を通し、`python -m tools.analyze_steady`
系で bag 対 sim の定常値を比較 → 残差が DR の幅に収まったら再学習（DR を実測誤差幅まで絞る）。

## 取得データの二次利用（world model / 流体シミュレータ向上）

上記の bag はそのまま「(state, action) → next state」の教師データになる。
- 定常実験（5, 6）は抗力の、過渡実験（7）は付加質量の教師信号。
- 収集時は**多様な励起**が価値になる: 較正実験の合間に 1–2 分、ランダムな
  手動操縦（teleop）区間を入れておくと、後で学習ベースの残差力モデル
  （解析モデルとの差分を NN が埋める形）を試すときの分布が確保できる。

---

## 2026-08-21 実施済み: servo-debug bag からのフィット結果

`/srv/share/bag_data/servo-debug`(74.9 s、水中 RL 稼働、duty ±0.2)に対して
`tools/bag_replay.py` の開ループ再生(0.5 s 窓のジャイロ予測 RMSE)でフィットした:

| パラメータ | 旧値 | フィット | 反映値 |
|---|---|---|---|
| `buoyancy_offset_above_com` | 0.05 m | **0.005–0.01 m**(1/5〜1/10) | 0.010 |
| 推力マップ | 線形 30 N | **プロペラ則(exp 2–3)**。duty 0.2 で実効 ~0.6–1.2 N | `thrust_curve_exp: 2.0` |
| 回転抗力 | — | 等倍で整合(×10 は復元過大の補償だった) | 変更なし |

RMSE: 線形+旧値 0.28–0.44 → 較正後 **0.095**(persistence 下限 0.047)。

**注意**: bag は |duty| ≤ 0.2 しかカバーしないので exp と最大推力は分離できない(§3 の秤で確定)。
テザーは未モデルで、復元/減衰の一部を肩代わりしている可能性がある(§4 の静置実験で分離)。

### 追加発見: IMU frame ミスマッチ(autonomy 側の修正が必要)

bag の解析から **IMU の world frame は z-up**(yaw ハンチング 57° が world-z 周り、body z が鉛直に
最も安定)。sim/学習 frame は +Y-up なので、`rl_attitude_node` が IMU 生値を無変換で観測に入れる限り
**pitch と yaw の成分が入れ替わって方策に入る**。エコー・配線と独立した第 3 の根本原因。
修正は autonomy 側: quat/gyro を CAD frame へ remap(`cad(x,y,z) = imu(x, -z, y)`、
`tools/bag_replay.py::FRAME_P` 参照。x 軸の対応は実機で要確認)。

### bag → npz エクスポータ(ROS python で 1 回実行)

```python
# source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri="<bagdir>", storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in reader.get_all_topics_and_types()}
POS = ("lf", "lb", "rb", "rf")
out = {k: [] for k in
       ["imu_t","imu_quat","imu_gyro","thr_t","thr_angle","thr_rpm","sp_t","sp_quat","sp_vel"]}
for p in POS:
    out[f"cmd_{p}_t"] = []; out[f"cmd_{p}_angle"] = []; out[f"cmd_{p}_duty"] = []
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == "/state/imu":
        m = deserialize_message(data, get_message(types[topic]))
        q, g = m.orientation, m.angular_velocity
        out["imu_t"].append(t*1e-9); out["imu_quat"].append([q.w,q.x,q.y,q.z])
        out["imu_gyro"].append([g.x,g.y,g.z])
    elif topic == "/state/thruster_state_all":
        m = deserialize_message(data, get_message(types[topic]))
        out["thr_t"].append(t*1e-9)
        out["thr_angle"].append([getattr(m,p).angle for p in POS])
        out["thr_rpm"].append([getattr(m,p).rpm for p in POS])
    elif topic == "/rl_attitude_node/current_setpoint":
        m = deserialize_message(data, get_message(types[topic]))
        q = m.orientation; v = m.velocity
        out["sp_t"].append(t*1e-9); out["sp_quat"].append([q.w,q.x,q.y,q.z])
        out["sp_vel"].append([v.x,v.y,v.z])
    elif topic.startswith("/cmd/direct/thruster_controller/output_"):
        p = topic.rsplit("_",1)[1]
        m = deserialize_message(data, get_message(types[topic]))
        out[f"cmd_{p}_t"].append(t*1e-9)
        out[f"cmd_{p}_angle"].append(m.angle); out[f"cmd_{p}_duty"].append(m.duty_cycle)
np.savez("out/<name>.npz", **{k: np.array(v) for k, v in out.items()})
```

### §2 追補: IMU 軸の確定

**コード調査済み (2026-08-21)**: core の FeedForward 行列・set_attitude・UI はすべて REP-103
(x前/y左/z上, roll=x/pitch=y/yaw=z) で一貫。IMU ドライバは AXIS_MAP_CONFIG/SIGN を書かず
チップ生軸のまま publish しているため、**未確定なのは基板の物理取り付け向きのみ**
(bag から z=鉛直は確認済み。面内回転と z の符号が残り)。取り付けが揃っていなければ
ドライバに AXIS_MAP_CONFIG/SIGN を 1 回書いてチップ内 remap するのが最小修正。

確認方法(ドライで 1 分):

機体を手で持ち、(a) 機首を上下(pitch)、(b) その場で水平回転(yaw)、(c) 左右に傾ける(roll)。
それぞれで `/state/imu` の gyro のどの成分が反応するか・符号を記録する。REP-103(x前/y左/z上)なら
pitch=y、yaw=z、roll=x。ずれていれば **IMU ドライバ側で REP-103 に標準化**する(ポリシー側は
rep103 契約のまま触らない)。
