# 10/03 プール実験の閉ループ再現 — 実機の control をそのまま sim に繋いで bag と比べる（2026-10-07）

## 結論（先出し）

- 実機の制御スタック（`sinsei_umiusi_control`、C++/ros2_control、無改変）を `umiusi_sim_bridge` 経由で sim に繋ぎ、bag の入力（`/cmd/attitude_target` / `/cmd/target` / `/cmd/thruster_runnable_all`）を記録時刻どおりに流す道具 `tools/closed_loop_replay.py` を作った。制御周期は /clock のロックステップで厳密 10.00 ms（CPU 負荷に依存しない、実時間比 ~1.9 倍）。
- **チューニング後（develop `da05604`、bag の parameter_events と同じ値）× 午後 bag**: 推力ベクトル（基ごとの水平/垂直成分）とヨーは実機とよく合う。AUTO run（hold_yaw の fb、145 s）: 垂直 corr 0.81〜0.91、水平 0.85〜0.90、gyro z corr 0.99、ヨーレート追従（指令 vs 実測/sim）0.94/0.95、tilt 差 RMSE 3.9°、|duty| 平均 0.123/0.123。MANUAL（13:45）: 垂直 0.83〜0.89、水平 0.79〜0.80、gyro z 0.96。
- **合わないのは roll/pitch**（gyro x: RMSE 0.13〜0.25 rad/s・corr ≤ 0.2、tilt 差 4〜8°）。原因は sim のプラント側（§4）: (1) 垂直推力の交互パターン（duty ±0.5）に対する roll/pitch 応答が sim は実機の数倍大きい（推力を 1/4 にすると tilt は合うがヨーが追従しなくなるので一様なスケール誤差ではない）、(2) ヨー旋回・横移動で sim は roll を −24° まで巻き込む（実機 +8°、符号も逆）、(3) 実機は常時 roll −2〜−4° のトリムを持ち duty に左右差が出るが sim は対称、(4) disarm 中の自由減衰で sim は roll ±5° を周期 ~4 s で振れ続ける。
- **前/後の違いは sim でも bag と同じ向きに出る**: 「前」`b1fb99b`（kp 1.0 / kd 0.0）を 12:13 bag に掛けると、垂直推力の指令で ±90° に届かずサーボが ±14° で往復（cecef01 が直した不具合そのもの）、roll/pitch は ±40〜55° まで振れる（実機 ±6°、duty は実機も sim も ±0.5 の bang-bang）。同じ bag に「後」を掛けると ±89°・duty ±0.5 に乗り、垂直 corr 0.94〜0.96（12:13 の実機は 12:02 の `cecef01` を含む枝で走っていたと整合 — [assumed]）。逆に 13:45 bag に「前」を掛けると |duty| 平均が 0.067（実機 0.145、「後」0.138）と半分以下で、vz の √2 倍と脱飽和が duty 量に効いていることが見える。
- **道中で見つけた bridge / sim_server の不具合（閉ループが成り立たない要因。この道具では自前で回避、既存ファイルは未変更）**:
  - `sim_server.py` の gyro は `mj_objectVelocity(mjOBJ_BODY, flg_local=1)` の値で、これは **body の慣性主軸フレーム**（`ximat`）。この船体では body 軸の並べ替えになり、全基水平推力でヨーしているのに「x 軸まわり 1.3 rad/s」が返る（実測）。fb の kd 項・ヨーレート帰還が別の軸に掛かる。`mjOBJ_XBODY` か `Rᵀ·(global rate)` にすれば正しい（`bag_replay.py` は後者で正しい）。
  - `thrusterN/servo/angle` の command interface は **rad**（mixer が ±π/2 でクランプ、`VescModel::make_servo_angle_frame(rad)`）だが、bridge と `sim_server.py` は deg と解釈して `np.radians()` を掛ける → sim のサーボが ±1.5° しか動かない。
  - 姿勢の frame: control は BNO055 の REP-103（x 前 / y 左 / z 上、ワールドも z 上）前提。bridge/sim_server は CAD（+Y 上 / +Z 右舷）の quaternion をそのまま渡す → body-up が y 軸になり fb が成り立たない。この道具は `R_imu = Pᵀ R_cad P`（P = `bag_replay.FRAME_P`）で変換する。
  - sim プラントの `max_duty` 既定 0.25 が bag の duty（±0.5）を途中で切る。閉ループでは control の `max_duty` 0.5 だけが効くべきなので、プラント側は 1.0 に外した（`--plant-max-duty`）。

## 1. 道具と起動手順

### 構成

```
bag (mcap) ──/cmd/* を記録時刻どおり publish──▶ gate ─▶ attitude ─▶ thruster×4 (実機と同じ controllers.yaml)
                                                              │ ros2_control command interfaces
closed_loop_replay.py ◀── Unix socket (umiusi_sim_bridge/MujocoSystem、無改変) ──┘
   ├ UmiusiSimulator を 1 制御周期ずつ step（サーボ rad / frame 変換 / duty cap 解除 / 正しい body gyro）
   ├ /clock を「要求を 1 つ捌くごとに +10 ms」publish（controller_manager は use_sim_time）
   ├ --segment 秒ごとにプラントの姿勢・角速度を bag の IMU に合わせ直す（控制器の内部状態は持ち越す）
   └ 各周期の duty / サーボ角 / quat / gyro を npz に記録 → compare で bag の npz と比較
```

### 起動（umiusi_sim で、ROS と ros2_ws を source してから uv で実行）

```bash
cd /home/satoimo/mujoco_ws/umiusi_sim
source /opt/ros/jazzy/setup.bash
source /home/satoimo/mujoco_ws/ros2_ws/install/setup.bash          # bridge + msgs + control(develop da05604)
# 「前」の control を使うときはさらに overlay（build 手順は §3）
# source /home/satoimo/mujoco_ws/ros2_ws_b1fb99b/install/setup.bash

D=/home/satoimo/mujoco_ws/data/20261003-pool
nice -n 10 uv run python -m tools.closed_loop_replay run \
  --bag $D/rosbag2_2026_10_03-13_45_43 --npz $D/npz/rosbag2_2026_10_03-13_45_43.npz \
  --start 36 --end 94 --segment 15 --out out/closed_loop/after_manual1345.npz
#   --param attitude_controller.feedback.kp_roll=1.0 ...   # controllers.yaml の上書き（node.dotted=value）
#   --servo-unit rad|deg  --plant-max-duty 1.0  --thrust-scale 1.0  --domain 77(ROS_DOMAIN_ID)
uv run python -m tools.closed_loop_replay compare --sim out/closed_loop/after_manual1345.npz --npz $D/npz/...npz
```

- 1 本ずつ回す（学習ジョブと同居、`nice -n 10`）。実行中は `ros2_control_node` と spawner を子プロセスで起こし、終了時に SIGINT で落とす（`timeout` で切っても後始末する）。`pkill` は使わない。
- `--start` は bag 先頭からの秒。control が起動する前（bag に controllers の parameter_events が出るまで）は意味が無いので、その後から始める。開始時点より前の最新の `/cmd/*` を 1 回ずつ流してから（ZOH）、arm の死に時間（サーボ推定の確定、最大 ~1 s）が 1 回だけ入る。
- 出力 npz: `t, dt, servo_rad[4], duty[4], allowed[4], quat[4](IMU frame), gyro[3](IMU frame), cmd_t, cmd_yaw_rate, ...`。`out/closed_loop/*.npz` に今回の全 run を置いた。
- 制御周期の実測: ロックステップなので request が伝える period は 10.00 ms（min = max）。Pi 実機は 1.6〜5.4 % の周期超過があった（HANDOFF）。1 本だけ 20.00 ms になった run があり（§5）、物理は controller の時間に追従するので比較自体は成り立つ。
- 古い bag（11:23、`hold_yaw` 追加前の `AttitudeTarget`）は 1 byte 詰めて現行型に読み替える（`read_bag`）。

### compare の指標（bag の npz の 50 Hz グリッド上、sim は ZOH）

- `duty_*`: 生の duty の RMSE/corr。**±90° 付近では (角度, duty) と (−角度, −duty) が同じ推力**（mixer の折り返し表現）なので、生の duty/角度は負相関になりやすい。物理的に意味があるのは次の 2 つ。
- `vert_*` = sin(角度)·duty（heave/roll/pitch 成分）、`horiz_*` = cos(角度)·duty（surge/sway/yaw 成分）の RMSE/corr。
- `|duty|`: 全基まとめた |duty| の RMSE/corr。`sat`: |duty| > 0.49 の割合（bag/sim）。
- `gyro_x/y/z`: IMU frame の角速度 RMSE/corr。`yaw cmd~bag` / `yaw cmd~sim`: 指令ヨーレート（`attitude_target.yaw_rate`、hold_yaw の補正は含まない）と実測/sim の RMSE/corr。`tilt sim-bag`: up ベクトル同士の角度差 RMSE [deg] と tilt 量の corr。

## 2. 比較表（ALL = 区間全体、15 s ごとにプラント再初期化）

RMSE/corr。duty は正規化 [-1,1]、gyro は rad/s、tilt は deg。各 run の区間別の行は `out/closed_loop/*.log` 相当（`compare` を再実行すれば出る）。

| control × bag | 区間 | vert lf/lb/rb/rf (RMSE/corr) | horiz lf/lb/rb/rf | \|duty\| | gyro x | gyro y | gyro z | yaw cmd~bag / cmd~sim | tilt 差 | mean\|duty\| bag/sim |
|---|---|---|---|---|---|---|---|---|---|---|
| 後 × MANUAL 13:45 | 36–57 s | 0.08/+0.83 · 0.07/+0.89 · 0.08/+0.87 · 0.07/+0.89 | 0.05/+0.80 · 0.05/+0.80 · 0.05/+0.79 · 0.05/+0.79 | 0.05/+0.92 | 0.25/−0.41 | 0.09/+0.04 | 0.07/+0.96 | 0.28/+0.69 / 0.28/+0.70 | 8.1 | 0.145/0.138 |
| 後 × AUTO 14:45 | 45–190 s | 0.04/+0.91 · 0.05/+0.88 · 0.05/+0.90 · 0.07/+0.81 | 0.05/+0.85 · 0.04/+0.90 · 0.04/+0.90 · 0.05/+0.87 | 0.05/+0.72 | 0.14/+0.21 | 0.03/+0.07 | 0.06/+0.99 | 0.19/+0.94 / 0.18/+0.95 | 3.9 | 0.123/0.123 |
| 前 × AM 12:13 | 0–64 s | 0.19/+0.07 · 0.22/+0.17 · 0.22/+0.07 · 0.19/+0.16 | 0.07/+0.95 · 0.08/+0.94 · 0.09/+0.93 · 0.10/+0.91 | 0.17/+0.66 | 0.76/+0.05 | 0.17/+0.07 | 0.13/+0.83 | 0.23/+0.56 / 0.23/+0.59 | 19.6 | 0.238/0.167 |
| 前 × AM 11:23 | 34–190 s | 0.09/+0.19 · 0.10/+0.45 · 0.11/+0.40 · 0.11/+0.21 | 0.08/+0.91 · 0.06/+0.95 · 0.07/+0.94 · 0.06/+0.94 | 0.08/+0.88 | 0.67/+0.01 | 0.09/−0.03 | 0.07/+0.92 | 0.14/+0.73 / 0.15/+0.70 | 15.1 | 0.137/0.113 |
| 前 × MANUAL 13:45（交差） | 36–57 s | 0.17/+0.18 · 0.14/+0.54 · 0.18/+0.06 · 0.14/+0.52 | 0.06/+0.79 · 0.08/+0.70 · 0.05/+0.80 · 0.08/+0.68 | 0.14/+0.43 | 0.24/−0.11 | 0.05/+0.48 | 0.10/+0.94 | 0.28/+0.69 / 0.22/+0.82 | 7.2 | 0.145/0.067 |
| 後 × AM 12:13（交差） | 0–64 s | 0.13/+0.79 · 0.18/+0.63 · 0.13/+0.79 · 0.14/+0.73 | 0.16/+0.72 · 0.16/+0.72 · 0.16/+0.71 · 0.17/+0.70 | 0.16/+0.65 | 0.42/+0.01 | 0.11/+0.09 | 0.09/+0.89 | 0.23/+0.56 / 0.24/+0.56 | 8.6 | 0.238/0.190 |
| 前 × AM 12:13、推力 ×0.5 | 0–64 s | 0.19/+0.10 · 0.22/+0.12 · 0.21/+0.11 · 0.19/+0.09 | — | 0.18/+0.64 | 0.38/+0.06 | 0.06/+0.06 | 0.11/+0.83 | 0.23/+0.56 / 0.25/+0.49 | 10.1 | 0.238/0.170 |
| 前 × AM 12:13、推力 ×0.25 | 0–64 s | 0.18/+0.14 · 0.21/+0.18 · 0.21/+0.12 · 0.18/+0.21 | — | 0.18/+0.63 | 0.22/−0.05 | 0.05/+0.16 | 0.14/+0.72 | 0.23/+0.56 / 0.27/+0.35 | 7.5 | 0.238/0.175 |
| 後 × MANUAL 13:45、推力 ×0.5 | 36–57 s | 0.10/+0.78 · 0.08/+0.86 · 0.09/+0.81 · 0.08/+0.85 | — | 0.06/+0.88 | 0.23/−0.31 | 0.07/−0.03 | 0.08/+0.96 | 0.28/+0.69 / 0.29/+0.67 | 7.5 | 0.148/0.153 |

読み方:
- 「後 × 午後」は推力ベクトルもヨーも合う。MANUAL の 36–57 s は実機の制御が動いていた区間（57 s 以降は state topic が止まっている）。
- 「前 × 12:13」の垂直成分 corr ≈ 0.1 は、sim の「前」mixer が ±90° に届かない不具合（下記）を再現しているため。水平成分（corr 0.91〜0.95）とヨー（0.83）は合う。
- 「前 × 11:23」も同じ形: 水平 0.91〜0.95・ヨー 0.92 は合い、垂直 0.2〜0.45・roll は合わない。64–74 s（duty ±0.5 の交互パターン）で実機 tilt ±5°、sim −40°。84–99 s で実機は ±90° に duty −0.04 で居るが、sim の「前」mixer は ±28/55° に留まる（retarget 閾値 0.10 未満で追従しない）。
- gyro x（roll）は全条件で合わない。sim のプラントの roll/pitch 応答の問題（§4）。
- `tilt 差` の 8° 前後は、yaw 旋回・sway 中に sim だけ roll/pitch が大きく崩れる区間が効いている。AUTO run（ほぼ直進 + ヨー）は 3.8°。

### 時系列で見た「前/後」の違い（12:13 bag、同じ入力）

| bag 時刻 | 実機（12:13） | sim 前 `b1fb99b` kp1/kd0 | sim 後 `da05604` |
|---|---|---|---|
| 8–16 s 垂直推力（vz 指令） | 角度 ±90°、duty ±0.37〜0.5 | 角度 ±14° で往復、duty ±0.08（垂直へ向かえない） | 角度 ±89°、duty ±0.5 |
| 20 s / 34 s ヨー ∓1 rad/s | gz −0.81 / +0.82 | −0.88 / +1.02 | −0.74 / +0.93 |
| 36–44 s sway 全開 + 姿勢 | duty ±0.5 の bang-bang、tilt ±6° 以内 | 同じ bang-bang、tilt ±40〜55° | tilt ±4〜18°、duty 0.2〜0.3 |

13:45 bag に「前」を掛けると（交差）、heave 区間 41–46 s で実機/「後」は duty ±0.30 なのに「前」は ±0.21（vz 列が 1 倍）。|duty| 平均 0.067 vs 実機 0.145。

## 3. 「前」の control の build と、どのバイナリが午前に動いていたか

```bash
git -C ros2_ws/src/sinsei_UMIUSI_control worktree add /home/satoimo/mujoco_ws/ros2_ws_b1fb99b/src/sinsei_UMIUSI_control b1fb99b
cd /home/satoimo/mujoco_ws/ros2_ws_b1fb99b
source /opt/ros/jazzy/setup.bash; source /home/satoimo/mujoco_ws/ros2_ws/install/setup.bash
MAKEFLAGS=-j2 nice -n 10 colcon build --packages-select sinsei_umiusi_control --cmake-args -DBUILD_TESTING=OFF   # 1 min 54 s
```
- 使うときは ros2_ws/install の上に `ros2_ws_b1fb99b/install/setup.bash` を source する。道具は `/proc/<cm>/maps` から実際にロードされた `libsinsei_umiusi_control_controller.so` のパスを印字する（`[replay] control libs:`）。
- パラメータは bag の `/parameter_events` から: 11:23 bag の attitude_controller は **kp 1.0 / kd 0.0**（yaml の 0.35 ではない）、`mixer.*` が一つも宣言されていない → 動いていたバイナリは `10b966d`（デッドバンドのパラメータ化、10/03 08:45）より古い。`AttitudeTarget` も `hold_yaw` 無しの旧型（msgs `dcd3642` 09-30 より前）。12:13 bag には parameter_events が無い（control 再起動なし）が `hold_yaw` 付きの新型で、13:45 の parameter_events は kp 2.0 / kd 0.5 / `esc_thrust_limit` 0.5。
- したがって「前 = `b1fb99b`」は近似: 11:23 は `b1fb99b` より古い（mixer デッドバンド 2° 固定、推力ヒステリシス無し）、12:13 は `cecef01`（12:02）を含む枝だった可能性が高い（sim の「後」が 12:13 の ±90° 挙動を再現し、「前」が再現しないことと整合）[assumed]。この表の「前」は `b1fb99b` + bag のゲイン（kp 1.0 / kd 0.0）。

## 4. sim と実機で合わないところ（候補と証拠）

- **roll/pitch の応答が大きすぎる（推力の大きさ・慣性/減衰）**: 12:13 の 30–45 s、実機も sim も duty ±0.5 の交互パターン（pitch モーメント）だが、実機は tilt ±6°・gyro x RMSE 基準、sim は ±40〜55°（gyro x RMSE 1.41）。推力を ×0.5 / ×0.25 にすると同区間の gyro x RMSE は 1.41 → 0.69 → 0.35、tilt 差 36° → 16° → 10° と下がるが、ヨーの追従（yaw cmd~sim corr）は 0.42 → 0.25 → −0.01 と崩れる（34 s: ×0.25 で gz 0.14 vs 実機 0.82）。→ 水平（ヨー）の推力効率は今の sim で合っていて、**垂直推力によるモーメント応答だけが数倍大きい**。候補: 垂直方向の推力効率（船体直下の prop）、roll/pitch の付加慣性・減衰（configs の 2026-09-27 注記と同じ方向）、実機のサーボが本当に ±90° に居たか（`thr_est_angle`）。
- **ヨー旋回・横移動で sim だけ roll/pitch が崩れる**: 13:45 の 50 s（ヨー +1 rad/s）で実機 roll +8.5° に対し sim −23.7°（符号も逆）。12:13 の 36–44 s（sway 全開）で実機 ±6°、sim 後で ±18°。水平推力→roll の結合（推力線の高さ、`cop_offset` / lift のモーメント）が実機と違う。
- **浮力トリム**: 実機は arm 中ずっと roll −2〜−4°・pitch −0.3〜−0.9°（13:45）で、それを打ち消す左右差（heave 中 lf +0.26 / rb −0.34）が duty に出る。sim は水平に浮くので ±0.30 で対称。
- **自由減衰**: disarm 中（13:45 の 58 s 以降）実機の IMU は動かないが、sim は 15 s ごとの再初期化後に roll ±5°・周期 ~4 s で振れる（buoyancy_offset 10 mm が 3 倍大きい既知の件）。
- **サーボの遅れ**: 今回の指標では分離できず。AUTO run の水平/垂直成分 corr 0.8〜0.94 なので、少なくとも 50 Hz・15 s 区間の尺度では支配的ではない。
- **ヨーの実測ゲイン**: ヨーレート指令 ±1 rad/s に対し実機 0.8、sim 0.9〜1.0。sim がやや強い（10〜20 %）。
- **frame**: bridge の経路は上記 3 点（quat の frame、gyro の慣性主軸、サーボ単位）を直さないと fb が成り立たない。この道具の変換（P = `bag_replay.FRAME_P`、`cad = imu(x, z, −y)`）で sim のヨー/roll/pitch の符号は controller の規約（mixer の a 行列）と一致することを単体で確認した（全基水平 + → gyro z +、lf/lb 上向き → gyro x +、lb/rb 上向き → gyro y +、surge + → v_x +）。

## 6. roll/pitch の較正（2026-10-08 追記）

開ループ（`tools/bag_replay.py` の K=25 ジャイロ再生、frame 修正後、12:13 / 11:23 / 13:45 / AUTO の 4 bag）で、垂直推力効率 × roll/pitch 回転抗力 × roll/pitch 付加慣性を格子探索し、閉ループで確認した。
プラントに opt-in の `thrust_vertical_eff`（推力 × (1 − (1 − eff)·sin²(servo))）を追加。`closed_loop_replay.py` の `--plant vertical_eff= / rp_drag= / rp_inertia=` で同じ knob を回せる。

| 開ループ RMSE [rad/s] | roll | yaw | pitch |
|---|---|---|---|
| floor（persistence） | 0.057 | 0.116 | 0.031 |
| stock（eff 1、×1、×1） | 0.170 | 0.090 | 0.066 |
| eff 0.25、抗力 ×4、慣性 ×3 | 0.091 | 0.077 | 0.039 |
| **eff 0.25、抗力 ×8、慣性 ×3（採用）** | **0.079** | 0.077 | 0.041 |
| eff 0.25、抗力 ×16、慣性 ×3 | 0.069 | 0.077 | 0.043 |

- eff は 0.25 より下げても変わらない。慣性 ×6 以上は付加質量の陽的積分が発散する。抗力 ×16 は roll を下げるが pitch が上がる。
- 推力の大きさ（`thrust_per_cmd`）は格子で 30 → 10 N が出たが、それは roll/pitch の誤差が支配していたときの値。閉ループのヨーが 30 N で相関 0.97〜0.99・ゲイン 0.9〜1.0（実機 0.8）なので、水平は 30 N のままにした。

| 閉ループ（ALL） | gyro x（roll） | gyro y | gyro z / 相関 | 傾き差 | mean\|duty\| bag/sim |
|---|---|---|---|---|---|
| 前 × 12:13、stock | 0.76 | 0.17 | 0.13 / 0.83 | 19.6° | 0.238 / 0.167 |
| 前 × 12:13、較正 | **0.10** | **0.04** | 0.10 / 0.89 | **4.6°** | 0.238 / 0.160 |
| 後 × 13:45、stock | 0.25 | 0.09 | 0.070 / 0.96 | 8.1° | 0.145 / 0.138 |
| 後 × 13:45、較正 | **0.076** | **0.035** | 0.060 / 0.97 | **3.1°** | 0.145 / 0.133 |

- 「前」の bang-bang（duty ±0.5、水平成分の相関 0.97）はそのまま再現し、傾きが実機の ±6° の尺度に収まる。これが「PID チューニング前後の違い」の sim 再現。
- 残り: roll の波形の相関は 0 付近（大きさは合ったが位相・向きが違う）。実機の常時 roll −2〜−4° のトリムと、ヨー旋回時の roll 結合の向きは未解決。
- `configs/umiusi.yaml` に反映（drag の idx 3/5 ×8、added_mass の idx 3/5 ×3、`max_duty: 0.5`）。
- **`thrust_vertical_eff` は 1.0 に戻した（同日）**: 0.25 は heave の力も 1/4 にし、競技 sim で 0.5 m の赤まで潜れず 0/8 になった。10/03 の実機は風船の高さまで潜れていた（前カメラに床の重りと風船が同じ高さで写る）ので 0.25 は実機と矛盾する。heave は未観測のため、roll/pitch の差は抗力・慣性の項で持つ。eff 1.0 でも較正に使っていない 4 bag で roll 誤差は旧比 −35〜−40%（0.115→0.067、0.071→0.043、0.158→0.095、0.106→0.063）で、0.25 のとき（0.063 / 0.036 / 0.094 / 0.058）とほぼ同じ。再現: `data/envfilt/` と同じ scratchpad の `gridrp*.py`、閉ループは §1 のコマンドに `--plant vertical_eff=0.25 --plant rp_drag=8 --plant rp_inertia=3`（yaml 反映後は不要）。

## 5. 詰まった点・確定できなかった前提

- controller_manager（Jazzy 4.45）は `use_sim_time` のとき `sleep_until(cycle_end + period)` で待つので、reply の直後に tick を打つと read()/update() 中に届いて 1 周期余計に待つ。reply 後 `--tick-delay` 2 ms 置いて tick、`--tick-timeout` 5 ms 内に request が来なければ次の tick を打つ。**CM 側の処理が 2 ms に収まるかは CPU 負荷次第**で、収まれば周期 10 ms、収まらなければ 20 ms になる（物理は CM の実測周期に追従するので controller 時間 = 物理時間は保たれ、controllers は元々 50 Hz なので制御側の挙動は同じ。違いは hardware read/write が 50 Hz になること）。今回: MANUAL / 12:13 系は 10.00 ms、AUTO と 11:23 は 20.00 ms。実測は `[replay] done:` の行に出る。
- 位相 A（spawner 待ちの wall-clock 動作）から lockstep に移る瞬間の request を「ジャンプの周期」として扱わないと、以後ずっと 2 tick/周期に固定される（直した）。
- `sim_server._recv_exactly` 由来の受信は、recv のタイムアウトで途中まで読んだ bytes を捨てる。bridge は長さ 4 byte と本体 80 byte を別の send で送るので、その間でタイムアウトするとストリームがずれて両側が永久に待つ（80 s 前後で 2 回再現、CM 側スレッドが `unix_stream_data_wait`）。`FrameReader` で置き換えた。
- bag 側の制約: MANUAL 13:45 は 57 s 以降 state topic が止まっている（control 停止）。11:23 は 29 s から control が起動。
- [assumed] 午前のバイナリは `b1fb99b` ではない（§3）。[assumed] 再初期化時の並進速度は 0（bag から分からない）。[assumed] 実機の `/cmd/*` の受信時刻 = 送信時刻（同一 Pi）。[assumed] sim の推力曲線（`thrust_per_cmd` 30 N、exp 2.0）は未校正のまま。
- 未変更の既存ファイル: `sim_server.py` / bridge の 3 点の不具合は報告のみ（直すなら `sim_server._encode_state` の gyro を `mjOBJ_XBODY` に、servo を rad に、quat/gyro/accel を P で REP-103 に変換）。
