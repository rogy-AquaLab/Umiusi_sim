# 実機の AUTO 経路で競技 / 風船 1 個を sim で回す — `tools/competition_ros.py`（2026-10-08）

## 結論（先出し）

- **実機の AUTO 経路だと、ヨーの向きが逆で風船から離れていく。** `competition.yaml` の `yaw_rate_scale: 1.0` のままだと、FSM の +yaw（右に回る＝画面右の風船へ向く）が control の +`yaw_rate`（REP-103 で左回り）になる。
  - control は指令どおりに回っている（指令 yaw_rate と gyro z の相関 0.74〜0.98）。
  - それでも、追っている風船の方位と回る向きの相関は **+0.52〜+0.66**（正しく向かうなら負）。風船の方へ回った時間の割合は 0.29〜0.33。
  - `--yaw-rate-scale -1` にすると、相関は −0.42〜−0.76、風船の方へ回った割合は 0.66〜1.0 になる。
  - field の結果: ≥1 個割れた run が **0/2 → 2/2**（3 個と 6 個、初回 15.8 s / 24.7 s）。
  - 実機で直すなら `ros2 param set /auto_target_generator yaw_rate_scale -1.0`。ただし、実機のカメラが上下逆に付いていないことが前提 [assumed]（field_card 6. の確認項目そのもの）。
- **風船 1 個（single）は、ヨーを直しても sim の既定プラントでは 0/4。** 原因は 2 つ。
  - **(a) 正浮力 +1.14 N で浮き上がる。** SEARCH の heave は平均 0 の揺らしで深さを保持しないので、水面（y≈3.05）まで浮く。そこからは床上 0.5 m の赤が視野（下 30°）に入らない。
  - **(b) 突進の当たり角が割れ判定の境界にかかる。** ピン先が風船に入ったときの角度が 21° と 34° で、判定の上限 20° を超えた（`POP_ANGLE_TOL_DEG`、未実測の値）。
  - 中性浮力（`--plant net_buoy=0`）にすると **3/4 で割れる**（初回 13.8 / 22.2 / 29.9 s、当たり角 4〜14°、閉じる速さ 0.37 m/s）。
- **(a) は sim 内の古い経路でも同じ**（single は 4 本とも水面に張り付いて 0/4）なので、実機経路だけの問題ではない。実機の浮力が本当に正なら、実機でも「低い風船が見えない」が起きる [assumed]（実機の正味浮力は未実測）。
- **`ki_heave` 0.3 の偏りが SEARCH に持ち越され、潜り続ける。** 実機経路の field（yaml の FSM、ヨー逆）では、2 本とも途中から沈み続けた（y 最小 −2.7 / −6.5 m。sim には床の当たりが無いので突き抜ける）。single の中性浮力の 1 本も同じ形。
  - 古い経路の既定（behavior.py の値）は `ki_heave` 0 なので、この症状は出ない。
  - competition.yaml が勧める値（`ki_heave 0.3`）で初めて出る。実機ではプールの床に張り付くはず [assumed]。
- **古い経路（competition_eval `--driver deploy`）とは、同じ FSM 値でもふるまいが違う。** field で FSM 値を揃え、ヨーも直した比較:
  - 古い経路: ALIGN 59 %、突進 2.5 回 / 本。
  - 実機経路: 突進 30 回 / 本、ALIGN 19 %。
  - 古い経路は「正対して止まる」ができず、実機経路は正対まで行けている。そこから先（突進の速さ、当たり角）は FSM の値の違い（`ram_surge` 0.26 と 0.6）が効く。どこで差が出るかは未確認（§4）。
- 道具の状態:
  - 制御周期はロックステップで 10.00 ms。実時間比は ×1.94〜2.00。
  - 3 分のエピソード 1 本が約 1.6 分（起動に +10 s 前後）。
  - `--video` で、前カメラ（検出の箱・状態・時刻・点数）と三人称の動画を作れる。動画はエピソードの後に描くので、実時間比は落ちない。

## 1. 道具と起動手順

### 構成

```
GT 検出（competition_eval の視野・距離・5 Hz・取りこぼし・誤検出・recall 曲線・phantom）
  -> BalloonDetection msg -> AutoTargetGenerator._to_detection   ┐ umiusi_autonomy を import（コピーしない）
  -> live_detections（detections_timeout_s 0.5）                  │ fsm_params で competition.yaml の fsm.* を反映
  -> BalloonBehavior.step(dets, IMU z × yaw_rate_sign, heading=0, dt=実測)
  -> to_control_setpoint -> /cmd/target, /cmd/attitude_target（水平・hold_yaw false）、/cmd/thruster_runnable_all（0.5 s ごと）
  -> sinsei_umiusi_control（ros2_control_node + gate/attitude/thruster×4、無改変、use_sim_time）
  -> umiusi_sim_bridge -> Unix socket -> tools.sim_server.SimServer（REP-103 姿勢・body gyro・サーボ rad）
       競技シーン（風船・ピン）、プラントの max_duty 1.0（control の 0.5 だけ効く）、水面 water_surface_y = POOL_DEPTH
```

- ロックステップ（1 要求ごとに /clock を +10 ms）と FrameReader・パラメータファイルの作り方は `tools/closed_loop_replay.py` のものを使う（import、または同じ手順）。
- 1 エピソードごとに control を起動し直す（コントローラの内部状態を持ち越さない）。
- 採点は competition_eval と同じ: `scn.popped`（20°、0.18 m/s）、`scn.entanglement`、`summarize`。
  - 割れたかの判定は、物理の 1 周期（10 ms）ごとに行う。competition_eval は 20 ms ごと。
  - 同じ seed なら同じ配置になる（ep seed = 1000 + seed + e、乱数を引く順も同じ）。
- 検出の 1 フレーム分は、competition_eval.run_episode の中のブロックを同じ乱数の順に写した `detector_frame`。competition_eval は別セッションが編集中だったので、関数に切り出さずに写した。直すときは両方を直す。
- FSM の周期は competition.yaml の `control_hz`（50 Hz、`--fsm-hz` で変えられる。10/03 実機の AUTO は 26 Hz）。認識は `--perception-hz` 5。

### 起動

```bash
cd /home/satoimo/mujoco_ws/umiusi_sim
source /opt/ros/jazzy/setup.bash
source /home/satoimo/mujoco_ws/ros2_ws/install/setup.bash     # control da05604 + bridge + umiusi_autonomy
export MUJOCO_GL=egl                                            # --video のとき
nice -n 10 uv run python -m tools.competition_ros --layout single --single-colour red --episodes 4 --minutes 3 --verbose \
    --out out/competition_ros/x --video out/competition_ros/x_ep0.mp4
#   --layout field|single  --single-range 1.5,4.0  --seed 0
#   --param attitude_controller.feedback.kp_roll=1.0     # controllers.yaml の上書き（node.dotted=value）
#   --fsm ram_surge=0.4                                  # fsm.* の上書き（competition.yaml の単位）
#   --fsm-defaults                                       # competition.yaml を使わず behavior.py の既定（competition_eval と同じ）
#   --yaw-rate-scale -1  --surge-sign 1                  # auto_target_generator のパラメータの上書き
#   --plant net_buoy=0  --plant vertical_eff=1.0         # 切り分け用のプラントの調整
#   --dropout / --bearing-noise-deg / --fp-per-frame / --recall-curve / --phantoms / --perception-latency（competition_eval と同じ）
#   --video-episode 0 --video-fps 12.5  --domain 78（closed_loop_replay は 77）
```

- 出力:
  - `--out` に `epNN_seedS.npz`（`trace` は FSM の周期ごと、列は `cols`: 位置・方位・gyro z・FSM 指令・Target・追跡中の方位/仰角・duty/サーボ角・状態）と `summary.json`。
  - verbose のときは、エピソードごとに「ピン先が風船に入ったときの閉じる速さ・角度・状態」を印字する（割れなかった理由を見るため）。
- 同時に回すのは 1 本まで。`ROS_DOMAIN_ID` が同じ run を 2 本並べないこと。

## 2. 結果（3 分、認識 5 Hz、劣化なし、seed 0）

各セルの値: ≥1 個割れた run / 全 run、初回の時刻、平均の点、FSM の状態の配分。

### single（赤 1 個、1.5〜4.0 m、どの方位にも置く。4 本）

| 経路 | FSM 値 | ヨー | プラント | ≥1 割れ | 初回 | FSM 配分 | 備考 |
|---|---|---|---|---|---|---|---|
| 古い（competition_eval、cap 0.5） | 既定 | 正（yaw_sign −1） | 既定 | 0/4 | — | SEARCH 88 / APPROACH 7 / ALIGN 4 / RECOVER 1 | 4 本とも水面（y 3.08）へ |
| 古い | 既定 | **逆**（yaw_sign +1） | 既定 | 0/4 | — | SEARCH 86 / APPROACH 9 / RECOVER 5 | 同上 |
| **実機経路（そのまま）** | yaml | **逆**（yrs +1） | 既定 | **0/4** | — | SEARCH 92 / APPROACH 4 / RECOVER 2 / ALIGN 1 / CONFIRM 1 | 方位と回る向きの相関 +0.52〜+0.66。1 本は沈んで −2.4 m |
| 実機経路 | yaml | 正（yrs −1） | 既定 | 0/4 | — | SEARCH 95 / APPROACH 2 / RAM 1 / ALIGN 1 | 2 本で突進が届いたが、当たり角 21° / 34° で割れず。2 本は水面 |
| 実機経路 | 既定 | 正 | 既定 | 0/4 | — | SEARCH 95 / APPROACH 2 / RAM 1 / ALIGN 1 | 同上 |
| **実機経路** | yaml | 正 | **中性浮力** | **3/4** | 13.8 / 22.2 / 29.9 s（中央値 22.2） | SEARCH 81 / APPROACH 11 / ALIGN 5 / RAM 3 | 当たり角 4〜14°、閉じる速さ 0.37 m/s。外れた 1 本は 54° で入り、その後沈み続けた |
| 古い | 既定 | 正 | 中性浮力 | 0/4 | — | SEARCH 70 / RAM 11 / APPROACH 7 / ALIGN 5 / RECOVER 4 / CONFIRM 3 | 突進 13 回 / 本、外れ 11 回 / 本、ひもの下くぐり 5〜14 回 |

### field（競技場、`scn.sample_layout`、正の風船 14 個。2 本）

| 経路 | FSM 値 | ヨー | ≥1 割れ | 割れた数 / 点 | 初回 | FSM 配分 | 突進 / 外れ（本あたり） | 備考 |
|---|---|---|---|---|---|---|---|---|
| 古い（cap 0.5） | 既定 | 正 | 0/2 | 0 / +0 | — | ALIGN 59 / APPROACH 23 / RECOVER 14 / RAM 2 | 2.5 / 3.0 | y 0.9〜1.8 を保つ |
| **実機経路（そのまま）** | yaml | **逆** | **0/2** | 0 / +0 | — | SEARCH 57 / APPROACH 19 / RECOVER 11 / ALIGN 7 / CONFIRM 5 / RAM 1 | 1.5 / 8.5 | 相関 +0.57 / +0.62。沈み続けて y −2.7 / −6.5。ひも 6.5 回。動画の 1 本も 0 |
| **実機経路** | yaml | **正** | **2/2** | 3・6 / 平均 +45 | 24.7 / 15.8 s | SEARCH 37 / ALIGN 21 / APPROACH 17 / RAM 15 / CONFIRM 6 / RECOVER 4 | 15 / 7 | ひも 5 回 / 本 |
| 実機経路 | 既定 | 正 | 1/2 | 0・1 / +5 | 149.9 s | RAM 25 / APPROACH 22 / SEARCH 22 / ALIGN 19 / RECOVER 10 / CONFIRM 2 | 30.5 / 28.5 | 突進は届くが遅い（`ram_surge` 0.26） |

- 全 run で制御周期は 10.00 ms（最大 10.02 ms）、実時間比 ×1.94〜2.00。
- `thrust_vertical_eff` 0.25 の時期に回した run（垂直推力 1/4、水面の扱いなし → 水面を突き抜けて y 12 m）は `out/competition_ros/stale_vert025/` に置いた。比較には使っていない。

### 動画（`out/competition_ros/`、左: 前カメラ + 検出 + 状態/時刻/点数、右: 三人称）

- `ros_field_ep0.mp4`: field、実機経路そのまま（ヨー逆）。風船から逸れていく様子。
- `ros_field_yrs-1_ep0.mp4`: field、ヨーを正にしたもの（3 個割る）。
- `ros_single_red_ep0.mp4` / `ros_single_red_yrs-1_ep0.mp4`: single、それぞれヨー逆 / 正。

## 3. 切り分け（1 つずつ）

1. **ヨーの向き（実機経路だけの問題）**
   - 実機経路の規約:
     - FSM の yaw と方位は「+ = 右（body +Z、画面右）」（behavior.py、`BalloonDetection.azimuth` のコメント）。
     - control の `AttitudeTarget.yaw_rate` は REP-103（+ = 左回り）。closed_loop_replay で「指令 + → gyro z +」を確認済み。
     - `to_control_setpoint` は `yaw_rate_scale` を掛けるだけで、符号は反転しない。したがって既定の +1 では逆に回る。
   - 古い経路（DeployDriver）は `yaw_sign=-1` で反転している（`tools/deploy_driver.py` の YAW SIGN の項）。
   - FSM に渡すヨーレートは、どちらの経路も「+ = 左回り」（実機経路は IMU z、古い経路は CAD +Y）。
     - ヨーの向きを直すと、KD 項（`KD_YAW·yaw_rate`）も減衰の向きになる。
     - 逆のままだと、P 項は風船から離れる向き、D 項は回転をあおる向きになる。
   - 数字: 表の相関と「風船の方へ回った割合」（`trace` から計算。§1 の npz）。
   - 古い経路で yaw_sign を +1 にしても single では差が出なかった。どちらも水面に張り付いて風船を見ていないため、この比較では向きを判別できない。
2. **浮上（両方の経路に共通、プラントと FSM）**
   - 何もしないとプラントは 0.06 m/s で浮く（net +1.14 N）。
   - SEARCH は深さを持たない（scan_heave の揺らしは平均 0）。control の heave は開ループの推力の割合。
   - そのため数十秒で水面に着き、床上 0.5 m の風船はカメラの下 30° の外になる（距離 2 m なら仰角 −51°）。
   - `net_buoy=0` にすると single は 0/4 → 3/4。実機の正味浮力は未実測で、configs は「バラスト前の船体」の値 [assumed]。
3. **沈み続ける（yaml の `ki_heave` 0.3）**
   - APPROACH で、仰角の誤差から heave の偏りを積分する（上限 0.25）。この偏りは SEARCH に入っても消えない。
   - 低い風船を追ったあとに SEARCH に戻ると、heave −0.25 前後のまま潜り続ける。sim は床で止まらないので突き抜ける。
   - 古い経路の既定（ki_heave 0）では出ない。
   - FSM 側の問題（目標が無いときの偏りの扱い）。実機ではプールの床に着き、上を向けないまま SEARCH を続けるはず [assumed]。
4. **当たり角（両経路）**
   - ピンはカメラより 0.30 m 前・2 cm 下にある。FSM はカメラの中心に風船を合わせ、`pin_offset` を渡していない（auto_target_generator も同じ）。
   - 球（半径 0.13 m）に入るときの角度は 4〜54° にばらつく。20° を超えると割れない扱いになる。
   - 20° は未実測の値（competition_eval の help）。
5. **古い経路と実機経路の差（同じ FSM 値・正しいヨー、field）**
   - 古い経路は ALIGN 59 % で突進 2.5 回、実機経路は突進 30 回。
   - 古い経路は正対（中央 6° 以内で 3 周期静止）を満たせずに ALIGN の時間切れを繰り返している、と読める。
   - 原因は、DeployDriver の姿勢・ヨー制御（bundle のゲイン、m/s の指令）と実機 control（推力の割合の指令、fb kp 2.0 / kd 0.5）の違い。ただし、ヨーレートの応答と横流れのどちらが効くかは未確認。

## 4. 未確認 / 前提

- [assumed] 実機のカメラは上下左右とも正しい向きに付いていて、画面右 = 右舷（sim と同じ）。逆さなら `yaw_rate_scale` +1 が正しい。field_card 6. の「右前に置いたら右に回るか」で確定させる。
- [assumed] 実機の `/state/imu` は control が出す IMU の値（sim では bridge に返す値そのもの）。ImuSanity は通していない（化けサンプルは sim に無い）。
- [assumed] 検出の遅れは 0（`--perception-latency` で入れられる）。auto_target_generator の受信時刻は sim 時刻で計っている。
- 古い経路との差（§3.5）の内訳は未確認。古い経路の正対の時間切れを trace で見るのが次の 1 手。
- 認識の劣化（取りこぼし・誤検出・recall 曲線）は入れていない。オプションはある。
- `tools/closed_loop_replay.py` の `ReplayServer` を確認した。`step_command` と `_encode_state_imu` を自前で持っていて、sim_server の新しい変換（REP-103・body gyro・rad）と式は同じ。基底の変換は通らないので、二重の変換にはなっていない。重複しているだけ（未変更）。
- 既存ファイルは変更していない。新規: `tools/competition_ros.py`、この文書、`out/competition_ros/`（run_stage*.sh・log・npz・mp4）。

## 5. 再現

```bash
cd /home/satoimo/mujoco_ws/umiusi_sim
# 表の各行（out/competition_ros/run_stage1.sh 〜 run_stage5.sh にそのまま入っている）
O=out/competition_ros
R="nice -n 10 uv run python -m tools.competition_ros --minutes 3 --verbose"
$R --layout single --single-colour red --episodes 4 --out $O/ros_single_red                  # そのまま
$R --layout single --single-colour red --episodes 4 --yaw-rate-scale -1 --out $O/ros_single_red_yrs-1_diag
$R --layout single --single-colour red --episodes 4 --yaw-rate-scale -1 --plant net_buoy=0 --out $O/ros_single_red_yrs-1_nb0
$R --layout field --episodes 2 --out $O/ros_field
$R --layout field --episodes 2 --yaw-rate-scale -1 --out $O/ros_field_yrs-1
$R --layout field --episodes 2 --yaw-rate-scale -1 --fsm-defaults --out $O/ros_field_yrs-1_fsmdef
# 古い経路（同じ seed）
nice -n 10 uv run python -m tools.competition_eval --layout single --single-colour red --episodes 4 --minutes 3 --perception-hz 5 --max-duty 0.5 --verbose [--yaw-sign 1] [--net-buoy 0]
nice -n 10 uv run python -m tools.competition_eval --layout field --episodes 2 --minutes 3 --perception-hz 5 --max-duty 0.5 --verbose
```
