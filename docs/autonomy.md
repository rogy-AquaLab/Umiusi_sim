# umiusi_sim — high-level autonomy (上位行動制御)

バルーンに対して *何をするか* を決定する**挙動レイヤ**である。
[perception](perception.md) スタックからの安定したトラック + IMU のヨーレートを入力とし、**ルールベースの有限状態機械(FSM)**を実行して、
3自由度の `{surge, heave, yaw}` インテントを出力する。これをフィードフォワードの配分器が8次元のスラスタアクションへ変換する。

> **ここに強化学習は存在しない。** 競技の自律制御は今日時点で**フィードフォワード / ルールベースのみ**である
> (`competition_run.py:5`, [`architecture.md`](architecture.md))。[`rl.md`](rl.md) にある RL ポリシーは*別個の*
> 低レベルトラック（姿勢 / 巡航）であり、バルーンミッションには接続されていない。

perception と同様に、FSM は ROS 非依存 / sim 非依存の `umiusi_perception` wheel に置かれているため、
sim (`tools/autonomy_run.py`) 上でもロボット (`navigator_node`) 上でも**ビット単位で同一**である。

**深度モード切替スーパバイザ（RL 側の上位ロジック）について:** 3-D の単一ポリシーが実用にならなかったため
（[`rl.md`](rl.md) 参照）、深度しきい値で**水平ポリシーと降下ポリシーを切り替えるスーパバイザ**が導入されている。
これはこの FSM とは別物で、**ロボット側に実装されている**（`sinsei_UMIUSI_autonomy` PR #17）。sim 側の
リハーサル（しきい値 / ヒステリシス / 切替過渡の測定）は `tools/mode_switch_eval.py` が担う。
運用上の不変条件: **水平ポリシーに鉛直の `v_cmd` を渡してはならない**（分布外であり、姿勢を崩す）。

## Where it lives

| piece | file |
|---|---|
| 挙動 FSM (`BalloonBehavior`) | `packages/perception/src/umiusi_perception/autonomy/behavior.py` (693 lines) |
| スラスタ配分（フィードフォワード） | `packages/perception/src/umiusi_perception/control.py` |
| マルチフレームトラッカ + 色サニタイザ | `.../tracker.py` |
| Sim フロントエンド（perception-in-the-loop） | `tools/autonomy_run.py` |
| グラウンドトゥルースドライバ（ビジョン無し） | `tools/competition_run.py` |
| 競技シナリオ（フィールド、得点） | `packages/sim/src/umiusi_sim/description/scenarios/competition_balloon.py` |
| シナリオテスト | `tests/test_competition_balloon.py` |
| フィールド設定 | `configs/umiusi.yaml` (`competition.balloons`) |
| ROS デプロイノード（兄弟 ws） | `ros2_ws/src/umiusi_autonomy/{perception_node,navigator_node}.py` |

---

## 1. The behaviour FSM — `BalloonBehavior`

ステートフルなルールベース FSM である (`behavior.py:209`)。エントリポイント `step(detections, yaw_rate, heading, dt, fresh)`
(`:408`) は制御ステップごとに呼ばれ、`(command, info)` を返す。ここで `command = {surge, heave, yaw}` であり、
**カメラのみに基づく判断であって、グラウンドトゥルースは一切読まない**。各ステップは順に (`:413`)、検出結果を距離ゲート / サイズフィルタで処理 →
`tracker.update` → 青回避ヨーを計算 → 諦め記憶を減衰 → ロック済み目標を更新、という流れである。

### States

| state | trigger | what it does |
|---|---|---|
| **SEARCH** (`:618`) | 到達可能な赤 / 黄が見えない | 360° その場ヨー旋回スイープ (`SEARCH_YAW=0.5`) + 高さスキャンのヒーブ (`SCAN_HEAVE=0.15·sin`)。空振りの旋回が一巡すると方向を反転し、新たな地点へ**並進**する (`SEARCH_SURGE=0.30`, ~1 s) |
| **APPROACH** (`:549`) | 到達可能な非青トラックをロック | 方位角へヨー `KP_YAW·az + KD_YAW·yaw_rate + avoid`; 深度へヒーブ; サージ `SPEED_CAP·max(0,cos az)·el_scale`。`|az|>45°` ならサージ ×0.3、青 / ワイヤを弧を描いて回避中は ×0.5 |
| **ALIGN** (`:523`) | `bbox_frac ≥ ALIGN_BBOX=0.18` | ゆっくりと、バルーンを精密に中央へ据える。クリープ前進 (`ALIGN_CREEP=0.12`)、`_committed` を設定。タイムアウト `>110` ステップ → REPOSITION |
| **RAM** (`:485`) | `bbox ≥ 0.26` かつ整列済みかつ安定（3 ステップ） | まっすぐ突入する (`RAM_SURGE=0.26`。ピン到達前にヒーブが深度レースに勝つよう、意図的に `SPEED_CAP=0.35` より小さくしてある) |
| **CONFIRM** (`:564`) | 突入(ram)後 | 後退し (`−0.34`)、ロック済みトラックを監視: `CONFIRM_FRAMES=55` の間消失 + 信頼できる → **POP**; 再出現 → **MISS** |
| **RECOVER** (`:600`) | ミス / タイムアウト後 | 逆進 (`−0.28`) ~0.6 s、その後、目標が生存していれば再 ALIGN、そうでなければ SEARCH |

- **ブラインド突き抜け** (`:446`): RAM 序盤でコミット中 (`<LUNGE_STEPS=26`) にトラックを見失った場合、
  凍結した方位角のまま駆動し続ける — ピンがバルーンに到達するのはフレームから外れた*後*だからである。
- **Pop の信頼性** (`:576`): `peak_bbox ≥ 0.26` かつ目標がほぼ中央 (`|az|,|el| ≤ 18°`) で消失した場合のみ pop とカウントする。
  背の高いバルーンを*下から*通過した場合は上端方向へ抜けるため MISS と判定され、pop にはならない — こうしてカメラのみの確認が過大カウントしないようにしている。
- **放棄 / 諦め** (`:362, :396`): 一つの目標に対する突入(ram) `MAX_ATTEMPTS=6` 回、または `MAX_PURSUIT_STEPS=500` (~10 s) で放棄する。
  **色**ごとの失敗記憶: ある色で `GIVEUP_FAILS=4` 回ミスした後は、その色を `SUPPRESS_STEPS=350` (~7 s) 抑制し、
  目標選択が到達可能な代替へ流れるようにする。

### Target selection — nearest reachable non-blue
`_pick_target` (`:248`) はトラッカの **CONFIRMED** トラックの中から選択する: **`colour ∈ {red, yellow}` から抑制中の色を除いた中で `min(range_m)`** である。
これは意図的に色優先 / 得点最大化ルールにはしていない (`:51`): 約19バルーンのフィールドでは、近いものから遠いものへと順に片付ける方がスループットを最大化し、
近くの黄を素通りして遠くの高得点の赤を追うよりも係留線の絡まりを防ぐのに優れているためである。

- **距離ゲート** `MAX_TARGET_RANGE=4.5 m` — これより遠い検出結果は到達不能 / 遠方の誤検出として破棄する。
- **PREEMPT** (`:468`): コミット前に、より近い非青トラックが現在の目標を `PREEMPT_MARGIN=0.30 m`（ヒステリシス）上回るなら再ロックする —
  経路上のバルーンを先に割り、係留線を片付ける。
- **青回避** (`_blue_avoidance`, `:312`): 確定済みの青が `|az|<28°` かつ `<1.6 m` 内 → 離れる向きのヨーバイアス。
  青は*決して*目標にならない（−10 のデコイである）。回避のためだけにトラックする。
- **ワイヤ回避** (`_wire_avoidance`, `:325`): 確定済みの非目標トラックが真正面から横方向に `PATH_CLEAR_RADIUS=0.40 m` 内かつ
  `<1.6 m` にある場合、その係留線の下を通るのではなく弧を描いて回避する (`0.40 = TETHER_RADIUS 0.20 + 船体半幅`)。
- **色の健全性チェック** (`sanitise_near_colours`, `tracker.py:329`): 近距離 (≤2.5 m) の赤 / 青についてボックスのピクセルを読み直す。
  「赤」が青を読んでいた場合はラベルを付け替える — −10 のデコイが決して割られないための安全策である。

### Detections + IMU → command (proportional / PD laws)
- **yaw** `= clip(KP_YAW·az + KD_YAW·yaw_rate + avoid + wire, −1, 1)`, `KP_YAW=1.1`, `KD_YAW=0.15`。
  `yaw_rate`（IMU / sim `ang_vel[1]` 由来）が**唯一の慣性入力**であり、D 項として用いる。
- **heave** `= clip(KP_HEAVE·(el + aim_bias), ±SPEED_CAP)`, `KP_HEAVE=1.5`。`aim_bias` は**黄のみ**で照準を中央より 5° 上に上げる
  （背の高い 1.5 m のバルーンにはやや上方から接触する）。
- **surge** — 状態依存で、`_el_surge_scale`（**深度優先ゲート**, `:657`）でスケールされる: 照準深度から 5° 以内でフルサージ、
  15° までに床値 `0.22` へテーパーする — 突入する前に車体を正しい高さに合わせるためである。

---

## 2. Competition scenario — `competition_balloon.py`

バルーン割りミッションである。物理は解析的な流体力学 sim であり、プールは**視覚上のみ**である
(`POOL 8×5×3.3 m`, 初期姿勢 `(0, 1, 0)` ≈床から 1 m)。

**Balloons** (`BALLOON_SPECS`, `:55`; 半径 0.10 m / 直径 0.20 m):

| colour | height | points | role |
|---|---|---|---|
| red | 0.5 m | **+30** | primary |
| yellow | 1.5 m | **+10** | tall (met from above) |
| blue | 0.7 m | **−10** | decoy (avoid) |

**Field** (`sample_layout`, `:328`): エピソードごとの Poisson-disk / リジェクションサンプラ。XY のみランダム化し、
高さは色で固定、シード付き → 再現可能である。バルーン 0 は常に決定論的な背の高い黄の「スタート」バルーンである（黄1個としてカウント）。
個数 + 最小分離は `configs/umiusi.yaml → competition.balloons` から取得する
（デフォルトは**赤 7 / 黄 7 / 青 5 = 合計 19**、`min_separation 0.6 m`; PLACEHOLDER、ルールブック比率は TBD）。

**Pop rule** (`popped`, `:239`) — 三つのゲートを同時に満たす:
1. **近接** — ピン先端がバルーン中心の `0.10 + POP_MARGIN 0.03 = 0.13 m` 内にある;
2. **正面** — ピン軸が 先端→中心 方向から `POP_ANGLE_TOL_DEG=20°` 内にある;
3. **速度** — 接近速度 ≥ `MIN_POP_SPEED=0.18 m/s`。

20° の許容は*実測された*スイートスポットである: ピンがカメラ / 重心から横方向にオフセットしているため、
狙いの良い突入(ram)は約 19–24° で球体に入る (`:110`)。**ピン**は `base_link` の剛体子であり、
`PIN_TIP=(0.40, 0.10, 0.0)`; `hide_balloon` (`:276`) は割れたバルーンの alpha=0 を設定してカメラから消し、
FSM のカメラのみの pop 確認を可能にする。

**Tether entanglement** (`entanglement`, `:378`): ロボットが水平距離 ≤ `TETHER_RADIUS=0.20 m` **かつ**
バルーンの高さより下にある場合、そのバルーンの下を通過したことになる。これがワイヤ回避ガードが最小化する指標である（低いほど良い）。

---

## 3. Feed-forward allocation — `control.py`

`feedforward_allocation(target_orientation[3], target_velocity[3], servo_range_deg=90)` → 8次元アクション
`[servo_1..4, esc_1..4]`, 各要素 ∈ [−1, 1] (`:57`)。実機の
`sinsei_umiusi_control` `feed_forward.hpp` AttitudeController の純フィードフォワード移植であり、フィードバックは無い。

- **8×6 配分行列 `_ALLOC`** (`:36`): 行 `[f1h,f1v,…,f4h,f4v]`, 列 `[Φx,Φy,Φz,Vx,Vy,Vz]`。
  すべての垂直行が `Vz=1.0` を持つため、**純粋なヒーブは4つのスラスタすべてを対称的に上向きに駆動する**
  （ヨーカップルは生じない — 一つの `Vz` が 0 であった既知バグが ~0.84 rad/s のスピンを引き起こしていた, `:33`）。
- `f = _ALLOC @ u` から各スラスタごとに: サーボ方位 `= atan(f_v / f_h)`（±90° にクランプ）; 大きさ
  `= hypot(f_h, f_v)`; 水平成分が後ろ向きを指すとき ESC 符号を反転; `thrust = ±mag/√2`。
- **フレーム規約** (`:13`): sim 軸 ≠ コントローラ軸 — 経験的に `Vz → +Y (heave up)`,
  `Vx → −X (surge, sign-flipped)`, `Vy → yaw couple`。呼び出し側はインテントを明示的にマッピングする:
  `feedforward_allocation([0,0,yaw], [−surge, 0, heave])` (`autonomy_run.py:255`, `navigator_node.py:143`)。
  FF フレームの符号整合は追跡中の**ハードウェア立ち上げのフォローアップ**である (README)。

---

## 4. Integration — sim front-end & ROS deploy

**Sim** (`autonomy_run.py`, `autonomy_step` `:231`), 制御ステップ (50 Hz) ごとに:
1. `perc_stride` ステップごとに perception ティック（~`--perception-hz`, デフォルト 10 Hz、その間は保持）:
   `sim.render_camera("front_cam", degrade=True)` → `sanitise_near_colours(rgb, detector(rgb))`, detector =
   `load_learned_detector("examples/balloon_detector/model.pt")`。
2. FSM: `behavior.step(detections, ang_vel[1], heading, dt, fresh)`。
3. 駆動: `feedforward_allocation([0,0,yaw], [−surge,0,heave])` → `sim.step(action)`。
4. グラウンドトゥルースの pop 判定 (`scn.popped(pin_tip, b_pos, pin_axis, pin_vel)`) → 得点加算 + `hide_balloon`;
   係留線の下潜りを集計。すべての正のバルーンが割られるか、`--minutes`（デフォルト 4）でエピソード終了。

ティック間で検出結果を保持することは、現実的な Pi-4 のタイミングを反映している。`competition_run.py` はグラウンドトゥルース版の
対応物である（バルーンのワールド座標を直接読み、同じ配分 + 得点、ビジョン無し）。

**ROS deploy** (`ros2_ws/src/umiusi_autonomy/`, 同じ wheel → ビット単位で同一の挙動):
- `perception_node.py` — `Image` → `load_learned_detector` + `sanitise_near_colours` → `BalloonDetectionArray`,
  レート上限 `max_rate_hz=10`, torch は遅延インポート。
- `navigator_node.py` — `control_hz=50` の薄い `BalloonBehavior` ラッパ; `fresh=True` は新しい検出メッセージ後の
  ティックでのみ真（sim の検出保持による再駆動を反映）; `{surge,heave,yaw}` →
  `feedforward_allocation` → `/cmd/direct/...` 上の4つの `ThrusterOutput`（変更されていない
  `sinsei_umiusi_control` スタックを駆動）。デプロイ上の注意: `ImuState.angular_velocity` は deg/s（変換される）であり、
  FF フレームの符号整合が未解決の立ち上げ項目である (`navigator_node.py:15`)。

---

## 5. Performance

ランナ (`autonomy_run.py:334`) は次を報告する: 最終スコア、popped/total（正 vs 青の内訳）、sim 秒数、perception ティック、
状態ごとの FSM 時間、突入(ram)試行 / カメラ確認 pop / ミス / **pop 率**、放棄目標、GT 得点 vs カメラ確認 pop、係留線の下潜りイベント。
セルフテスト PASS (`:367`) = `score > 0 AND popped ≥ 1 AND no blue popped`。

| metric | value | source |
|---|---|---|
| グラウンドトゥルースドライバ (`competition_run`) の典型的な最終スコア | **~80** | README |
| 突入/pop の信頼性（現在の律速レバー） | **~15 %** hit-rate | README, `ram_eval.py:5` |
| 検出器の赤 / 青リコール（学習済み、40枚ベースライン） | 0.82 / 0.63 | README |

**Status** (README): 競技 sim ✅, perception + FSM ✅（エンドツーエンドで実行可能）; 🟡 sim-to-real
チューニング + Pi-4 デプロイは保留中。未解決のレバー: **突入/pop の信頼性 (~15 %)**、sim-to-real の検出器品質、
FF フレームの符号整合、および Pi 用の aarch64 MuJoCo ビルド。`tools/ram_eval.py` は突入の信頼性を単独で攻略するためのツールである
（完璧な perception → FSM、失敗の分類）。
