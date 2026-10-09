# 引き継ぎ: sim・学習側の進捗（2026-10-09）

10/03 の実機実験のあと（10/05〜10/08）に sim・学習側でやったことのまとめ。共同作業者向け。
実機（Pi）で確かめたものは無い。数値はすべて 10/03 の bag・映像を手元で解析したもの。

## 1. 結論

| 項目 | 結果 | 詳細 |
|---|---|---|
| 検出器 | 初見の濁ったプール（10/03）で F1 **0.10 → 0.53**。`balloon_F320_20261007` として autonomy #46 で実機の既定に採用済み | `env_filter_study.md` |
| sim 画像の効果 | 実映像から測った「会場フィルタ」を sim 画像に掛けると、初見会場で **0.14 → 0.65**。その会場の実画像があるときは効かない | §2 |
| 機体の物理 | 10/03 の bag（PID チューニング前後）を実機の control ごと sim で再現。roll/pitch を実機に合わせて較正 | `closed_loop_replay_20261003.md` |
| 競技 | 実機の AUTO 経路そのままで、sim の競技場 3 分で 8 個・+120〜180 点。**ただし `yaw_rate_scale` を −1 にした場合** | `competition_ros_20261008.md` |
| 学習済み policy（RL） | 変更なし。RL は封印中で、競技は古典制御 + FSM。同梱の policy（`models/av_mode13` など）は従来どおり | `rl.md` |

## 2. 検出器と sim 画像

- 会場フィルタ（`tools/env_filter.py`）: 実映像 100〜200 フレームから、かすみ（低周波の色）・コントラストの落ち方・ぼけを測り、sim 画像に同じ劣化を掛ける。ラベル不要。
- 効き方:

| 条件 | F1 |
|---|---|
| 汎用 sim を混ぜる（初見会場） | 0.35 → 0.29（**逆効果**） |
| 手で寄せた sim（同じ会場の実画像あり） | 0.70 → 0.64（効かない） |
| **会場フィルタ付き sim（初見の 10/03）** | **0.14 → 0.65** |
| 同上・JAMSTEC 保留 | 0.47 → 0.46（変わらず） |

- 実画像にフィルタを掛けるのは逆効果（風船が消えて箱だけ残る）。
- 取れていないもの: 遠く淡い風船（10/03 144510: 0.29 → 0.16）、重りを赤と誤検出、水面の鏡像・光の粒。sim にはこれらが無い。
- 運用の閾値は **0.3**。0.4 だと濁りで全滅する。
- 同梱物: `examples/balloon_detector/balloon_F320_20261007.pt` / `_320.onnx`（PyTorch と 40/40 枚で一致）/ `envfilt_bank_20261007.npz`（フィルタの参照）。
- 会場での再学習の手順（約 3.5 時間、CPU 8 コア）: `env_filter_study.md` の末尾。

### 画像（共有フォルダ `/srv/share/`）

| フォルダ | 中身 |
|---|---|
| `umiusi_sim_images-20261009/` | sim 画像の生 / 10/03 フィルタ後（正解の箱つき）24 組、会場別フィルタの一覧、10/03 の実フレーム |
| `umiusi_sim_videos-20261008/` | 競技シナリオ 1 回分の動画（実機経路、3 分） |
| `umiusi_detector_overlay-20261008/` | 新検出器の検出結果を重ねた実画像 42 枚 |
| `umiusi_labels_check-20261008/` | ラベル vs 検出の比較 32 組（JAMSTEC のラベルの問題の確認用） |

## 3. 物理（sim を実機に合わせた）

- **推力・サーボの符号は 10/03 の実機と一致**。以前「sim の duty が逆」と言っていたのは解析ツールの座標変換の誤り（修正済み）。
- `tools/sim_server.py`: 姿勢・gyro・加速度を REP-103（x 前 / y 左 / z 上）で返す。サーボ指令は rad（control と同じ）。以前は deg と解釈しており、±1.5° しか動いていなかった。
- `tools/closed_loop_replay.py`: bag の `/cmd/*` → 実機の `sinsei_umiusi_control`（無改変）→ bridge → sim。
  - ヨーレートの相関 0.99、各基の推力ベクトルの相関 0.8〜0.9。
  - チューニング前（kp 1 / kd 0）の振動（サーボ ±14°、duty ±0.5 のバンバン）と、後の安定化が sim でも出る。
- 既定のプラントを変更（`configs/umiusi.yaml`）: roll/pitch の減衰 ×8、付加慣性 ×3。holdout の roll 誤差 −35〜40%。
- 水面と床を追加（opt-in。`water_surface_y` / `floor_y`）。無いと浮上し続け・沈み続けていた。
- 合っていないもの: 並進の推力の大きさ（ジャイロからは duty 0.5 で sim の約 1/3）。秤での実測が本筋（`calibration_plan.md` §3）。

## 4. 競技（FSM）

- 新しいプラントでは既定の FSM は割れない。sim で詰めた値:
  - 赤も 5° 上を狙う（`AIM_ABOVE_COLOURS` に red）
  - 探索中に下向きの heave −0.05（`SEARCH_HEAVE_BIAS`。sim の FSM には追加済み、autonomy の `fsm_params` には未追加）
- 結果（実機の AUTO 経路）: 競技場 +120 / +180、風船 1 個の赤 3/4。
- **実機で最初に確認してほしいこと**: `competition.yaml` の `yaw_rate_scale: 1.0` だと sim では風船と逆に回る。
  プールで「右前に置いた風船に右へ回るか」を見る。左なら `ros2 param set /auto_target_generator yaw_rate_scale -1.0`。
- プールでの動作確認: 「最初の 1 個を狙う」専用モードは無い。FSM は探索 → 接近 → 正対 → 突進 → 確認の繰り返しで、置いた風船を追う確認にそのまま使える。
- 既知の問題: `ki_heave` の積分が探索に入っても残り、床まで沈むことがある。

## 5. ラベル

- 画像の枚数（ラベル済みの実画像 1,736 枚。sim 4,000 枚。未ラベルの他チーム映像のフレーム 4,744 枚など）。
- JAMSTEC のラベルに水面の反射が含まれ、遠く小さい風船が抜けている。人が直す道具 `tools/label_review.py`（ブラウザで箱を直し、反射を外して COCO に書き出す）を作った。修正は作業中。

## 6. 再現・置き場所

- コード: `exp/env-filter` ブランチ（このファイルを含む PR）。
- 動かし方は各 docs の先頭。競技の動画は `MUJOCO_GL=egl uv run python -m tools.competition_ros --layout field --minutes 3 --yaw-rate-scale -1 --video out.mp4`（ROS 2 Jazzy と ros2_ws の build が要る。全条件は `competition_ros_20261008.md` §5）。
- 実機・autonomy 側への引き継ぎ（10/08）: `mujoco_ws/data/20261003-pool/HANDOFF_to_autonomy_20261008.md`。
