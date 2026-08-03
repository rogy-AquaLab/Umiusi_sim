# umiusi_sim — body-dynamics RL (本体挙動)

**低レベル制御の学習**トラック。自己受容感覚(固有受容) + 姿勢誤差を 8 次元のスラスタ行動に写像する PPO ポリシーを、解析的ハイドロの MuJoCo シム上でオフライン学習する(力モデルは [`physics.md`](physics.md) を参照)。出力されるポリシーは姿勢保持、姿勢+深度、巡航しながらの保持を行う。

> これはバルーンミッションとは**別のトラック**である。コンペティションの [autonomy](autonomy.md) は RL ではなくフィードフォワードである。RL ポリシーをハードウェア上で動かす場合(将来)は、検出器と同じパターンに従う。すなわち Python で学習 + エクスポートし、薄い `umiusi_perception.policy` ランタイム経由でロードする(`architecture.md:86`)。

## 配置場所

| piece | file |
|---|---|
| 学習エントリ(PPO/SAC/TD3、カリキュラム、VecNormalize) | `packages/sim/src/umiusi_rl/train.py` |
| 評価 / メトリクス / 動画 | `packages/sim/src/umiusi_rl/eval.py` |
| Gym env (`UmiusiPoseEnv`) | `packages/sim/src/umiusi_rl/envs/umiusi_pose_env.py` |
| MuJoCo ラッパ (`UmiusiSimulator`) | `packages/sim/src/umiusi_sim/simulator.py` |
| MJCF モデル | `packages/sim/src/umiusi_sim/description/umiusi.xml` |
| 物理(浮力/抗力/揚力/推力) | `.../physics/hydrodynamics.py`, `physics/thruster.py` |
| RL 設定 | `configs/train_ppo.yaml` · シムパラメータ `configs/umiusi.yaml` |
| 検証ゲート | `tools/validate_sim.py` · 定常状態診断 `tools/analyze_steady.py` |
| キーボード操作 / ROS 操作 | `tools/drive.py` · `tools/ros_policy.py` |

スタック: stable-baselines3 (2.9.0)、gymnasium (1.3.0)、torch 2.12 CPU、mujoco 3.10。**CPU 専用**で、`--n-envs` (`SubprocVecEnv`) でスケールする。

---

## 1. アルゴリズム

**PPO** がデフォルトであり、実際に使われている唯一のアルゴリズムである(学習済みの各 `meta.yaml` はすべて `algo: ppo`)。`train.py:28` では `SAC` と `TD3`(`--algo` スイッチ)も配線されているが、これらは SB3 のデフォルト + 共有ネットワークアーキテクチャにフォールバックする。ポリシー = `MlpPolicy`、アクターとクリティックの **net_arch `[256, 256]`**。

**PPO ハイパーパラメータ**(`configs/train_ppo.yaml:105`): `n_steps=2048`、`batch_size=512`、`n_epochs=10`、`gamma=0.99`、`gae_lambda=0.95`、`ent_coef=0.01`(巡航推力パターンを見つけるため探索を高く保つ)、`learning_rate=3e-4`、`clip_range=0.2`。デフォルト `n_envs=8`、`seed=0`、`total_timesteps=300k`(実行ごとにオーバーライド)。

**VecNormalize**(`train.py:119`): `norm_obs=True`、`norm_reward=True`(PPO)、`clip_obs=10.0`。統計は `vecnormalize.pkl` に保存され、評価時に再ロードされて推論が学習と一致するようにする。すべての利用側(`eval.py`, `drive.py`, `ros_policy.py`, `analyze_steady.py`)は `model.predict(deterministic=True)` の前に `clip((o−mean)/√(var+eps), ±10)` を再構成する。

**カリキュラム**(`CurriculumCallback`, `train.py:31`)— `attitude_velocity` のみ。学習の最初の `frac=0.5` にわたって `vel_cmd_cone_deg`、`yaw_target_deg`、`tilt_target_deg` を 0 → 設定目標値へ線形に広げる。まず直進巡航を学習し、その後に一般化する。「何もしない」局所最適を回避するためである。

---

## 2. 環境 — `UmiusiPoseEnv`

**500 Hz** のシム上で **50 Hz 制御**を行う `gym.Env`(`umiusi_pose_env.py:68`)(制御ステップあたり 10 物理サブステップ、ステップ内でサーボ/ESC のスルーを含む)。

**行動**(`spaces.Box(-1,1,(8,))`): `[servo_1..4, esc_1..4]`。`servo_k·90°` = 目標方位角(スルー 250°/s)、`esc_k·30 N` = 推力(スルー 4.0 units/s)。

**観測** — 常に **16 次元の自己受容感覚(固有受容)** = `servo_n(4) + thrust_n(4) + prev_action(8)`、加えて `obs_mode` による外界受容:

| obs_mode | extero contents | total |
|---|---|---|
| `imu` | `ori_err(3)` + `w_body(3)` | **22** |
| `imu_depth` | + `depth_err(1)` | **23** |
| `imu_depth_dvl` | + `depth_err(1)` + `lin_vel_body(3)` | **26** |
| `full` | `pos_err(3)` + `ori_err(3)` + `lin_vel(3)` + `w_body(3)` | **28** |

`ori_err` = 回転ベクトル誤差 `mju_subQuat(target, current)`(AHRS/BNO055、磁方位を含む絶対値)、`w_body` = ボディフレームのジャイロ。`attitude_velocity` は **3 次元のフィードフォワード速度*コマンド***(観測であり計測値ではない)を付加し → `imu` は **25 次元**になる。水平 X,Z は `full` でのみ観測可能(水中では GPS なし)。

**タスク**(`--task`)、それぞれ現実的なセンサ構成に対応(報酬/成功は常に*真の*状態を使う。限定的な観測はタスクの一部を観測不能にするだけである):

| task | goal | default obs | note |
|---|---|---|---|
| `attitude` | ランダムな目標向きを追従 | `imu` | 水平・深度がドリフト(観測されない) |
| `attitude_depth` | 向き + ランダムな深度 | `imu_depth` | 水平がドリフト |
| `attitude_velocity` | 向きを保持 + 指令された**方向**に巡航 | `imu` (25-D) | 方向のみ(DVL なしでは速度が観測不能) |
| `pose` | go-to-pose: ランダムな位置、直立 | `full` | 位置基準が必要 |

**エピソード**: ホライズン **600 制御ステップ = 12 s**、開始位置は ±0.10 m のジッタ付き。`pose` のみ早期に終了する(範囲外 `|pos|>[2.0,1.5,2.0]`)。姿勢タスクは車体をドリフトさせ、ホライズンで打ち切る。

**報酬**(`umiusi_pose_env.py:312`、重みは `train_ppo.yaml:39`)= 密なペナルティ + `exp(−(err/scale)²)` の近接ボーナス:
- **tracking**: `−w_ori·ori_err`(1.0、0.05 rad の不感帯)+ `w_ori_bonus 3.0`。pose は `−w_pos` + ボーナスを追加、depth は `−w_depth 2.0` + ボーナスを追加、cruise は `+w_vel_dir 10.0·v_along`(希望値で頭打ち)と `−w_vel_perp 4.0·v_perp`(横方向ドリフト)を追加。
- **economy / smoothness**(サーボが固まって追従を壊さないよう*軽く*保つ、「att_v3 の教訓」): `w_effort 0.03`、`w_servo_rate 0.35`、`w_thrust_rate 0.15`、`w_action_rate 0.02`。
- ゴール付近での**整定**(`ori_err<0.15`): 保持タスクはサーボ + 推力を停止方向へ押し込む(`w_settle_servo 0.80`、`w_settle_thrust 0.40`)。巡航はサーボのみを減衰(`w_settle_servo_cruise 0.60`)。
- 許容範囲内の間、ステップごとに `goal_bonus 5.0`。`out_of_bounds_penalty 50`(pose)。

**成功**(`:353`): `ori_err<0.20 rad`、かつ(pose)`pos<0.15 m`、(depth)`|depth_err|<0.10 m`、(cruise)`v_along>0.7·v_cmd` かつ `v_perp<0.10 m/s`。

---

## 3. MuJoCo シミュレータと物理

`UmiusiSimulator`(`simulator.py`)は MJCF `umiusi.xml` を駆動する: CAD フレーム **+Y up**、`implicitfast` 積分器、`timestep 0.002`。`base_link` は計測された慣性を持つ `freejoint`(**9.964 kg**、完全な慣性が与えられている)。4 つのスラスタボディはそれぞれ **hinge** `servo_k`(±90°、軸 = 45° 外側の取り付けアーム)、約 0.63 kg/unit を持ち、先端サイトで推力が作用する。4 つの位置アクチュエータ(`kp=40, kv=2.0`、forcerange ±1 N·m = HS-646WP のストールトルク)。衝突はオフ。車体は自由に浮遊し、スラスタ + 解析的ハイドロのみで駆動される。総質量 ≈ **12.48 kg**、`displaced_volume 0.0126 m³` → 弱い正の浮力。

力(ステップごと、`mj_applyFT` により真の作用点で。詳細は [`physics.md`](physics.md)):

1. **浮力** `−ρVg`、システム CoM より 0.05 m 上の CoB に作用 → 受動的な**復元モーメント**。
2. **抗力** 対角の線形+二次、ボディフレーム、**水流に対する** CoM 速度に作用。並進力は**center-of-pressure**オフセットに適用 → Munk 的な並進モーメント。
3. **揚力** 斜め流れに⟂(`coef 12.0`、迎角、有効。純粋なサージ/ヒーブ/スウェイではゼロ)。
4. **スラスタ** 各先端でサーボ回転した軸に沿った力。サーボが推力を水平↔垂直に傾ける。
5. **サーボ/ESC スルー** 一次のレート制限(250°/s、4.0 units/s)。
6. デフォルトでオフ: 付加質量、非対角カップリング。

`tools/validate_sim.py` は定性的な不変条件(浮遊/自己水平化、6 軸すべてで抗力が速度に対抗、ヒーブ/サージのデカップリング、有界な開ループ、NaN なし)をゲートし、較正値(終端垂直速度、中立体積、DR 帯域)を出力する。

---

## 4. Sim-to-real: ドメインランダム化と外乱

どちらも**デフォルト OFF** で、`--domain-rand` / `--disturb` で有効化し、`meta.yaml` に記録されるので評価が学習条件を再現する。

- **ドメインランダム化**(リセットごとに適用): `buoyancy_frac 0.05`、`thrust_frac 0.05`、`drag_frac 0.10`(乗算的)、`obs_noise 0.005`(ガウシアン)、`action_latency_steps 1`(~20 ms の制御→アクチュエーション遅延)。*教訓:* 中程度の DR(0.10/0.10/0.20)は巡航方向を悪化させた(ドリフト **0.04 → 0.11 m/s**、成功 **100 % → 60 %**)ため、出荷される fracs は意図的に軽くしてある。
- **外乱**: エピソードごとの定常**水流** `U(0, 0.30) m/s`(抗力は `v − current` を使い、車体を引きずる)+ ランダムな**力インパルス**(`prob 0.01`/step、`25 N`、5 steps)。

---

## 5. 学習済みモデルファミリ(`models/`)

すべて PPO、seed 0、`obs_mode imu`、設定 `train_ppo.yaml`、VecNormalize 有効。`meta.yaml` は設定/フラグのみを保存する(メトリクスは焼き込まれない)。2 つのファミリ:

**`att_v*` — タスク `attitude`**(姿勢保持、AHRS のみ、22 次元)。タイムステップは反復を通じて増加した(300k → `att_v6` 1.0M、`att_v2` 1.5M)。これは**サーボチャタリング / 定常状態振動**の低減努力を追跡している(`tools/analyze_steady.py` で診断。設定の「att_v3 の教訓」= economy ペナルティを軽く保つ)。

**`av_* ` — タスク `attitude_velocity`**(保持 + 方向巡航、25 次元)。固定 +X ベースライン(`av_fix`、cone=0)から、徐々に長くなるカリキュラム実行(`av_curr4` **2.5M**、`av_long2` **6M**)を経て進展し、加えて外乱/DR を追加した **sim2real** 実行: `av_dist`(水流)、`av_robust`(水流+DR)、`av_sim2real` / `av_lightdr`(DR)。**`examples/cruise_policy/` は `av_curr4` のコピー**であり、出荷されるデモポリシーである。

`ppo_v1` = 最初期のベースライン(300k、デフォルト `pose`)。

---

## 6. 評価と性能

`eval.py` は `meta.yaml` からタスク/観測/DR/外乱を再ロードし、VecNormalize を適用して次を報告する: 平均リターン、最終 ori/pos/depth 誤差、コマンド方向への巡航速度 + 横方向ドリフト、**保持割合**(定点保持の質)、成功 %、推力使用量、**平均角速度(ふらつき)**、サーボ動作(振動)、推力変化。トグル: `--no-disturb`(ポリシー自身の安定性を分離)、`--domain-rand`(モデルミスマッチのストレス)、`--legacy-hydro`(旧来の対角抗力モデルに対する A/B)、`--record`(トラッキングカメラ経由の mp4、`MUJOCO_GL=egl` でヘッドレス)。`tools/analyze_steady.py` はエピソード末尾にわたってエピソードを `VIBRATING` か `settled` に分類する。

### 引用値

| metric | value | source |
|---|---|---|
| 巡航ポリシー(av_curr4, 2.5M)の巡航タスク成功 | **~100 %**(一部サーボ動作あり) | `examples/README.md:5` |
| 中程度 DR の悪化 | drift 0.04 → 0.11 m/s, success 100 % → 60 % | `train_ppo.yaml:76` |
| 定常巡航の角ジッタ | **~0.3 rad/s**(削減には `w_angvel≈1.5` + 行動ローパスが必要) | `train_ppo.yaml:43` |
| 成功許容値 | ori 0.20 rad · pos 0.15 m · depth 0.10 m · vel 0.10 m/s | `train_ppo.yaml:29` |

> 上記を超える実測の成功/ドリフト/ふらつきの数値はリポジトリにコミットされていない。学習済みの `meta.yaml` ファイルは設定のみを保持する。生成するには:
> `python -m umiusi_rl.eval --model models/<run>/final.zip --episodes N`(+ `analyze_steady.py`、`validate_sim.py -v`)。

**デプロイ経路**: `tools/drive.py` はプレーンなシーンでポリシーをキーボード操作する(セルフテストは cmd·計測の方向コサインを報告する)。`tools/ros_policy.py` は ROS2 の `umiusi_sim_bridge` MuJoCo シムを rosbridge 経由で駆動し、ライブの `ImuState` + `ThrusterStateAll` から正確な 25 次元観測を再構成し、4 つの直接オーバーライド `ThrusterOutput` コマンドを発行する。どちらも `UmiusiPoseEnv._get_obs` + VecNormalize 統計を再利用するので、観測レイアウトがずれることはない。
