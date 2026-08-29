# UMIUSI simulator — physics (hydrodynamics + thruster propulsion)

解析モデルが**流体力の力**（浮力 + 抗力）と**スラスタ推進**をどのように計算するか。実装は `packages/sim/src/umiusi_sim/` にある。解析モデルは*リファレンス*となる物理計算であり（明示的で、可読で、調整可能）、MuJoCo は計算結果の力を積分する。パラメータは `configs/umiusi.yaml` に置かれ、力の計算コードは `simulator.py::_apply_external_forces` で、効果ごとの関数は `physics/hydrodynamics.py` と `physics/thruster.py` にある。

## Conventions
- **Frame:** CAD フレーム、**+Y が上**。4 つのスラスタは X–Z 平面上にある。前方は +X。
- **Body 6-vectors:** `[x, y, z, roll, pitch, yaw] = [linear(3), angular(3)]`、body 軸。
- **Integration:** 物理タイムステップ `0.002 s`（500 Hz）、制御/コマンドレート `50 Hz`。重力は
  MuJoCo が body 質量を通じて適用する。ここで解析的に加えるのは**浮力、抗力、付加質量（有効）、揚力 + CoP モーメント**である。
- **Application (important):** すべての外力は `mj_applyFT` を介してその**真の作用点**に適用され（`qfrc_applied` に累積され、各ステップでクリアされる）、base の CoM にまとめて載せることはしない。
  `base_link` の CoM は車体全体の CoM からオフセットしているため、誤った点に合力を適用すると偽のトルクが注入される（さらに、速度依存の抗力については暴走フィードバックとなる）。

## Mass properties (`configs/umiusi.yaml`)
| | value |
|---|---|
| hull (`base_link`) mass | 9.9639 kg, CoM `[-0.2411, 0.0934, 0.0003]` m, 完全な慣性テンソルを与える |
| thruster mass | 0.6292 kg × 4 |
| **total mass** | ≈ **12.48 kg** |
| neutral-buoyancy volume | 12.48 kg / 1000 kg/m³ = **0.01248 m³** |

---

## 1. Buoyancy (浮力)
`physics/hydrodynamics.py::buoyancy_force_world`

重力に対抗する world フレームの浮力：

```
F_buoy = − ρ · V · g          (world)     magnitude = ρ·V·|g|
```

- `ρ = 1000 kg/m³`（`water.density`）、`V = 0.0126 m³`（`water.displaced_volume`）、`g = [0,−9.81,0]`。
- `V` は**排除（外殻）体積**であり、中立の 0.01248 m³ より**わずかに大きく**設定して、車体をわずかに正の浮力にする（アイドル時にゆっくり浮上する）。バリデーションで調整する。

**Point of application — center of buoyancy (CoB):** 車体全体の CoM（base + 4 スラスタ）の**水平方向の真上**に置き、その `buoyancy_offset_above_com = 0.010 m` **上**に配置する（2026-08-21 の実機 bag 較正で 0.05 m から置換。`docs/calibration_plan.md` 参照）。base の body フレームで表現されるため、**船体とともに回転する**：

```
cob_local = (system CoM in base frame) + [0, 0.010, 0]
cob_world = xpos[base] + R · cob_local
```

浮力の作用線が CoM より*上*に位置するため、いかなる傾きも**受動的な復元モーメント**（自己水平化）を生む。CoB を（base の CoM ではなく）*system* の CoM と水平方向に揃えることで、偽の定常偶力を回避する。

---

## 2. Hydrodynamic drag / damping (流体抵抗)
`physics/hydrodynamics.py::drag_wrench_body`

**body フレーム**での対角の線形 + 2 次の減衰レンチであり、6 軸それぞれについて成分ごとに計算する：

```
W_drag = − ( D_lin · v  +  D_quad · |v| · v )     (body, length-6 [force(3), torque(3)])
```

ここで（`configs/umiusi.yaml::drag`、body 軸 `[x, y, z, roll, pitch, yaw]`）：

| | x | y | z | roll | pitch | yaw |
|---|---|---|---|---|---|---|
| `D_lin`  (N/(m/s), N·m/(rad/s))   | 15 | 25 | 20 | 3 | 3 | 3 |
| `D_quad` (N/(m/s)², N·m/(rad/s)²) | 40 | 80 | 60 | 5 | 5 | 5 |

（2026-08-21 更新: PLACEHOLDER から **CAD シルエット面積 × BlueROV2 実測 Cd による推定値**へ置換。導出は `python -m tools.estimate_hydro`、誤差帯 ±30 %、実測較正の手順は `docs/calibration_plan.md`。）

**Velocity used (important subtlety):** 線形項は `mj_objectVelocity` による body 原点の速度ではなく、**CoM の並進速度**（`mj_subtreeVel → subtree_linvel`）を使う。body 原点は CoM からオフセットしているため、回転があると大きな `ω × r` 項が注入され、スピンが抗力の*力*へ連成して数値的暴走を引き起こす。角度項は**純モーメント**として適用する。

**Water current / disturbance:** 速度は**水に対する相対速度**として取る。`v_lin_body = Rᵀ (subtree_linvel − current_world)` であり、水流が車体を引きずる（RL では外乱として使用）。レンチは `mj_applyFT` を介して **system CoM** に適用される。

**Added mass:** `physics/hydrodynamics.py::added_mass_wrench_body` は `−M_A · a`（対角）を与える。2026-08-21 から **有効**（`diag = [2.69, 7.10, 3.16, 0.082, 0.165, 0.122]`、Lamb 楕円体 × ソリディティを BlueROV2 公表値で較正した推定。`tools/estimate_hydro` 参照）。ヒーブ付加質量は機体質量の 57 % あり、0（旧値）では鉛直・ピッチの過渡応答を大きく誤る。1 サブステップ遅れの有限差分加速度による明示的適用だが、validate_sim の open_loop / バンバン指令 30 s で発散しないことを確認済み。

### 2b. Higher-fidelity hydro: lift, translation-induced moment, cross-coupling
3 つの config でゲートされる項が、「各軸で速度に対抗する力」を超えて対角の抗力を拡張する。これらの係数は **PLACEHOLDER**（物理的に妥当なもので、tow/PMM 試験に対してキャリブレーションする予定）であり、それぞれ旧モデルを復元するデフォルトを持ち、**lift + CoP オフセットはデフォルトで ENABLED**（控えめ）である。

**Lift** — `physics/hydrodynamics.py::lift_force_body`、CoM で純力として適用される。body フレームの並進速度に*垂直*な力であり、流れと縦方向の基準軸（`lift.ref_axis = +X`、流線型/細長い軸）との間の**迎角** `α` とともに増大し、大きさは `∝ |v|²`：

```
F_lift = coef · |v|² · sin(α)·cos(α) · n̂        (body)   coef = ½ρ·Cl·A  (lumped)
       = coef · |v|² · (v̂·ê_ref) · (ê_ref − (v̂·ê_ref) v̂)
```

`n̂` は（`v`, `ê_ref`）平面内で `v` に⟂な単位ベクトルであり、`ê_ref` の向きを指す。`sin·cos`（= ½ sin 2α）という形状により、揚力は**基準軸に沿った流れ（α=0）とそれに垂直な流れ（α=90°）の両方で消失する**。したがって純サージ / 純ヒーブ / 純スウェイでは**揚力ゼロ**となり、真に**斜め**な流れのみが揚力を生む。これが、軸に沿った `validate_sim` の不変条件（サージは水平を保ち、ヒーブはスピンせずに上昇する）が依然として成立する理由である。`lift.coef = 12.0` N/(m/s)² → 0.4 m/s の巡航時で最大約 1 N（控えめ）。`coef = 0` = OFF。

**Translation-induced moment via a center-of-pressure (CoP) offset** — （線形+2 次の）並進抗力は、CoM ちょうどではなく `CoP = system CoM + R·cop_offset`（body フレーム）に適用され、**横方向の並進が旋回/復元モーメントを誘起する**（Munk 的）。モーメントアーム `(CoP−CoM) × F_drag` が偶力となる。`cop_offset = [0.03, 0, 0]` m（前方に数 cm）。`[0,0,0]` で旧モデル（抗力は CoM を通り、並進モーメントはゼロ）を復元する。角度減衰モーメントは位置に依存しない自由ベクトルのままなので、アームを得るのは*並進*抗力のみである。

**Off-diagonal (cross-axis) damping** — `physics/hydrodynamics.py::coupling_moment_body`、オプションで**デフォルト OFF**。並進速度に連成する body モーメント `M += −(c_lin·v + c_quad·|v|·v)` を加える。`coupling.sway_yaw`（body-Z のスウェイ → +Y 周りのヨー、風見効果）および `coupling.heave_pitch`（body-Y のヒーブ → +Z 周りのピッチ）。CoP オフセットが既に並進→モーメント連成を供給しているためデフォルトで 0 のままにしているが、特定のクロス項の調整が必要な場合に利用できる。

---

## 3. Thruster propulsion (スラスタ推進力)
`physics/thruster.py` + `simulator.py::step / _apply_external_forces`

4 つの**アジマススラスタ**（T 字型ダクトプロペラ）。各ユニットは**サーボ**（アジマス角）と **ESC**（推力の大きさ）を持つ。サーボはユニット全体をその**取り付けアーム**（約 0.15 m、45° 外向き）周りに回転させる。**推力軸はアームに垂直**であるため、サーボは推力を**水平接線方向**（servo = 0）と**垂直/上向き**（servo = +90°）の間で傾ける。

### Action → servo & thrust (`step`)
8 次元のアクションは `[servo_1..4, esc_1..4]` であり、それぞれ `[−1, 1]` にクランプされる：

```
servo_target_k = clip(a_servo_k, −1, 1) · servo_range          # servo_range = 90° = 1.5708 rad
servo_ctrl_k   = track(servo_ctrl_k → servo_target_k, 250°/s, τ=0.05 s, dt)  # slew + 1次遅れ (HS-646WP)
esc_current_k  = slew(esc_current_k → clip(a_esc_k,−1,1), 4.0 /s, dt)   # rate-limited ESC command
thrust_mag_k   = sign(u)·|u|^thrust_curve_exp · thrust_per_cmd   # u = esc_current_k, exp = 2.0, thrust_per_cmd = 30 N
```

- サーボは `track(x→target, rate, τ, dt)`: `rate = clip(err/τ, ±slew)` — 大角度ではレート制限、
  目標近傍（slew·τ ≈ 12.5°以内）では時定数 τ の 1 次遅れで**指令に収束する**（実 RC サーボの位置制御ループを近似）。
  旧 pure-slew はフラッピング指令下で永遠に収束せず、「指令が到達不能」という病理を学習から隠していた
  （issue #2/#3）。`servo_tau_s: 0` で旧モデルに戻る。ESC は従来どおり `slew`（`thrust_slew = 4.0 esc-units/s`）。
- 推力マップは**プロペラ則**である: `F = sign(u)·|u|^thrust_curve_exp · thrust_per_cmd`（`thrust_curve_exp = 2.0`）。
  2026-08-21 の bag 再生フィットで線形マップから置換した（低 duty の実効推力が線形では過大だった。
  `thrust_curve_exp: 1.0` で旧来の線形マップに戻る）。bag は |duty| ≤ 0.2 しか含まないため、指数と最大推力
  `thrust_per_cmd` の分離は秤の実測（`calibration_plan.md` §3）待ちである。
- `servo_ctrl_k` は MJCF のサーボアクチュエータ（`data.ctrl`、ラジアン）に書き込まれる。body がどう回転するかについては MJCF のヒンジ（アーム周り）が権威を持つ。サーボアクチュエータの力はストールトルク（約 0.94〜1.14 N·m）に制限される。

### Thrust force (`_apply_external_forces`)
各スラスタについて、推力はその**（サーボで回転した）ローカル軸**に沿って作用し、先端のサイトに適用される：

```
F_thr_k(world) = thrust_mag_k · ( R_body_k · thrust_axis_local_k )
mj_applyFT(F_thr_k, point = site_xpos[t k _thrust], body = thruster_k)
```

- `thrust_axis_local_k` は**中立（servo = 0）の推力方向** = 水平接線方向（アームに垂直）であり、スラスタの body フレームで表される。サーボがスラスタ body を回転させると `R_body_k` がこの軸を傾けるため、同じ格納軸で正しく傾いた推力が生成される。
- ユニットごとのピボット + 中立軸は `thrusters.units`（id → 名前 `1=lf 2=lb 3=rf 4=rb`、**+Z = 右舷**）にある。例えばユニット 1 は
  `thrust_axis ≈ [0.708, 0, 0.706]`。これは *geometry* の命名であり、**アクションのチャネル順は別契約**である
  （`action_order: [lf, lb, rb, rf]` — autonomy 側の POSITIONS と一致させるためのもの）。
- 正味の効果：サーボが推力を**水平**成分（→ サージ / スウェイ / ヨー）と**垂直**成分（→ ヒーブ / ロール / ピッチ）に分割する。`φ = atan(f_vertical / f_horizontal)` であり、実機の `sinsei_umiusi_control` の FeedForward 配分に一致する。

### Modeling scope & remaining limitations
中核の減衰は**対角 2 次減衰**モデル（Fossen/Morison の 1 次近似）であり、いまや §2b の各項（揚力、CoP オフセットによる並進モーメント、オプションのクロス連成）で拡張されている。§2b 以降：
- **Lift IS modeled**（流れに⟂な迎角の力。§2b）— 控えめ、ENABLED、placeholder の `Cl·A`。
- **Translation-induced moments ARE modeled** — CoP オフセット（§2b）を介する。横方向の並進はいまや
  Munk 的な旋回/復元モーメントを誘起する。ENABLED、小さな（3 cm）placeholder オフセット付き。
- **Off-diagonal damping**（sway→yaw、heave→pitch）は利用可能だが**デフォルト OFF**（`coupling.* = 0`）。

残る単純化：
- **Direction dependence = constant per-axis coefficients** — *抗力*の大きさについて。方向ごとに異なる実効
  `Cd·A` は、body 軸ごとに異なる `D_lin`/`D_quad`（流線型の +X < 横方向の +Y）に織り込まれている。任意の流れ角における投影面積を動的に再計算するわけではない。
- **Lift shape is a simple ½sin 2α**（明示的なストール/`Cl(α)` 曲線なし）。CoP オフセットは**固定**点（速度・角度に依存しない）。
- **Added mass は対角のみ**（交差項なし。対角は推定値で有効）。
- §2b のうち **lift `coef` / `cop_offset` / coupling** はキャリブレーション（tow/PMM 試験）待ちの **PLACEHOLDER** である
  （同じ §2 でも drag 係数と付加質量は `tools/estimate_hydro` 由来の推定値、buoyancy オフセットは実機 bag で較正済み）。

これは、定点保持 + 低速巡航を行う低速の箱型 ROV には十分であり、§2b の各項が斜め / 横方向の機動に対して 1 次の揚力/並進モーメントの忠実度を追加する。さらなる高度化としては、完全な 6×6 減衰行列と角度分解された `Cl(α)` / 移動する CoP が挙げられる。

---

## 4. Feed-forward allocation (command → per-thruster) — control side
`control.py`（`sinsei_umiusi_control` の FeedForward の移植）

§3 の逆問題：所望の body レンチが与えられると、解析的な **8×6 配分行列**がそれを 4 つの
`(servo_angle, esc)` ペアへ写像する。`competition_run` で使われ、ROS コントローラからも利用可能である。
**Frame mapping note:** FF 軸 ≠ sim 軸（`Vx → sim −X`、`Vz → +Y`、`Vy → +Z sway`。旧記述の
「yaw couple」は右舷 2 基の行入れ替え移植バグの症状で 2026-08-26 修正済み）。これは
`control.py` に文書化されており、`validate_sim.check_ff_allocation`（純ヒーブはスピンせず上昇し、純サージは水平を保つ）で守られる。FF フレームの符号調整はトラッキング中のフォローアップ項目である。

---

## Summary of the per-step force pipeline (`_apply_external_forces`)
1. `qfrc_applied` をクリアする。
2. **buoyancy** `−ρV g` を CoB に（CoM より上 → 復元モーメント）。
3. **drag** `−(D_lin v + D_quad |v| v)`（+ オプションのクロス連成モーメント）を、水流に対する相対の CoM 速度を使って計算し、並進力は **CoP**（`CoM + R·cop_offset` → 並進モーメント）に、角度減衰は純モーメントとして適用する。
4. **lift** `coef·|v|²·½sin 2α · n̂` を流れに⟂に、CoM で（軸に沿った流れではゼロ）。
5. **added mass**（デフォルト **ON**、対角 `[2.69, 7.10, 3.16, 0.082, 0.165, 0.122]` の推定値）+ 外部の **impulse** 外乱を CoM で。
6. **thrust** `mag · R·axis` を各スラスタ先端サイトで。
7. MuJoCo が積分する（重力は body 質量経由）、500 Hz。

残る PLACEHOLDER は **lift `coef`、`cop_offset`、coupling、`displaced_volume`、`thrust_per_cmd` の絶対値**である（drag 係数は `tools/estimate_hydro` の推定値、buoyancy オフセットと推力曲線の指数は 2026-08-21 の実機 bag で**較正済み**）。残りは `docs/calibration_plan.md` の水中実験で実測に置き換える。`tools/validate_sim.py` が定性的な不変条件（浮く/水平になる、heave/surge の分離、NaN なし）を守る。
