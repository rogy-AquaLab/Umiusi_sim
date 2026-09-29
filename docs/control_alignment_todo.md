# sim を control の新仕様へ合わせるための修正点（2026-09-29）

実機側の方針が 2026-09-29 に決まった（`sinsei_UMIUSI_autonomy/docs/autonomy_control_delta.md`）:

- **仕様は `sinsei_UMIUSI_control` に統一**する。autonomy は畳む
- **D-1**: yaw は**レート制御を既定とし、`AttitudeTarget.hold_yaw`（bool）で保持**。
  エッジでラッチ・誤差クランプ（初期値 90°）でラッチし直し・**寿命は付けない**
- **D-2**: `Target.velocity` は **正規化値 `[-1, 1]`**（m/s でも N でもない）
- **D-3**: **ソフトの浮力トリムは持たない。** 機体側のバラストで中性浮力に寄せる。
  深度センサは**未搭載**（大会までに載る可能性あり）。載れば深度を閉ループに

sim はこれと**食い違っている**。食い違ったまま回すと、sim で詰めたゲインが実機で意味を持たない
（= A-11 / B-14 と同じ「sim と実機で別物」の形）。

以下、**効く順**。S-1 と S-2 は他の全部の前提。

---

## S-1. 【最優先】浮力トリムを外した状態を既定にする

**なぜ最優先か**: トリムは単なる 1 項ではなく、**機体の動作点そのもの**を決めていた。
9/13 の実機 bag では**指令 0 でも cap の 52〜68% をトリムが消費**していた
（`sinsei_UMIUSI_autonomy/docs/known_issues.md` B-21）。**これを外すと動作点が動くので、
いま sim で詰めてあるものは全部その上に乗っている。**

- [ ] `packages/perception/src/umiusi_perception/classical.py` の `fz_trim`
      （`-self.net_buoy_up / f_max_total(...)`、`wrench()` 内）を**既定で 0 にする**
      経路を作る。**消すのではなくスイッチにする** — 「トリムありの過去の結果」と
      比べられなくなるため（`ClassicalController(..., buoy_trim=False)` 等）
- [ ] `ClassicalController` の docstring を直す。いま
      「Attitude PID + **exact buoyancy trim** + observer-corrected cruise」と、
      **トリムがあることが設計の売り**として書かれている
- [ ] 同じく `classical.py` の「EVERY CAP-DEPENDENT QUANTITY IS SOLVED」の 4 つのうち
      `buoyancy trim  fz_mode = -net_buoy_up / f_max(cap)` を落とす（3 つになる）

## S-2. プラントの浮力を実機（バラスト調整後）に合わせる

**いまの sim は実機より浮力復元が強すぎる**（CoB オフセット: 実測 2〜4 mm に対し sim 10 mm、
**DR 範囲 4〜16 mm に実機が入っていない**）。これは以前から分かっていたが、
**バラスト調整で実機が中性浮力に寄る以上、もう先送りできない。**

- [ ] CoB オフセットを **10 mm → 3 mm** 前後へ。DR 範囲も実機を含むように取り直す
- [ ] `net_buoy_up` を**バラスト調整後の実測値**に入れ替える（いま `classical_bundle.json` は
      `1.1708529300000095`。これは**調整前**の値）
- [ ] **プラントを変えるので、既存ゲインの再確認が要る**（kp 2.2 / kd 0.45 / k_ff 1.0 / k_v 1.2）
- [ ] `CONTRACT_VERSION` を上げ、バンドルを再 export（`tools/export_classical.py`）。
      **fingerprint が入るので、古いバンドルで走ると気付ける**

> ⚠ 実機のバラスト調整値が出るまで S-2 は**確定できない**。それまでは
> 「今の値のまま + トリム off」で S-1 の影響だけ先に見ておくとよい。

## S-3. yaw を「レート既定 + `hold_yaw` で保持」に合わせる

**いまの sim は yaw を常に絶対保持している。** `packages/sim/src/umiusi_rl/envs/umiusi_pose_env.py`
は `_sample_target_quat()` で**ランダムな絶対 yaw を含む `target_quat`** を作り、
`ori_err`（`mju_subQuat`）を 3 軸そのまま観測に出している（`yaw_target_deg` 既定 **180**）。
つまり **`hold_yaw = false`（実機の既定モード）が sim で一度も評価されていない。**

- [ ] `hold_yaw = false` に相当するタスクを足す — **yaw の `ori_err` を落とし、
      yaw はレート指令で駆動する**（roll/pitch は今までどおり絶対）
- [ ] `hold_yaw = true` 側は**エッジラッチを sim でも同じ規則で**作る
      （ランダムな episode 目標ではなく、**ラッチした時点の実測 yaw**）
- [ ] `packages/perception/src/umiusi_perception/autonomy/behavior.py` の FSM は
      すでに **yaw レート**を出している（`{surge, heave, yaw}`）ので、**`hold_yaw` を
      いつ立てるかを FSM 側で決める**必要がある。SEARCH の 360° スイープ中は false、
      APPROACH で狙いを定めたら true、が素直
- [ ] 誤差クランプ（90°）とラッチし直しを sim にも入れ、**S-4 の跳躍モデルで試験する**

## S-4. IMU の yaw 跳躍をモデルに入れる

**D-1 のリスクは sim から完全に見えていない。** sim の AHRS は yaw が完璧で、
実機で唯一測れている壊れ方（**yaw だけ約 180° 跳ぶ**、`known_issues.md` A-1）が起きない。
**クランプの妥当性を sim で確かめられるのはここだけ。**

- [ ] 観測に**ヨー限定の跳躍注入**を足す（重力軸まわりの回転を quat に掛ける。
      roll/pitch は無傷にすること — **実測がそうだった**）
- [ ] 実測値を既定に: **169°**、継続時間 1 サンプル（20 ms）、跳んだ先で正常に追従
- [ ] 発生率は**上限で振る**。実測は「陸上 150 秒で 1 回 / 水中 61 分で 0 回」なので
      真の率は分からない。**率を決め打ちせず、クランプが効くかどうかを見る道具として使う**
- [ ] 合否: 跳躍を入れても**ラッチし直して数百 ms で復帰すること**。
      クランプ無しでは 180° 旋回指令になることも**同時に示す**（回帰の対照）

## S-5. 深度を「センサ無し」が既定だと分かる形にする

`umiusi_pose_env.py` の観測モードに `imu_depth` / `imu_depth_dvl` があり、**完璧な深度**が入る。
実機には**深度センサが無い**（大会までに載る可能性はある）。

- [ ] **deploy に使うのは `imu` だけ**と docs に明記する（`docs/rl.md` / `docs/architecture.md`）
- [ ] `imu_depth` 系を使った結果を**実機の予測として引用しない**ようにする注記
- [ ] センサが載ったら: **完璧な深度ではなくノイズ・レート・オフセット付き**でモデルする
      （`/state/pressure` の実レートが分かってから）
- [ ] `k_v_vert` 既定 0 は**据え置き**。ただし **S-1 でトリムが消えると z チャンネルは
      「指令 0 なら出力 0」になる**ので、そのことをテストで固定する

## S-6. `velocity` の単位を正規化値として揃える

- [ ] sim 側で `v_cmd` を **m/s** として扱っている箇所（`reachable_speed` / `v_ref` で
      クランプしている経路）と、**control が受け取る正規化値**の対応を 1 箇所に書く。
      **どちらかに寄せるのではなく、変換がどこにあるかを明示する**のが要点
- [ ] `feedforward_allocation` の `thrust_curve_exp` 既定 **1.0** と、プラントの実体 **2.0**
      の食い違いを直す（`sinsei_UMIUSI_autonomy/docs/thrust_calibration.md`）。
      **この経路だけ推力の仮定がプラントと違う**
- [ ] **後進の非対称は入っていない**ことを注記する。
      `F = sign(u) * |u|^exp * thrust_per_cmd` は符号に対して対称だが、実機のペラは対称ではない

## S-7. 特異点まわりを測り直す

`classical.py` の docstring に「station holding では必要な力がほぼ**純鉛直（＝浮力トリム）**に
なるので `|phi|` の中央が 84°、59% のステップで 80° 超」とある。**この記述の前提が S-1 で消える。**

- [ ] トリム off で `|phi|` の分布を取り直す（実機 9/13 の実測は中央 47〜55°・80° 超 17〜18%
      だが、**lf 欠損の 3 基配分**なので直接は比べられない）
- [ ] `prefer_deg`（零空間を特異点回避に使う量）を**取り直した分布の上で**決め直す
- [ ] 反転頻度の評価（`reversal_check.py` 相当）をやり直す。
      **sim 1.85 回/秒 vs 実機 0.02〜0.60 回/秒**の乖離が、トリム由来だったのかが分かる

## S-8. インタフェースのパリティ

- [ ] `AttitudeTarget` に `hold_yaw` が入るので、`umiusi_sim_bridge` と
      `tools/classical_control.py` / `tools/navigator_sim.py` の経路を合わせる
- [ ] `Target.orientation` は control 側で**削除**される。sim 側で参照している箇所を洗う
- [ ] **ゴールデンベクタ**（`port_classical_to_control.md` の案 (A)）— sim の Python と
      control の C++ が同じ入力で同じ出力を出すことを縛る。**言語が 2 本になるなら必須**

---

## やらないこと

- **RL 系は触らない**（`feat/rl-attitude` 系は封印中）。S-3 / S-4 は古典側だけで足りる
- **深度閉ループの実装**はセンサが載ってから。載る前に sim で先行実装しない
  （実機で検証できない経路を大会直前に増やさない）
- 欠損スラスタ（1 基落ち）の作り込み — lf が直って 4 基運用に戻ったので当面の律速ではない

## 未検証の前提（この文書が乗っているもの）

- **バラスト調整後の実機の浮力は未測定。** S-2 の数値は調整が終わるまで確定しない
- **yaw 跳躍の真の発生率は不明**（陸上 n=1 / 水中 0 件）。S-4 は率ではなく
  **クランプが効くかどうか**を見る道具として設計すること
- **`dev-0921` は同梱パラメータのままだと推力が出ない**（`known_issues.md` B-20）。
  実機で突き合わせる前にこれを踏むこと
