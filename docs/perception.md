# umiusi_sim — perception (検出器 · 水中復元 · トラッカ)

劣化した水中カメラフレームを、方位角 + 距離を持つ安定した色ラベル付きバルーン**検出結果**へと変換するビジョンスタックの仕組みと、その背後にあるモデルについて。これは**ロボット搭載側**の半分である。ROS 非依存 / sim 非依存のホイール `umiusi_perception`（`packages/perception`）に置かれており、そのため検出器 + トラッカは sim（`tools/autonomy_run.py`）とロボット（`ros2_ws/src/umiusi_autonomy`）とで**ビット単位で同一**である。下流の挙動については [`autonomy.md`](autonomy.md) に別途記載する。

## 配置場所

| piece | file | role |
|---|---|---|
| Detection データクラス、古典的 HSV 検出器 | `packages/perception/src/umiusi_perception/balloon_detector.py` | 色ブロブ → 検出結果 |
| 学習済み tiny-CNN（**出荷パス**） | `.../learned_detector.py` | `TinyBalloonNet` CenterNet-lite |
| Hough 円形状検出器（実験） | `.../hough_detector.py` | 円の復元/確定 |
| 水中色補正 | `.../underwater.py` | 決定論的 RGB 復元器 |
| マルチフレームトラッカ + サニタイザ | `.../tracker.py` | 対応付け → 確定 → 持続 |
| IoU / PRF 評価ハーネス | `.../eval.py` | precision/recall/F1 |
| 学習 | `tools/perception_train.py`, `tools/gen_sim_dataset.py`, `tools/perception_pseudolabel.py` | CenterNet 学習、合成データ、擬似ラベル |
| レイテンシベンチ（Pi-4 予測） | `tools/perception_bench.py` | fps / GFLOPs / int8 ONNX |
| 制御分離診断 | `tools/ram_eval.py` | 完全知覚 → FSM |

パッケージ依存（`packages/perception/pyproject.toml`）: `numpy>=1.24, torch>=2.2, scipy>=1.10,
opencv-python-headless>=4.8`。`umiusi_sim` / `umiusi_rl` からは何もインポートしない。

## 共通出力 — `Detection`

3 つの検出器ファミリはすべて**同一**の `Detection` データクラス（`balloon_detector.py:107`）を出力するため、評価ハーネスおよび FSM 内でドロップイン交換可能である:

```
colour, points, bbox(u0,v0,u1,v1), centroid, area_px,
bearing(az,el) [rad], range_m, confidence, mean_s, mean_v, is_reflection
```

- **幾何**（`_pinhole`, `balloon_detector.py:161`）: 正方ピクセルのピンホール、`fy = (H/2)/tan(fovy/2)`、
  `fx = fy`、デフォルト `fovy = 60°`。見かけサイズからの距離:
  `range_m = BALLOON_DIAMETER_M · fx / apparent_pixel_diameter`、ただし `BALLOON_DIAMETER_M = 0.20 m`
  （半径 0.10 m の球）。色ごとのポイント: **red +30, yellow +10, blue −10**（`balloon_detector.py:100`）。

#### バルーンサイズ仮定の前提と頑健性
`BALLOON_DIAMETER_M = 0.20 m` は**単眼距離推定のためだけ**の仮定であり、実物の直径が厳密に指定されない・多少
ばらつく（大小・非真球）ことは、以下の理由で**軽度なら問題にならない**（設計上、距離は最も粗い量として扱う）:

- **検出（見つける/色付け）はサイズ非依存**: `TinyBalloonNet` は物理直径を仮定せず `(w,h)` を回帰するだけ。
  大小の球は「距離違い」に見えるだけで、強いスケール拡張（Affine 0.7–1.3, RandomResizedCrop, Downscale, §4）で
  学習済み → ±20–30% 程度は学習分布内。
- **距離推定だけが直径に線形依存**: 真の直径が係数 k ずれると `range_m` も k 倍ずれる。だが `range_m` の用途は
  **粗い判断のみ**（`MAX_TARGET_RANGE=4.5 m` 到達ゲート、最近傍の目標選択、`AVOID_RANGE=1.6 m` 回避、
  `PREEMPT_MARGIN=0.30 m`）。系は元々距離を最ノイジー量として扱う（`RANGE_EMA=0.4`、`ASSOC_RANGE_FRAC=2.0`＝
  フレーム間2倍の振れを許容）ので、系統的な±20–30%誤差は既に許容済みのノイズより小さい。
- **精密な終盤（ALIGN/RAM/CONFIRM/pop）は物理直径を使わない**: すべて見かけの `bbox_frac`・方位角センタリング・
  仰角の深度優先ゲートで駆動（`behavior.py:468,505,523`、いずれも画像から直接計測）。成功判定はカメラからの
  **消失**（`hide_balloon`／CONFIRM）。方位角・仰角は角度なのでサイズと独立。大きい球は「少し遠い真距離で
  RAM に入る」だけで、直進ラム＋消失確認の閉ループは収束する。
- **サイズ整合ゲートはサイズフィルタではない**: `size_consistent`（`tracker.py:227`, `SIZE_TOL=0.6`）は、現行検出器
  では距離を見かけ径から導くため比が**構造的に約1.0**（コード注釈明記）。弾くのはサブピクセル箱（`MIN_BBOX_PX=6`）・
  不可能な近距離（`MIN_RANGE_M=0.12`）・極端なアスペクト（`0.55–2.6` 外）といった総崩れケースのみ。

**効いてくる/注意すべきケース**: 直径が**大きく・系統的にずれる**（例: ある色だけ実際は2倍）と絶対距離が同率で
バイアスされ、到達ゲート・回避距離・ラム突入距離がずれて**効率が落ちる**（`RAM_MAX_STEPS`＋RECOVER で回復はする）。
極端な**非球形**（アスペクト逸脱）は `size_consistent` で弾かれうる。実機の破裂は**ピンの物理接触**で決まるので
そもそも仮定直径に非依存（sim の `BALLOON_RADIUS=0.10` は sim 側の真値判定で別問題）。対策が要るなら第一手は
`BALLOON_DIAMETER_M` を実測平均に較正（距離バイアスがそのまま消える）、次に既知高さ事前分布や第二の距離手掛かり。

---

## 1. 検出器モデル（3 つのファミリ）

### (a) 古典的 HSV 色検出器 — `balloon_detector.py`
`detect_balloons()`（`:214`）。RGB → HSV（pure-numpy の `rgb_to_hsv`, `:124`）→ 色ごとの閾値マスク →
`scipy.ndimage` 連結成分 → 面積/充填率フィルタ → ブロブごとの `Detection` → オプションの反射除去。
出力は `area_px` の降順にソートされる。

- 2 つの色**プロファイル**:
  - `SIM_THRESHOLDS`（`:59`, デフォルト; `perception_demo` によりリグレッション保護）— クリーンにレンダリングされた色。
  - `REAL_THRESHOLDS`（`:85`, 約 110 万個のラベル付き px からデータ駆動）— 水中でのシフト: yellow は
    *green* `[(90,160)]` として読まれる; blue はタイトなシアン窓 `[(188,212)]` s≥0.90（プール水と重なる;
    このタイト窓により blue の偽陽性の氾濫を train で約 740 → 約 120 に削減、recall 約 0.17）; red は
    マルーンへ減衰（recall 約 7 %）。
- ブロブフィルタ `MIN_AREA_PX=40`, `MIN_FILL_RATIO=0.35`（`:103`）; confidence `= clip(fill/(π/4),0,1)`。
- 反射除去（`_reject_reflections`, `:168`）— 幾何的: フレーム下部 約 55 % にある、より暗い/小さい同色ブロブを
  破棄する。真の yellow の 約 1/5 のコストがかかる; **デフォルトでは無効**。

### (b) Hough 円形状検出器 — `hough_detector.py`
`detect_hough()`（`:186`）、色検出を補完する実験。RGB → gray → CLAHE（clip 2.0）→ メディアンブラー
（k=5）→ `cv2.HoughCircles` → 内側ディスク（72 %）の HSV 色投票を `REAL_THRESHOLDS` に対して行う。主要パラメータ:
`DP=1.5`, `MIN_RADIUS=10`, `MAX_RADIUS=100`（GT 半径 13–94 px）, `MIN_DIST=24`, `PARAM1=120`,
`PARAM2_ALT=0.4`, `COLOUR_MIN_FRAC=0.18`。`detect_combined()`（`:256`）: `mode="recover"` は色パスが
見落とした円を追加（dedup IoU 0.3）、`mode="confirm"` は円と一致する色ブロブのみを残す。

### (c) 学習済み tiny-CNN — `learned_detector.py`（**推奨、出荷**）
`TinyBalloonNet`（`:64`）— **CenterNet-lite、完全畳み込み(fully-convolutional)、ストライド 8** のアンカーフリー
検出器。約 0.1–0.5 GFLOPs、幅 16 で **約111k パラメータ**（`.pt` 約446 KB / float32）。以降で設計思想から
デコード・学習・エクスポートまで詳述する。

#### 設計思想 — なぜ「学習済み」かつ「極小」か
古典的検出器は水中で**再現率(recall)の物理的な壁**に当たる（`learned_detector.py:3-10`）: red は暗いマルーンへ
減衰し(recall 約7%)、blue はプール水とほぼ同じシアンになる。学習モデルは色閾値が使えない
**テクスチャ / 陰影 / 形状**の手掛かりを原理的に利用できる — ただし**ロボットの Pi 4 上で動く限りにおいて**。
したがって設計の厳しい制約は**レイテンシ**（Pi 4 で ≥5–10 fps を満たすこと）であり、640px のフルサイズ検出器では
なく、意図的に極小な CenterNet-lite ヘッド（約0.1–0.5 GFLOPs）を採用している。

#### アンカーフリー CenterNet パラダイム
物体を「アンカーボックス群への回帰」ではなく、**中心点(center)のヒートマップのピーク**として捉える。
色ごとに 1 枚の中心度ヒートマップを出し、その**局所ピーク = バルーン中心**、ピーク位置でサイズヘッドから
`(w,h)` を読む。これにより古典的なアンカー設計・IoU ベースの NMS が不要になり、ピーク抽出は
**3×3 maxpool** だけで済む（後述、Pi 4 に優しい）。出力は古典的検出器と**同一の `Detection`** なので
`tools/perception_eval` にそのまま差し込める（`:19-20`）。

#### レイヤ構成（幅 `w=16` デフォルト、`conv_bn` = Conv(bias 無し)→BatchNorm→ReLU, `:55`）
バックボーン 5 層で入力を **1/8** に縮小し、2 つのヘッドが分岐する。入力は正方（160/256/320 いずれも可）:

| ブロック | 種別 | 出力ch (w=16) | 特徴マップ (入力=320) |
|---|---|---|---|
| stem | `conv_bn(3,w,↓2)` | 16 | 160×160 (H/2) |
| — | `conv_bn(w,2w,↓2)` | 32 | 80×80 (H/4) |
| — | `conv_bn(2w,2w)` | 32 | 80×80 |
| — | `conv_bn(2w,4w,↓2)` | 64 | 40×40 (H/8) |
| — | `conv_bn(4w,4w)` | 64 | 40×40 |
| **hm_head** | `conv_bn(4w,2w)`→`Conv1×1(2w,3)` | **3** | 40×40（red/yellow/blue の中心度ロジット） |
| **wh_head** | `conv_bn(4w,2w)`→`Conv1×1(2w,2)` | **2** | 40×40（入力に正規化した w,h） |

ストライド 8 なので出力格子は 入力320→40×40 / 256→32×32 / 160→20×20。ヒートマップヘッドの最終 bias は
`−2.19` に初期化し（`:90`）、学習初期を「物体なし」側へ寄せて安定化する（CenterNet の定番トリック）。
チャネル順は `COLOURS = ["red","yellow","blue"]`（`:50`）で、**COCO id ではなく色名でマッピング**する
（`COLOUR_TO_IDX[name]` / `COLOURS[channel]`）ため、内部順序が COCO(red=1/blue=2/yellow=3)と違っても安全。

#### 前処理 → 推論 → デコード
- **前処理** `preprocess`（`:119`）: `(H,W,3)` uint8 → `(1,3,input,input)` float [0,1]、バイリニアで正方リサイズ
  （グレースケール/RGBA も吸収）。色補正は不要（学習時の水中データ拡張で色不変に鍛えてある、§4）。
- **デコード** `decode`（`:137`）— 1 画像分の `(hm, wh)` を**元画像ピクセル座標**の `Detection` 群へ:
  1. `sigmoid(hm)` → `_nms_peaks`（`:130`, 3×3 maxpool で `mx==hm` の格子のみ残す局所ピーク抽出）。
  2. 色チャネルごとに `topk=40` を取り、`conf_thresh=0.3` を下回った時点で打ち切り（スコア降順）。
  3. 各ピーク格子 `(gx,gy)` → 入力px 中心 `((gx+0.5)·8, (gy+0.5)·8)` → `scale = orig/input` で元px中心 `(cx,cy)`。
  4. サイズは `wh` を入力サイズ倍して bbox 化、フレーム内にクランプ（潰れた箱は破棄）。
  5. **古典パスと同じピンホール**で方位角 `az=atan2(cx−cx0, fx)`・仰角 `el=atan2(cy0−cy, fx)`、距離
     `range_m = BALLOON_DIAMETER_M·fx / ((bw+bh)/2)` を計算（`balloon_detector` の `_pinhole` を共有 → 幾何が
     完全一致）。最後に `area_px` 降順にソート。
- 呼び出しは `detect_learned`（`:190`, `@torch.no_grad()`）。

#### 学習目的（`tools/perception_train.py`, §4 も参照）
CenterNet 流の 2 損失: ヒートマップに **focal loss**（ペナルティ軽減付き、GT 中心に**ガウシアンを描画**
`_draw_gaussian`、半径は `_gaussian_radius`(min_overlap 0.7) で決定）+ サイズに **GT 中心での L1**
（`--wh-weight 0.1`）。Adam `lr=2.5e-3` + CosineAnnealing、`--epochs 40 --batch 8 --width 16
--input-size 256|320`。`--init-from` で sim→real ファインチューンをウォームスタート。

#### ロードと設定の優先順位
`load_learned_detector(path)`（`:201`）は `{"state_dict","cfg":{width,input_size,conf_thresh}}` を読み、
`rgb → [Detection]` のクロージャを返す。優先順位: **`width` はチェックポイントが必ず勝つ**（重みと一致必須、
bare state_dict のときのみ引数 `width=16` がフォールバック）; `input_size`/`conf_thresh` は
**呼び出し側の明示引数 > チェックポイント値 > 既定(256 / 0.3)**。→ 呼び出し側が `--conf` を渡さない限り
チェックポイントの `conf_thresh` が有効になる点に注意（`campaign_results.md` の追跡中フォローアップ）。

#### 姉妹モデルとエクスポート
- `PatchVerifierNet`（`:97`）— Hough 提案 + CNN 検証**ハイブリッド**用の極小 3×32×32→4 クラス
  (bg/red/yellow/blue) 分類器。`tools/perception_bench` のベンチ候補のためだけに定義（出荷パスは `TinyBalloonNet`）。
- **エクスポート**（`perception_bench.py` 内、ランタイムモジュールではない）: ONNX opset 13 + 静的な
  チャネルごとの int8 QDQ（活性化 QUInt8 / 重み QInt8）を実フレームでキャリブレーション; 量子化が失敗した
  場合は正直に fp32 へフォールバックする。

**モデルファイル** — `models/perception_learned/`、タグ方式 `camp{round}_{data}.pt`:
- data サフィックス: `_real`（161 枚の実画像）, `_sim`（1000 枚の sim）, `_sim2real`（sim からウォームスタートした
  real ファインチューン）, `_mix`（sim+real 結合）, `_w32`（幅 32 バリアント、幅 16 の 約446 KB に対し 約1.7 MB）。
- round プレフィックス: `camp_*` = R1, `camp2_*` = R2（cast ドメインランダム化）, `camp3_*` = R3
  （照明 DR + 320 px）, `camp4_sim` = R4 sim run。初期ベースライン: `tiny_balloon{,_aug,_real}.pt`。
- **出荷**: `examples/balloon_detector/model.pt` = `camp3_mix.pt`（バイト単位で同一）。その `meta.yaml`:
  `arch TinyBalloonNet, width 16, input_size 320, conf_thresh 0.3, colours [red,yellow,blue]`、
  "sim-pretrain + colour-invariant domain randomisation (campaign round-3 'mix')" で学習。

#### モデル選定の理由 — なぜ既製の YOLO ではなく自前の極小モデルか
出荷モデルは YOLO のような既製検出器ではなく、**タスク特化の極小自前モデル**（`TinyBalloonNet`, 約111k param /
約0.1–0.5 GFLOPs）である。ただし YOLO を検討しなかったわけではなく、**`tools/perception_bench.py` が
YOLOv8n を比較候補として明示的にベンチしている**（候補は (a) TinyBalloonNet / (b) YOLOv8n /
(c) Hough+`PatchVerifier` ハイブリッドの3つ、`ultralytics` も `learn` extra に含まれる）。その比較の上での選択:

| 観点 | 自前 `TinyBalloonNet` | YOLOv8n（nano） |
|---|---|---|
| パラメータ | **約111k** | 約3.2M（約30倍） |
| 計算量 | 約0.1–0.5 GFLOPs | 約8.7 GFLOPs @640 |
| Pi 4 CPU スループット | int8 で余裕をもって **≥10 fps**（@320 で 約12–30 fps 予測） | ncnn で **3.1 fps @640** → 約12 fps @320 / 約19 fps @256 |
| 必要な表現力 | 3色・既知サイズ・無地背景の球のみ。中心+サイズだけ出せば足りる | COCO 80クラス級の容量・多スケール FPN（このタスクには過剰） |
| ライセンス | 制約なし | ultralytics YOLOv8 は **AGPL-3.0**（実機配布で足かせになりうる） |

**決定要因は Pi 4 CPU のレイテンシ**である（`learned_detector.py:3-10`）。この検出器は Pi 4 の CPU 単体で、
50 Hz の制御ループ内（トラッカ＋FSM＋配分と同居）で回す必要がある。YOLOv8n は「動かなくはない」が余裕が乏しく、
知覚以外に割ける時間が消える。加えてタスクが極端に単純（**3色の球・物理サイズ固定・ほぼ無地のプール背景**）で、
直径が既知なので距離は共有ピンホールで見かけ径から幾何計算でき、YOLO の重い箱回帰も 80 クラス容量も要らない。
さらに自前ネットなら幅 / int8 量子化 / sim→real ウォームスタート / 色不変データ拡張レジームまで一気通貫で制御でき、
`Detection` 出力が古典検出器と互換なので `perception_eval` で同一指標の公平比較が成立する（§5）。

**再検討の余地がある条件**（＝ YOLO 系や重量級を評価し直す価値が出るケース）:
- **タスクの難化**: 検出クラスが増える / 雑然・遮蔽の多いシーン / 事前学習バックボーンの転移が効く状況。
- **ハードの変化**: Coral・NPU 等のアクセラレータが使え、CPU レイテンシ制約が外れる場合。
- **精度優先**: 計算予算を気にせず生の recall（特に困難ケース）を最大化したい場合。YOLOv8n は事前学習特徴で
  難シーンの recall が上回りうる。ただし現状は動作範囲（4 m 以内）で recall 約88–90% が出ており、ミッション上は十分（§5）。
- **ライセンス許容**: AGPL-3.0 が配布形態上問題にならない場合。

要するに「自前が常に正義」ではなく、**「3色・既知サイズ・無地背景」を「Pi 4 CPU の 50 Hz ループ内」で解く**という
特定タスク＋特定ハード制約における accuracy/latency 比の最適解として、極小自前モデルを採っている。

---

## 2. 水中色補正 — `underwater.py`
`underwater_correct(rgb, red_compensate=True, clahe=True)`（`:45`）— 決定論的、フィットなし、Pi-4 に優しい。
人間によるラベリング用にフレームを整えるためと、固定の検出器前処理ステップとして（ボックスは不変 — ピクセルは
動くが幾何は動かない）の両方で使われる。3 段階:
1. `_red_compensate`（`:19`）— Ancuti 風の green からの red 補充: `r' = r + (mean(g)−mean(r))·(1−r)·g`。
2. `_gray_world`（`:28`）— 画像ごとのホワイトバランス、ゲイン `= clip(mean_all/mean_ch, 0.5, 2.5)`。
3. `_clahe_l`（`:38`）— Lab L チャネルへの CLAHE（clip 2.0, 8×8 グリッド）。

> この決定論的**復元器**は、合成学習データを生成する物理ベースの sim **劣化**フォワードモデル
> （`umiusi_sim.rendering.underwater_sim`）とは別物である（§4）。バッチツール:
> `tools/underwater_correct.py`（フォルダ → フォルダ）。

---

## 3. トラッカ — `tracker.py`（FSM の前に実行）
`Tracker.update(detections, fresh)`（`:107`）、制御ステップごとに 1 回呼ばれる。ちらつくフレームごとの
検出結果を、挙動 FSM がコミットできる安定した ID に変換する。

- **対応付け**（全ゲート, `:53`）: 色は一致必須; 方位角ゲート `14°`; 距離ゲート `×2.0`（緩い —
  見かけサイズによる距離はフレーム間で 約2 倍振れる）。方位角最近傍の貪欲法、確定済み優先。
- **確定投票**（ちらつき FP を排除, `:68`）: 見えた fresh フレームで +1、見逃した fresh フレームで −1、
  上限 `VOTE_MAX=6`; 投票数 ≥ `CONFIRM_VOTES=3` で **CONFIRMED**、かつスティッキー（確定を解除しない）。
- **持続**（`:76`）: 最大 `PERSIST_FRAMES=12` 連続の見逃しステップを ID/状態を保持したまま生き延びる
  （ドロップアウトを橋渡しし、同じバルーンを再対応付けする）。
- **平滑化**（`:81`）: EMA `BEARING_EMA=0.85`（応答性が高い）、`RANGE_EMA=0.4`（距離が最もノイジー）。
- `plausible_detections`（`:256`）— 距離/サイズゲート: アスペクト `h/w ∈ [0.55, 2.6]`, `min(w,h) ≥ 6 px`,
  距離 `∈ [0.12, max_range]`, 見かけ対期待の直径が `SIZE_TOL=0.6` 以内。
- `sanitise_near_colours`（`:329`）— 近距離（≤ 2.5 m）での実際の bbox ピクセルからの red 対 blue 再確定;
  blue と読まれる "red" は再ラベルされる（安全策: −10 の blue デコイは決してポップしない）。

`tracker.confirmed()` のトラックのみが FSM に到達する。

---

## 4. 学習パイプライン

**合成データ** — `tools/gen_sim_dataset.py`。フレームごと: 実プールスケールの世界をランダム化（18×12 m、
12–30 個のバルーン、色重み red 0.4 / blue 0.3 / yellow 0.3、散在または集塊）→ RGB + メトリック深度 +
セグメンテーションをレンダリング → 深度バッファを用いて物理ベースの `underwater_sim` でクリーン RGB を**劣化** →
セグメンテーションから正確な GT ボックスを抽出。COCO を出力（`red=1, blue=2, yellow=3`）。
`MAX_LABEL_RANGE=7.0 m`: これを超えるバルーンはラベルなしのままとなるため、検出器は**遠方では発火しない**ことを
学習する — 距離ゲートをモデルに焼き込む。生成 約 0.81 s/frame。

**学習** — `tools/perception_train.py`。CenterNet: ヒートマップへの focal loss + GT 中心での wh への L1
（`--wh-weight 0.1`）、ガウシアンターゲット。Adam `lr=2.5e-3`, CosineAnnealing, `--epochs 40 --batch 8
--width 16 --input-size 256|320`。`--init-from` は sim→real ファインチューン用にウォームスタートする。強力なデータ拡張
（albumentations）: flips/affine/crop/brightness/blur、加えて重要な **`_underwater` aug**（p=0.85）:
blue↔green-grey を掃引する積極的なチャネルごとの WB + 深度ベール + 脱彩度により CNN が**絶対色に依存できない**
ようにする（色不変性）、そして実際の 約332×176 圧縮フッテージを模した解像度劣化（`Downscale 0.25–0.6` +
`ImageCompression 28–75`）。

**擬似ラベル** — `tools/perception_pseudolabel.py`: 未ラベルフレームに学習済み検出器を実行 →
COCO json（CVAT/Label Studio/Roboflow へインポート）+ 人間による修正用のプレビュー JPG。

---

## 5. 評価とパフォーマンス

**IoU / PRF ハーネス** — `eval.py`（CLI は `tools/perception_eval.py`, `perception_eval_learned.py`）。貪欲な
IoU マッチ `IOU_TP=0.3`、**色は一致必須**; 色ごと + 全体の precision/recall/F1、加えて FP 数と
下位 ⅓ の FP（水/反射の特徴付け）。`compare()` は 1 つのスプリットで学習済み対 color/hough/combined を実行する。
データセットは**リポジトリ外**（ユーザー提供フォルダ）にある。`DATA_ROOT` は環境変数 `UMIUSI_BALLOON_DATA`
があればそれを、無ければリポジトリと同階層の `../ai/balloon`（= `<repo>/../ai/balloon`）を既定とし、`--data-root`
でも上書きできる。同じ解決ロジックを `eval.py` / `tools/perception_train.py` / `tools/perception_bench.py` の
`_default_data_root()` が共有する（以前ハードコードされていた `/home/satoi/...` の絶対パスは撤去済み）。

**レイテンシベンチ** — `tools/perception_bench.py`: int8 ONNX + onnxruntime、x86 上でシングルスレッド、**Pi 4 へ
予測**（定数 `PI4_INT8_GFLOPS_PER_CORE=10.0`、Cortex-A72 コアあたり 6–15 GFLOP/s の帯域; 4 コアで 約3–3.5×）。
クロスチェックの基準: MobileNet-v1-SSD int8 約28 fps @300、YOLOv8n ncnn 3.1 fps @640 → 約12 fps @320。
int8 で ≥10 fps をクリアする最大サイズの tiny-CNN を推奨する。

**制御分離** — `tools/ram_eval.py`: **合成の完全な検出結果**で実 FSM を駆動し、制御を知覚から分離する。劣化スイープ付き
（`--bearing-noise-deg`, `--range-noise`, `--dropout`, `--fp-rate`, `--perception-hz`）。失敗の分類:
POP / UNDER_TETHER / MISS_ANGLE / MISS_SLOW / MISS_NEAR / MISS_WIDE / NO_COMMIT。

### 引用数値

| metric | value | source |
|---|---|---|
| sim 古典的検出（色+方位角+距離） | 約100 % | README:334 |
| 学習済み 40 枚ベースライン、red recall | 0.00 → **0.82** | README:338 |
| 学習済み 40 枚ベースライン、blue recall | → **0.63** | README:338 |
| 出荷 `camp3_mix` の sim_eval での 4 m 付近 recall | **約0.88** | `examples/balloon_detector/meta.yaml` |
| Pi-4 レイテンシ（int8 ONNX @320 px、予測） | **約12–30 fps** | README:339, `perception_bench.py` |
| 検出距離ゲート | 約4.5 m | README:355 |
| 古典的 real-thresh recall / FP 氾濫削減 | 約0.17 / 740 → 120 | `balloon_detector.py:83` |

**学習キャンペーンのハイライト**（`ai/balloon/campaign_results.md`; sim_train 1000/6753 ボックス、sim_eval
200/1371、real_train 161/863、real_val 25/134）:
- **実フッテージ上で最良のデプロイ可能モデルは `real_only`** — R1 real_val F1 **0.803**（P0.718/R0.910）。
  sim データを追加しても、小さな 25 フレームの real_val ではこれを上回らない（ΔF1 < 0.05 はノイズとして扱う）。
- sim の実証された価値: 見上げ / 遠方 / 高濁度での recall、条件層別化された大規模なストレス+評価セット、そして
  **色不変**なジェネラリストとしての `mix`。R2 の cast-DR は sim-only を 約2 倍改善し（F1 0.184 → 0.349）、
  色キャストのバケット間でフラットにした（色不変性を達成）。R3 の照明 DR + 320 px は sim_eval
  F1 0.41 → 0.49 と 見上げ F1 0.29 → 0.42 を引き上げつつ、Pi-4 の予算を維持した。
- **動作範囲の真実:** sim 学習モデルは **4 m 以内のバルーンの 約88–90 %** を見る（ミッションに関連する
  範囲）; 低い*全体* F1 は、意図的に無視された遠方テールに純粋に起因する。autonomy に推奨:
  `camp3_mix`（出荷）または `camp3_sim`。
