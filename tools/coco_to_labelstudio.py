"""COCO の仮ラベル (perception_pseudolabel の出力) を Label Studio の事前予測付き
タスク JSON に変換する。人手は「確認・修正するだけ」の状態で読み込める。

Label Studio 側の前提 (ai/balloon/raw/LABELING_README.md の手順どおり):
  * LOCAL_FILES_SERVING_ENABLED=true で起動し、Local Storage の Absolute local path に
    --img-dir と同じディレクトリを登録してあること (画像は /data/local-files/?d=<file> で配信)
  * Labeling Interface は ai/balloon/raw/label_config.xml
    (RectangleLabels name="label" toName="image"、red/blue/yellow)

Usage:
    python -m tools.coco_to_labelstudio \
        --coco ai/balloon/raw/pseudo_pool_20260825.json \
        --out  ai/balloon/raw/pool_20260825_labelstudio.json \
        --model-version camp_real@conf0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", required=True, help="COCO json (perception_pseudolabel output)")
    ap.add_argument("--out", required=True, help="Label Studio tasks json to write")
    ap.add_argument("--model-version", default="pseudolabel")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    tasks = []
    for im in coco["images"]:
        w, h = im["width"], im["height"]
        result = []
        for a in by_img.get(im["id"], []):
            x, y, bw, bh = a["bbox"]  # COCO: 絶対ピクセル / Label Studio: パーセント
            result.append({
                "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                "original_width": w, "original_height": h,
                "value": {"x": 100.0 * x / w, "y": 100.0 * y / h,
                          "width": 100.0 * bw / w, "height": 100.0 * bh / h,
                          "rectanglelabels": [cats[a["category_id"]]]},
            })
        tasks.append({
            "data": {"image": f"/data/local-files/?d={im['file_name']}"},
            "predictions": [{"model_version": args.model_version, "result": result}],
        })

    Path(args.out).write_text(json.dumps(tasks, ensure_ascii=False))
    n_boxes = sum(len(t["predictions"][0]["result"]) for t in tasks)
    print(f"{len(tasks)} tasks / {n_boxes} pseudo-boxes -> {args.out}")


if __name__ == "__main__":
    main()
