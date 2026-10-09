#!/usr/bin/env python3
"""人が見て箱を直すための確認ページ（標準ライブラリだけの小さな HTTP サーバ）。

COCO の画像と箱を 1 枚ずつ出し、ブラウザで 消す / 色を変える / 描き足す / 反射として消す をして保存する。
結果は ``--out`` の JSON に 1 枚ずつ追記保存（途中でやめても続きから）。``export`` で COCO に戻す。

  python -m tools.label_review serve --coco ANN.json:IMGDIR [...] --filter ot_j --out review.json \
      --host 100.x.y.z --port 8765
  python -m tools.label_review export --review review.json --coco ANN.json:IMGDIR --out new.json

操作（ページ下にも出す）: 箱をクリック = 選択、Delete/x = 消す、r = 反射として消す、1/2/3 = 赤/青/黄、
空いた所をドラッグ = 新しい箱（今の色）、Enter/→ = 保存して次へ、← = 前へ、f = 自信なし印。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CATS = {1: "red", 2: "blue", 3: "yellow"}
CAT_ID = {v: k for k, v in CATS.items()}


def load_items(specs, filt):
    """[(key, image path, w, h, [{"colour", "bbox"}])] from ``ANN.json:IMGDIR`` specs (key = file_name)."""
    items = []
    for spec in specs:
        ann, imgdir = spec.rsplit(":", 1)
        d = json.load(open(ann))
        cats = {c["id"]: c["name"].replace("balloon_", "") for c in d["categories"]}
        per = {}
        for a in d["annotations"]:
            per.setdefault(a["image_id"], []).append({"colour": cats[a["category_id"]],
                                                      "bbox": [round(v, 1) for v in a["bbox"]]})
        for im in d["images"]:
            if filt and filt not in im["file_name"]:
                continue
            items.append((im["file_name"], pathlib.Path(imgdir) / im["file_name"], im["width"], im["height"],
                          per.get(im["id"], [])))
    items.sort(key=lambda x: x[0])
    return items


PAGE = r"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ラベル確認</title><style>
:root{--bg:#111;--fg:#eee;--mut:#999}body{background:var(--bg);color:var(--fg);font:14px system-ui;margin:0;padding:8px}
#wrap{position:relative;max-width:1400px}#img{display:block;width:100%;height:auto;user-select:none;image-rendering:auto}
canvas{position:absolute;left:0;top:0;width:100%;height:100%;cursor:crosshair}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}button{background:#333;color:var(--fg);border:1px solid #555;padding:6px 10px;border-radius:4px}
button.on{outline:2px solid #fff}.r{color:#f55}.b{color:#58f}.y{color:#fd3}.mut{color:var(--mut)}
</style><body>
<div class="bar"><b id="pos"></b><span id="name" class="mut"></span><span id="state"></span></div>
<div id="wrap"><img id="img"><canvas id="cv"></canvas></div>
<div class="bar">
<button data-c="red" class="r">1 赤</button><button data-c="blue" class="b">2 青</button><button data-c="yellow" class="y">3 黄</button>
<button id="del">x 消す</button><button id="refl">r 反射として消す</button><button id="faint">f 自信なし</button>
<button id="prev">← 前</button><button id="next">保存して次 →</button><button id="todo">未確認へ</button></div>
<div class="mut">箱クリック=選択 / 空所ドラッグ=新規（今の色）/ Enter=保存して次。反射（水面の裏に映った像）・光の粒・壁の模様は消す。
小さくても本物の風船なら付ける。点線 = 反射として消したもの。</div>
<script>
let N=0,i=0,cur=null,boxes=[],sel=-1,colour="red",drag=null,scale=1;
const img=document.getElementById("img"),cv=document.getElementById("cv"),ctx=cv.getContext("2d");
const COL={red:"#ff4040",blue:"#4a8cff",yellow:"#ffd400"};
async function load(k){const r=await fetch("item?i="+k);cur=await r.json();i=k;N=cur.n;
 boxes=JSON.parse(JSON.stringify(cur.boxes));sel=-1;img.src="img?i="+k;
 document.getElementById("pos").textContent=(k+1)+" / "+N+"（確認済み "+cur.done+"）";
 document.getElementById("name").textContent=cur.key;
 document.getElementById("state").textContent=cur.checked?"✔ 確認済み":"未確認";history.replaceState(null,"","#"+k);}
img.onload=()=>{cv.width=img.naturalWidth;cv.height=img.naturalHeight;scale=img.naturalWidth/img.clientWidth;draw();};
function draw(){ctx.clearRect(0,0,cv.width,cv.height);const lw=Math.max(2,cv.width/400);
 boxes.forEach((b,k)=>{const[x,y,w,h]=b.bbox;ctx.lineWidth=k==sel?lw*2:lw;ctx.strokeStyle=COL[b.colour];
  ctx.setLineDash(b.tag=="reflection"?[6,4]:[]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);
  ctx.fillStyle=COL[b.colour];ctx.font=(12*lw/2+8)+"px system-ui";ctx.fillText((b.tag=="reflection"?"反射 ":"")+(b.faint?"? ":"")+b.colour[0].toUpperCase(),x,Math.max(12,y-3));});
 if(drag&&drag.w){ctx.strokeStyle=COL[colour];ctx.lineWidth=lw;ctx.strokeRect(drag.x,drag.y,drag.w,drag.h);}
 document.querySelectorAll("[data-c]").forEach(e=>e.classList.toggle("on",e.dataset.c==colour));}
function pt(e){const r=cv.getBoundingClientRect(),t=e.touches?e.touches[0]:e;return[(t.clientX-r.left)*cv.width/r.width,(t.clientY-r.top)*cv.height/r.height];}
function hit(x,y){let best=-1,area=1e18;boxes.forEach((b,k)=>{const[bx,by,bw,bh]=b.bbox;if(x>=bx&&x<=bx+bw&&y>=by&&y<=by+bh&&bw*bh<area){best=k;area=bw*bh;}});return best;}
cv.onpointerdown=e=>{const[x,y]=pt(e);const k=hit(x,y);if(k>=0){sel=k;draw();return;}drag={x0:x,y0:y,x,y,w:0,h:0};cv.setPointerCapture(e.pointerId);};
cv.onpointermove=e=>{if(!drag)return;const[x,y]=pt(e);drag.x=Math.min(x,drag.x0);drag.y=Math.min(y,drag.y0);drag.w=Math.abs(x-drag.x0);drag.h=Math.abs(y-drag.y0);draw();};
cv.onpointerup=e=>{if(drag&&drag.w>3&&drag.h>3){boxes.push({colour,bbox:[drag.x,drag.y,drag.w,drag.h].map(v=>Math.round(v*10)/10)});sel=boxes.length-1;}drag=null;draw();};
function setc(c){colour=c;if(sel>=0){boxes[sel].colour=c;}draw();}
function del(tag){if(sel<0)return;if(tag){boxes[sel].tag=boxes[sel].tag==tag?undefined:tag;}else{boxes.splice(sel,1);sel=-1;}draw();}
async function save(){await fetch("save?i="+i,{method:"POST",body:JSON.stringify({boxes})});}
async function next(){await save();if(i+1<N)load(i+1);else load(i);}
document.querySelectorAll("[data-c]").forEach(e=>e.onclick=()=>setc(e.dataset.c));
document.getElementById("del").onclick=()=>del();document.getElementById("refl").onclick=()=>del("reflection");
document.getElementById("faint").onclick=()=>{if(sel>=0){boxes[sel].faint=!boxes[sel].faint;draw();}};
document.getElementById("next").onclick=next;document.getElementById("prev").onclick=()=>{if(i>0)load(i-1);};
document.getElementById("todo").onclick=async()=>{const r=await fetch("todo");const j=await r.json();load(j.i);};
document.onkeydown=e=>{if(e.key=="1")setc("red");else if(e.key=="2")setc("blue");else if(e.key=="3")setc("yellow");
 else if(e.key=="Delete"||e.key=="x"||e.key=="Backspace")del();else if(e.key=="r")del("reflection");else if(e.key=="f"){if(sel>=0){boxes[sel].faint=!boxes[sel].faint;draw();}}
 else if(e.key=="Enter"||e.key=="ArrowRight")next();else if(e.key=="ArrowLeft"){if(i>0)load(i-1);}else return;e.preventDefault();};
window.onresize=draw;load(parseInt(location.hash.slice(1))||0);
</script></body></html>"""


class Store:
    def __init__(self, items, out):
        self.items, self.out, self.lock = items, pathlib.Path(out), threading.Lock()
        self.rev = json.loads(self.out.read_text()) if self.out.exists() else {}

    def item(self, i):
        key, _, w, h, boxes = self.items[i]
        r = self.rev.get(key)
        return {"n": len(self.items), "key": key, "w": w, "h": h, "checked": r is not None,
                "boxes": r["boxes"] if r else boxes, "done": len(self.rev)}

    def save(self, i, boxes):
        with self.lock:
            self.rev[self.items[i][0]] = {"boxes": boxes, "checked": True}
            tmp = self.out.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.rev, ensure_ascii=False, indent=0))
            tmp.replace(self.out)

    def todo(self):
        return next((k for k, it in enumerate(self.items) if it[0] not in self.rev), 0)


def serve(args) -> int:
    store = Store(load_items(args.coco, args.filter), args.out)
    print(f"{len(store.items)} images, {len(store.rev)} already reviewed -> {args.out}")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            i = int(q.get("i", ["0"])[0])
            if u.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/item":
                self._send(200, json.dumps(store.item(i)).encode(), "application/json")
            elif u.path == "/img":
                p = store.items[i][1]
                self._send(200, p.read_bytes(), "image/png" if p.suffix == ".png" else "image/jpeg")
            elif u.path == "/todo":
                self._send(200, json.dumps({"i": store.todo()}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/save":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            store.save(int(parse_qs(u.query)["i"][0]), body["boxes"])
            self._send(200, b"{}", "application/json")

    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"open http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def export(args) -> int:
    rev = json.loads(pathlib.Path(args.review).read_text())
    items = load_items(args.coco, args.filter)
    images, anns, n_refl = [], [], 0
    for k, (key, _, w, h, boxes) in enumerate(items, start=1):
        if args.reviewed_only and key not in rev:
            continue
        images.append({"id": k, "file_name": key, "width": w, "height": h})
        for b in (rev[key]["boxes"] if key in rev else boxes):
            if b.get("tag") == "reflection":
                n_refl += 1
                continue
            x, y, bw, bh = b["bbox"]
            anns.append({"id": len(anns) + 1, "image_id": k, "category_id": CAT_ID[b["colour"]],
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
    out = {"images": images, "annotations": anns, "categories": [{"id": i, "name": n} for i, n in CATS.items()]}
    pathlib.Path(args.out).write_text(json.dumps(out))
    print(f"{len(images)} images ({sum(1 for it in items if it[0] in rev)} reviewed), {len(anns)} boxes, "
          f"{n_refl} reflections dropped -> {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "export"):
        p = sub.add_parser(name)
        p.add_argument("--coco", nargs="+", required=True, help="ANN.json:IMGDIR (image path = IMGDIR/file_name)")
        p.add_argument("--filter", default="", help="keep only file_names containing this")
        if name == "serve":
            p.add_argument("--out", required=True, help="review JSON (resumable)")
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8765)
        else:
            p.add_argument("--review", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--reviewed-only", action="store_true")
    args = ap.parse_args()
    return serve(args) if args.cmd == "serve" else export(args)


if __name__ == "__main__":
    sys.exit(main())
