#!/usr/bin/env python3
"""Build the PRISM-CoT SFT delivery (multimodal SFT delivery spec v1.0, image) from the released dataset.

  python build_delivery.py --src <PRISM-CoT dir with data.json + images/> --out <source_root>

source_root/
  manifest.json            image list with sha256 / bytes / width / height
  vision/prism_cot.jsonl   10,000 records: user = [image, text] blocks, assistant = string
  assets/<sha256[:16]>.<ext>
"""
import argparse, hashlib, json, shutil
from datetime import date
from pathlib import Path
from PIL import Image

EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
MAX_PIXELS = 8192 * 1024      # spec 4.2: <= 8192 expanded tokens per image (16px patch, 2x2 merge)
MAX_EDGE = 4096               # spec 3.1: long edge <= 4096 px


def fit(im):
    """Downscale (keeping aspect) only if the image exceeds the per-image budget."""
    w, h = im.size
    s = min(1.0, (MAX_PIXELS / (w * h)) ** 0.5, MAX_EDGE / max(w, h))
    if s >= 1.0:
        return None
    return im.resize((max(32, int(w * s) // 32 * 32), max(32, int(h * s) // 32 * 32)), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", default="prism_cot_v1")
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "assets").mkdir(parents=True); (a.out / "vision").mkdir()
    rows = json.load(open(a.src / "data.json", encoding="utf-8"))
    files, path_of, converted, resized = {}, {}, 0, 0
    with open(a.out / "vision/prism_cot.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            src = a.src / "images" / r["image"]
            if src not in path_of:
                raw = src.read_bytes()
                im = Image.open(src)
                fmt = im.format
                small = fit(im)
                if fmt not in EXT or im.mode not in ("RGB", "L", "RGBA", "P") or small is not None:
                    from io import BytesIO
                    out_im = (small if small is not None else im)
                    if out_im.mode not in ("RGB", "L"):
                        out_im = out_im.convert("RGB")
                    fmt = fmt if fmt in ("JPEG", "PNG") else "PNG"
                    buf = BytesIO(); out_im.save(buf, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
                    raw = buf.getvalue()
                    resized += small is not None; converted += small is None
                digest = hashlib.sha256(raw).hexdigest()
                rel = f"assets/{digest[:16]}{EXT[fmt]}"
                if rel not in files:
                    (a.out / rel).write_bytes(raw)
                    w, h = Image.open(a.out / rel).size
                    files[rel] = {"path": rel, "sha256": digest, "bytes": len(raw), "width": w, "height": h}
                path_of[src] = rel
            rec = {"messages": [
                {"role": "user", "content": [{"type": "image", "image": path_of[src]},
                                             {"type": "text", "text": r["question"]}]},
                {"role": "assistant", "content": r["response"]}]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = {"version": "1.0",
                "dataset": {"name": "prism_cot", "run_id": a.run_id, "created_at": date.today().isoformat(),
                            "source": "https://huggingface.co/datasets/andyc03/PRISM-CoT@d8556fc"},
                "files": sorted(files.values(), key=lambda x: x["path"])}
    json.dump(manifest, open(a.out / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"records": len(rows), "unique_images": len(files), "re_encoded": converted, "downscaled": resized,
                      "assets_bytes": sum(x["bytes"] for x in files.values())}))


if __name__ == "__main__":
    main()
