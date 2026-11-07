#!/usr/bin/env python3
"""Convert a spec-format source_root into ms-swift's multimodal jsonl (messages + images, <image> tag).

  python to_swift.py --root source_root --out prism_cot_swift.jsonl
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for jf in sorted(a.root.rglob("*.jsonl")):
            if any(p.startswith(".") for p in jf.relative_to(a.root).parts):
                continue
            for line in open(jf, encoding="utf-8"):
                if not line.strip():
                    continue
                obj = json.loads(line)
                msgs, images = [], []
                for m in obj.get("trace", obj)["messages"]:
                    c = m["content"]
                    if isinstance(c, list):
                        text = ""
                        for b in c:
                            if b["type"] == "image":
                                text += "<image>"; images.append(str((a.root / b["image"]).resolve()))
                            else:
                                text += b["text"]
                        c = text
                    msgs.append({"role": m["role"], "content": c})
                f.write(json.dumps({"messages": msgs, "images": images}, ensure_ascii=False) + "\n"); n += 1
    print(f"{n} records -> {a.out}")


if __name__ == "__main__":
    main()
