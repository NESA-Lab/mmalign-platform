#!/usr/bin/env python3
"""Validate a multimodal SFT delivery (spec v1.0, image) and compute the info-table statistics.

  python validate_delivery.py --root <source_root> --processor <Qwen3.5-VL processor dir> [--max-seq-length 262144]

Checks (spec section 6), all must pass:
  1 structure   2 content blocks   3 tools   4 media (paths, manifest, sha256, size, decode)
  5 budget (processor-expanded length; non-empty assistant after right truncation)
  6 rendering (chat template renders; assistant span non-empty)
Writes <root>/../validation_report.json with pass/fail per check and token statistics.
"""
import argparse, hashlib, json, re, sys
from collections import Counter
from pathlib import Path, PurePosixPath
from PIL import Image

ROLES = {"system", "user", "assistant", "tool"}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
NAME_OK = re.compile(r"^[A-Za-z0-9._/-]+$")


def jsonl_files(root: Path):
    for p in sorted(root.rglob("*.jsonl")):
        if not any(part.startswith(".") for part in p.relative_to(root).parts):
            yield p


def path_problem(p: str):
    if not isinstance(p, str) or not p:
        return "empty"
    if p.startswith(("/", "~")) or ".." in PurePosixPath(p).parts or "~" in p:
        return "absolute/parent/home path"
    if re.match(r"^(https?|data|file):", p):
        return "url/base64"
    if "\\" in p or not NAME_OK.match(p):
        return "illegal characters"
    if any(part.startswith(".") for part in PurePosixPath(p).parts[:-1]):
        return "hidden directory"
    if PurePosixPath(p).suffix not in EXTS:
        return "extension not allowed"
    return None


def stats(xs):
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"mean": round(sum(xs) / len(xs), 1), "p50": q(.5), "p90": q(.9), "p99": q(.99), "max": xs[-1], "min": xs[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--processor", required=True)
    ap.add_argument("--max-seq-length", type=int, default=262144)
    a = ap.parse_args()
    from transformers import AutoProcessor
    P = AutoProcessor.from_pretrained(a.processor)
    pad_id = P.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    errors = {k: [] for k in ("structure", "blocks", "tools", "media", "budget", "render")}
    def err(k, msg):
        if len(errors[k]) < 50:
            errors[k].append(msg)

    # ---- media: manifest vs files on disk
    man = json.load(open(a.root / "manifest.json", encoding="utf-8"))
    reg = {}
    for f in man.get("files", []):
        if f["path"] in reg:
            err("media", f"manifest duplicate {f['path']}")
        reg[f["path"]] = f
    on_disk = {str(p.relative_to(a.root)).replace("\\", "/") for p in a.root.rglob("*")
               if p.is_file() and p.suffix.lower() in EXTS | {".gif", ".bmp", ".tif", ".tiff", ".mp4"}}
    for p in sorted(on_disk - set(reg)):
        err("media", f"image not in manifest: {p}")
    for p, f in reg.items():
        prob = path_problem(p)
        if prob:
            err("media", f"{p}: {prob}"); continue
        fp = a.root / p
        if not fp.exists():
            err("media", f"registered but missing: {p}"); continue
        raw = fp.read_bytes()
        if hashlib.sha256(raw).hexdigest() != f.get("sha256"):
            err("media", f"sha256 mismatch {p}")
        if len(raw) != f.get("bytes"):
            err("media", f"bytes mismatch {p}")
        if len(raw) > 20 * 2**20:
            err("media", f"> 20 MB {p}")
        try:
            im = Image.open(fp); im.load()
            if im.size != (f.get("width"), f.get("height")):
                err("media", f"size mismatch {p}")
            fmt = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}, "WEBP": {".webp"}}.get(im.format, set())
            if fp.suffix not in fmt:
                err("media", f"extension does not match encoding {p} ({im.format})")
        except Exception as e:  # noqa: BLE001
            err("media", f"cannot decode {p}: {e}")

    # ---- records
    n_rec = 0; total_tok, sup_tok, img_tok, img_per = [], [], [], []
    roles, used = Counter(), set()
    for jf in jsonl_files(a.root):
        for ln, line in enumerate(open(jf, encoding="utf-8"), 1):
            if not line.strip():
                continue
            where = f"{jf.relative_to(a.root)}:{ln}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                err("structure", f"{where}: invalid JSON ({e})"); continue
            obj = obj.get("trace", obj)
            msgs, tools = obj.get("messages"), obj.get("tools")
            if not isinstance(msgs, list) or not msgs:
                err("structure", f"{where}: messages missing"); continue
            n_rec += 1
            bad = False
            for i, m in enumerate(msgs):
                r = m.get("role"); roles[r] += 1
                if r not in ROLES:
                    err("structure", f"{where}: bad role {r}"); bad = True
                if r == "system" and i != 0:
                    err("structure", f"{where}: system not first"); bad = True
                c = m.get("content")
                if isinstance(c, list):
                    for b in c:
                        t = b.get("type")
                        if t not in ("text", "image"):
                            err("blocks", f"{where}: block type {t}"); bad = True
                        elif t == "text" and not (isinstance(b.get("text"), str) and b["text"]):
                            err("blocks", f"{where}: empty text block"); bad = True
                        elif t == "image":
                            if r != "user":
                                err("blocks", f"{where}: image in {r} message"); bad = True
                            prob = path_problem(b.get("image"))
                            if prob:
                                err("blocks", f"{where}: image path {prob}"); bad = True
                            elif b["image"] not in reg:
                                err("media", f"{where}: image not registered {b['image']}"); bad = True
                            else:
                                used.add(b["image"])
                elif not isinstance(c, str):
                    if not (r == "assistant" and m.get("tool_calls")):
                        err("structure", f"{where}: content must be string or block list"); bad = True
                for tc in m.get("tool_calls") or []:
                    args = tc.get("function", {}).get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = None
                    if not isinstance(args, dict):
                        err("tools", f"{where}: tool arguments not an object"); bad = True
            last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=None)
            if last_user is None:
                err("structure", f"{where}: no user query"); continue
            if not any(m["role"] == "assistant" and m.get("use_loss", True) for m in msgs[last_user + 1:]):
                err("structure", f"{where}: no supervised assistant after last user"); continue
            if bad:
                continue
            # ---- render + expand (processor = receiver's counting)
            def to_proc(ms):
                out = []
                for m in ms:
                    c = m["content"]
                    if isinstance(c, list):
                        c = [{"type": "image", "image": str(a.root / b["image"])} if b["type"] == "image" else b for b in c]
                    out.append({**{k: v for k, v in m.items() if k not in ("use_loss", "use_thinking")}, "content": c})
                return out
            try:
                pm = to_proc(msgs)
                full = P.apply_chat_template(pm, tools=tools, tokenize=False)
                prefix = P.apply_chat_template(pm[:last_user + 1], tools=tools, tokenize=False, add_generation_prompt=True)
                if not full.startswith(prefix) or not full[len(prefix):].strip():
                    err("render", f"{where}: empty assistant span"); continue
                imgs = [Image.open(a.root / b["image"]).convert("RGB") for m in msgs if isinstance(m["content"], list)
                        for b in m["content"] if b["type"] == "image"]
                ids = P(text=[full], images=imgs or None, return_tensors="pt")["input_ids"][0]
                pre = P(text=[prefix], images=imgs or None, return_tensors="pt")["input_ids"][0]
            except Exception as e:  # noqa: BLE001
                err("render", f"{where}: {type(e).__name__}: {str(e)[:150]}"); continue
            n, nsup = len(ids), len(ids) - len(pre)
            npad = int((ids == pad_id).sum())
            if n > a.max_seq_length:
                err("budget", f"{where}: {n} tokens > max_seq_length")
            if min(n, a.max_seq_length) - len(pre) <= 0:
                err("budget", f"{where}: no supervised tokens after right truncation")
            total_tok.append(n); sup_tok.append(nsup); img_tok.append(npad + 2 * len(imgs)); img_per.append(len(imgs))
            if n_rec % 1000 == 0:
                print(f"  checked {n_rec}", flush=True)
    for p in sorted(set(reg) - used):
        err("media", f"registered image never referenced: {p}")

    checks = {k: {"pass": not v, "errors": v} for k, v in errors.items()}
    hist_edges = [512, 1024, 2048, 4096, 8192]
    hist = Counter(next((f"<={e}" for e in hist_edges if t <= e), f">{hist_edges[-1]}") for t in total_tok)
    report = {
        "root": str(a.root), "processor": a.processor, "max_seq_length": a.max_seq_length,
        "passed": all(c["pass"] for c in checks.values()), "checks": checks,
        "records": n_rec, "records_with_image": sum(1 for x in img_per if x), "records_text_only": sum(1 for x in img_per if not x),
        "images_registered": len(reg), "images_referenced": len(used), "roles": dict(roles),
        "tokens": {"total": sum(total_tok), "trainable": sum(sup_tok), "image": sum(img_tok),
                   "per_sample_total": stats(total_tok), "per_sample_trainable": stats(sup_tok),
                   "per_image": stats([t for t, k in zip(img_tok, img_per) if k]) if any(img_per) else None,
                   "total_length_hist": {k: hist[k] for k in [f"<={e}" for e in hist_edges] + [f">{hist_edges[-1]}"]}},
    }
    out = a.root.parent / "validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "records", "records_with_image", "images_registered")}, ensure_ascii=False))
    print("checks:", {k: v["pass"] for k, v in checks.items()})
    for k, v in checks.items():
        for e in v["errors"][:5]:
            print(f"  [{k}] {e}")
    print("report ->", out)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
