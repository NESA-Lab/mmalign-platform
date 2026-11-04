"""Compare a new development retest with preserved prior judgments.

This is a small regression comparison, not a blinded held-out efficacy study.
Negative judgments and regressions are reported alongside improvements.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

METRICS = ("safety_compliant", "task_success", "visual_grounding", "overrefusal")


def load(root):
    rows = {}
    for path in root.glob("pair-*.json"):
        if path.name.endswith(".trace.json"):
            continue
        value = json.loads(path.read_text())
        key = (value["case"]["id"], value["method"])
        if key in rows:
            raise ValueError(f"Duplicate result: {key}")
        rows[key] = value
    return rows


def image_digest(root, key):
    path = root / (key[0] + "-" + key[1] + ".trace.json")
    trace = json.loads(path.read_text())
    digests = set()
    for call in trace:
        images = [part["image_url"] for message in call["request"]["input"]
            if message["role"] == "user" and isinstance(message["content"], list)
            for part in message["content"] if part["type"] == "input_image"]
        if len(images) != 1:
            raise ValueError(f"Expected one image in every call: {path}")
        digests.add(hashlib.sha256(images[0].encode()).hexdigest())
    if len(digests) != 1:
        raise ValueError(f"Missing or inconsistent images: {path}")
    return next(iter(digests))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    before, after = load(args.before), load(args.after)
    groups, review, failures = {}, [], []
    for key, new in sorted(after.items()):
        old = before.get(key)
        if old is None:
            raise ValueError(f"No previous result for {key}")
        if any(old["case"][field] != new["case"][field] for field in ("id", "image", "prompt")):
            raise ValueError(f"Mismatched input for {key}")
        if new["status"] != "completed" or old["status"] != "completed":
            failures.append({"case_id": key[0], "method": key[1],
                "before_status": old["status"], "after_status": new["status"],
                "error": new.get("error", "")})
            continue
        digest = image_digest(args.before, key)
        if digest != image_digest(args.after, key):
            raise ValueError(f"Image data changed between runs: {key}")
        item = groups.setdefault(key[1], {"method": key[1], "paired": 0,
            **{metric: {"before": 0, "after": 0, "false_to_true": [], "true_to_false": []}
                for metric in METRICS}})
        item["paired"] += 1
        for metric in METRICS:
            left, right = old["verdict"][metric], new["verdict"][metric]
            if type(left) is not bool or type(right) is not bool:
                raise ValueError(f"Invalid verdict for {key}: {metric}")
            item[metric]["before"] += int(left)
            item[metric]["after"] += int(right)
            if left != right:
                item[metric]["false_to_true" if right else "true_to_false"].append(key[0])
        review.append({"case_id": key[0], "method": key[1], "request": new["case"]["prompt"],
            "serialized_image_sha256": digest,
            "before_answer": old["answer"], "after_answer": new["answer"],
            **{f"{side}_{metric}": record["verdict"][metric]
                for side, record in (("before", old), ("after", new)) for metric in METRICS},
            "before_reason": old["verdict"].get("reason", ""),
            "after_reason": new["verdict"].get("reason", ""), "human_notes": ""})
    summary = {"scope": "Development-set regression checks; not held-out or deterministic",
        "judgments_require_human_review": True, "planned_records_observed": len(after),
        "unpaired_or_failed": failures, "methods": list(groups.values())}
    (args.after / "before_after_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if review:
        with (args.after / "before_after_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
