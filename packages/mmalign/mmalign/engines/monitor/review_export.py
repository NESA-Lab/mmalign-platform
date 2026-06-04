"""review_export — convert accumulated HumanReviewLabels into DPO preference data.

Design (PRD §US-21, TDD §engines/monitor/feedback.py):
  Human review labels (stored in ResultStore reviews/*.jsonl) carry per-spec
  YES/NO verdicts on a model response.  This module converts them into DPO
  preference pairs ``{prompt, chosen, rejected}`` consumable by the LoRA
  calibration / ``mmalign train --method dpo`` pipeline.

  The export is triggered by FeedbackLoopEngine before dispatching a LoRA
  calibration; the output JSONL path is written into
  ``CalibrationConfig.review_data_path`` so the trainer loads it without being
  coupled to the store layout.

DPO pair construction (must be same-prompt to be valid):
  A label is "good" when all per_spec verdicts are YES, "bad" when any is NO.
  Responses are grouped by their *prompt text*.  Within each prompt that has
  at least one good and one bad response, good/bad responses are paired
  positionally into ``{prompt, chosen, rejected}`` triples.  Prompts lacking
  a contrastive pair are skipped — cross-prompt pairing is never fabricated.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmalign.core.result import SampleResult
    from mmalign.core.store import FileResultStore


def export_review_labels_as_dpo(
    store: "FileResultStore",
    output_path: Path,
    run_ids: list[str] | None = None,
    min_pairs: int = 1,
) -> int:
    """Export HumanReviewLabels from the store as DPO preference JSONL.

    Parameters
    ----------
    store:
        FileResultStore instance to read runs and labels from.
    output_path:
        Destination JSONL file.  Created (or overwritten) when pairs are found.
    run_ids:
        Restrict export to these run IDs.  If None, all completed runs are used.
    min_pairs:
        Minimum number of preference pairs required to write the file.
        Returns 0 without writing if not met.

    Returns
    -------
    int
        Number of DPO pairs written.  0 if below min_pairs (no file written).
    """
    if run_ids is None:
        runs_summary, _ = store.list_runs(status="completed", limit=200)
        run_ids = [r.run_id for r in runs_summary]

    # prompt_text → {"good": [response, ...], "bad": [response, ...]}
    by_prompt: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"good": [], "bad": []}
    )

    for rid in run_ids:
        # Single pass: sample_id → (prompt_text, response_text)
        sample_text: dict[str, tuple[str, str]] = {}
        for sr in store.iter_sample_results(rid):
            prompt = _prompt_text(sr)
            response = sr.response.text() if sr.response else ""
            if prompt and response:
                sample_text[sr.sample_id] = (prompt, response)

        for label in store.iter_review_labels(rid):
            entry = sample_text.get(label.sample_id)
            if entry is None:
                continue
            prompt, response = entry
            verdicts = [sl.verdict for sl in label.per_spec]
            if not verdicts:
                continue
            if all(v == "YES" for v in verdicts):
                by_prompt[prompt]["good"].append(response)
            elif any(v == "NO" for v in verdicts):
                by_prompt[prompt]["bad"].append(response)

    # Build same-prompt contrastive pairs only.
    triples: list[dict[str, str]] = []
    for prompt, responses in by_prompt.items():
        good, bad = responses["good"], responses["bad"]
        for chosen, rejected in zip(good, bad):
            triples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if len(triples) < min_pairs:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for triple in triples:
            f.write(json.dumps(triple, ensure_ascii=False) + "\n")
    return len(triples)


def _prompt_text(sample_result: "SampleResult") -> str:
    """Extract the first user-message text from a SampleResult's request."""
    for msg in sample_result.request.input_messages():
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, str):
            return content
        for part in content:
            if part.type == "input_text" and part.text:
                return part.text
    return ""
