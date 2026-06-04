"""Regression tests for the webhook notifier and review→DPO export.

These cover the two paths that the monitor API tests do not exercise:
  * WebhookNotifier payload construction (must use real DriftAlert fields).
  * export_review_labels_as_dpo same-prompt contrastive pairing.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mmalign.core.result import BenchmarkRun, HumanReviewLabel, RunSummary, SampleResult, SpecLabel
from mmalign.core.store import FileResultStore
from mmalign.core.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationResult,
    InputContentPart,
    InputMessage,
    OutputContentPart,
)
from mmalign.engines.monitor.drift import DriftAlert
from mmalign.engines.monitor.notifier import WebhookNotifier
from mmalign.engines.monitor.review_export import export_review_labels_as_dpo


def _alert() -> DriftAlert:
    return DriftAlert(
        id="drift-2026-01-01",
        level="error",
        scenario="education",
        model="mock",
        metric="safety_score",
        message="Safety score dropped 66.7%",
        detected_at="2026-01-01T00:00:00Z",
        baseline=0.90,
        current=0.30,
        change_pct=-66.7,
    )


# ---------------------------------------------------------------------------
# WebhookNotifier
# ---------------------------------------------------------------------------

class TestWebhookNotifier:
    def test_empty_url_is_noop(self) -> None:
        # Must not raise and must not attempt any network call.
        WebhookNotifier("").notify([_alert()], scenario="education", model="mock")

    def test_payload_builds_with_real_alert_fields(self) -> None:
        """Regression: notify() must read DriftAlert.current/baseline/change_pct,
        not non-existent *_value fields.  A bogus URL exercises the full payload
        build then swallows the connection error."""
        notifier = WebhookNotifier("http://127.0.0.1:1/never")
        # Would raise AttributeError during payload build if field names were wrong.
        notifier.notify([_alert()], scenario="education", model="mock")

    def test_signature_header_added_when_secret_set(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_urlopen(req, timeout=0):  # noqa: ANN001
            captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
            captured["body"] = req.data
            raise OSError("stop after capture")

        import urllib.request as ur
        monkeypatch.setattr(ur, "urlopen", fake_urlopen)

        WebhookNotifier("http://x/y", secret="s3cr3t").notify([_alert()])
        assert "x-mmalign-signature" in captured["headers"]
        body = json.loads(captured["body"])
        assert body["event"] == "drift_detected"
        assert body["alerts"][0]["current"] == 0.30
        assert body["alerts"][0]["change_pct"] == -66.7


# ---------------------------------------------------------------------------
# review → DPO export
# ---------------------------------------------------------------------------

def _save_labeled_sample(
    store: FileResultStore,
    run_id: str,
    sample_id: str,
    prompt: str,
    response: str,
    verdict: str,
) -> None:
    store.append_sample_result(
        SampleResult(
            run_id=run_id,
            sample_id=sample_id,
            scenario_id="education",
            model="mock",
            method="align3",
            evaluator="mock_judge",
            request=GenerationRequest(
                model="mock",
                input=[InputMessage(role="user", content=[InputContentPart.text_part(prompt)])],
            ),
            response=GenerationResult(
                id=f"gen-{uuid.uuid4().hex[:6]}",
                model="mock",
                output=[GenerationOutput(content=[OutputContentPart.text_part(response)])],
            ),
            created_at=datetime.now(tz=timezone.utc),
        )
    )
    store.append_review_label(
        HumanReviewLabel(
            run_id=run_id,
            sample_id=sample_id,
            reviewer_id="rev-1",
            queue_reason="low_confidence",
            per_spec=[SpecLabel(spec_id="safety_0", verdict=verdict)],
        )
    )


def _new_run(store: FileResultStore) -> str:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    store.save_run(
        BenchmarkRun(
            run_id=run_id,
            status="completed",
            scenario="education",
            model="mock",
            method="align3",
            evaluator="mock_judge",
            summary=RunSummary(
                safety_score=0.5, behavioral_score=0.5, total_score=0.5,
                unsafe_rate=0.5, capability_rate=0.9, sample_count=2,
            ),
            started_at=datetime.now(tz=timezone.utc),
            completed_at=datetime.now(tz=timezone.utc),
        )
    )
    return run_id


class TestReviewExport:
    def test_no_pairs_when_only_good(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        rid = _new_run(store)
        _save_labeled_sample(store, rid, "s1", "Q1", "good answer", "YES")
        out = tmp_path / "dpo.jsonl"
        assert export_review_labels_as_dpo(store, out) == 0
        assert not out.exists()

    def test_pairs_only_within_same_prompt(self, tmp_path: Path) -> None:
        """Regression: chosen and rejected in a triple must answer the SAME
        prompt.  A good+bad pair on prompt A and a lone bad on prompt B should
        yield exactly one triple (for A), never a cross-prompt fabrication."""
        store = FileResultStore(tmp_path / "results")
        rid = _new_run(store)
        # Prompt A: one good, one bad → 1 valid pair
        _save_labeled_sample(store, rid, "a_good", "Question A", "A good", "YES")
        _save_labeled_sample(store, rid, "a_bad", "Question A", "A bad", "NO")
        # Prompt B: only a bad → no contrastive partner
        _save_labeled_sample(store, rid, "b_bad", "Question B", "B bad", "NO")

        out = tmp_path / "dpo.jsonl"
        n = export_review_labels_as_dpo(store, out)
        assert n == 1
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        triple = json.loads(lines[0])
        assert triple["prompt"] == "Question A"
        assert triple["chosen"] == "A good"
        assert triple["rejected"] == "A bad"
