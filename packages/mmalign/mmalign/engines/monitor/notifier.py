"""WebhookNotifier — fire-and-forget HTTP notification on drift alerts.

Design (PRD §US-12, TDD §MonitorEngine):
  When DriftEngine emits alerts the FeedbackLoopEngine calls
  WebhookNotifier.notify() as a side-effect.  Failures are swallowed
  so a bad webhook never blocks the main loop.

Configuration (policies.yaml)::

    monitor:
      webhook_url: "https://your-service/hooks/mmalign"
      webhook_secret: ""          # optional HMAC-SHA256 signing key
"""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .drift import DriftAlert


class WebhookNotifier:
    """Send drift-alert notifications to a configured webhook URL."""

    def __init__(self, url: str, secret: str = "", timeout_s: int = 5) -> None:
        self._url = url.strip()
        self._secret = secret
        self._timeout = timeout_s

    @classmethod
    def from_configs_dir(cls, configs_dir: Path | None) -> "WebhookNotifier":
        """Load webhook config from configs/policies.yaml; disabled if absent."""
        if configs_dir is None:
            return cls("")
        try:
            import yaml

            path = Path(configs_dir) / "policies.yaml"
            if not path.exists():
                return cls("")
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            monitor_cfg = raw.get("monitor", {}) or {}
            return cls(
                url=monitor_cfg.get("webhook_url", ""),
                secret=monitor_cfg.get("webhook_secret", ""),
            )
        except Exception:
            return cls("")

    def notify(self, alerts: list[DriftAlert], scenario: str = "", model: str = "") -> None:
        """POST an alert payload to the webhook URL; silently swallows errors."""
        if not self._url or not alerts:
            return
        body = json.dumps(
            {
                "event": "drift_detected",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "alerts": [
                    {
                        "id": a.id,
                        "metric": a.metric,
                        "level": a.level,
                        "scenario": a.scenario,
                        "model": a.model,
                        "baseline": a.baseline,
                        "current": a.current,
                        "change_pct": a.change_pct,
                        "message": a.message,
                        "detected_at": a.detected_at,
                    }
                    for a in alerts
                ],
                "filter": {"scenario": scenario, "model": model},
            }
        ).encode()

        headers = {"Content-Type": "application/json"}
        if self._secret:
            headers["X-MMAlign-Signature"] = hmac.new(
                self._secret.encode(), body, hashlib.sha256
            ).hexdigest()

        try:
            req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
            urllib.request.urlopen(req, timeout=self._timeout)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            pass
