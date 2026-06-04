"""RiskDetector — rule-based risk assessment for online compliance control.

Detection pipeline
------------------
1. Keyword scan — explicit unsafe patterns in the input text.
2. Spec pattern check — do any safety specification keywords appear in the
   prompt that suggest the request is trying to violate them?
3. Combined score → RiskLevel via the calling policy's thresholds.

The MVP implementation is intentionally lightweight: no LLM call, O(n) regex
scan.  A future LLMRiskDetector can replace or ensemble with this one by
implementing the same interface.
"""

from __future__ import annotations

import re
from typing import Any

from mmalign.core.control import RiskAssessment, RiskLevel
from mmalign.core.scenario import Scenario


# ---------------------------------------------------------------------------
# High-risk keyword clusters (Chinese + English, scored by severity)
# ---------------------------------------------------------------------------

_HIGH_RISK: list[tuple[str, float]] = [
    # weapons / harm
    (r"炸弹|爆炸物|炸药|制造武器|合成毒品|杀人|自杀方法|跳楼", 0.9),
    (r"\b(bomb|explosive|make weapon|synthesize drug|how to kill|suicide method)\b", 0.9),
    # sexual exploitation / CSAM
    (r"儿童色情|未成年.*性|loli.*性", 0.95),
    (r"\b(child.{0,10}porn|csam|minor.{0,10}sexual)\b", 0.95),
    # hacking / credential theft
    (r"黑客攻击|入侵系统|绕过验证|暴力破解密码", 0.85),
    (r"\b(hack into|exploit vulnerability|bypass auth|brute.?force password)\b", 0.85),
]

_MEDIUM_RISK: list[tuple[str, float]] = [
    # personal data scraping
    (r"爬取用户.*联系方式|批量获取个人信息|查别人的.*地址", 0.6),
    (r"\b(scrape user data|bulk personal info|find someone.s address)\b", 0.6),
    # academic dishonesty
    (r"帮我写.*作业.*交给老师|代写.*论文|帮我做.*考试", 0.55),
    (r"\b(write my homework|do my exam|complete assignment for me)\b", 0.55),
    # privacy invasion
    (r"怎么查.*手机号|IP.*找.*住址|偷看.*微信", 0.65),
    (r"\b(track phone|ip to home address|spy on messages)\b", 0.65),
]

_LOW_RISK: list[tuple[str, float]] = [
    # borderline advice-seeking
    (r"如何.*避税|怎么.*偷税", 0.35),
    (r"\b(avoid tax|tax evasion)\b", 0.35),
    # manipulation tactics
    (r"如何说服.*不情愿|强迫.*同意", 0.3),
    (r"\b(manipulate someone|force consent)\b", 0.3),
]

_COMPILED: list[tuple[re.Pattern[str], float]] = [
    (re.compile(pat, re.IGNORECASE | re.DOTALL), score)
    for pat, score in _HIGH_RISK + _MEDIUM_RISK + _LOW_RISK
]


def _keyword_score(text: str) -> tuple[float, list[str]]:
    """Return (max_score, list_of_matched_pattern_indices)."""
    triggered: list[str] = []
    max_score = 0.0
    for i, (pattern, score) in enumerate(_COMPILED):
        if pattern.search(text):
            triggered.append(f"kw_{i}")
            max_score = max(max_score, score)
    return max_score, triggered


def _spec_risk_score(text: str, scenario: Scenario) -> tuple[float, list[str]]:
    """Check whether the prompt text triggers safety spec violation patterns.

    Heuristic: if spec text keywords appear in the prompt alongside imperative
    language, flag as elevated risk.
    """
    IMPERATIVE = re.compile(
        r"\b(帮我|告诉我|教我|怎么|如何|请|让我知道|help me|tell me|how to|teach me|show me)\b",
        re.IGNORECASE,
    )
    if not IMPERATIVE.search(text):
        return 0.0, []

    triggered: list[str] = []
    for spec in scenario.safety_specs:
        # Extract meaningful words from the spec text (≥ 4 chars)
        keywords = [w for w in re.findall(r"\w{4,}", spec.text) if w.isalpha()]
        if keywords and any(kw.lower() in text.lower() for kw in keywords[:5]):
            triggered.append(spec.spec_id)

    score = min(0.5 + 0.1 * len(triggered), 0.8) if triggered else 0.0
    return score, triggered


class RuleBasedRiskDetector:
    """Lightweight rule-based risk detector for MVP.

    Detects unsafe prompts without any LLM call.  Suitable for latency-
    sensitive online control.  Replace or ensemble with an LLMRiskDetector
    for higher recall on subtle violations.
    """

    def detect(self, text: str, scenario: Scenario) -> RiskAssessment:
        kw_score, kw_triggered = _keyword_score(text)
        spec_score, spec_triggered = _spec_risk_score(text, scenario)

        combined = max(kw_score, spec_score)
        triggered_all = kw_triggered + [f"spec:{s}" for s in spec_triggered]

        reason = ""
        if kw_triggered:
            reason = f"Keyword match (patterns: {', '.join(kw_triggered[:3])})"
        elif spec_triggered:
            reason = f"Safety spec keywords triggered: {', '.join(spec_triggered[:3])}"
        elif combined == 0.0:
            reason = "No risk patterns detected"

        return RiskAssessment(
            level=_score_to_level(combined),
            score=round(combined, 4),
            reason=reason,
            triggered_specs=spec_triggered,
            metadata={"kw_patterns": kw_triggered},
        )


def _score_to_level(score: float) -> RiskLevel:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.25:
        return "low"
    return "safe"
