# Intern-S2 inference method adaptation

This extends the six-method integration to all seventeen inference entries
(including `raw`). DPO and SFT weight training are outside this integration.

## Runtime configuration

`configs/models.yaml` includes `intern-s2-preview-35b-api` with the `intern`
backend provider. `configs/models_intern_s2.yaml` also provides local serving
configuration. Set `INTERN_API_KEY` outside source
control for the official service. The local endpoint requires its own running
model service and credentials; these tests do not start a local weight server.

The adapter keeps the answer budget at least 16384 tokens, rejects empty final
answers and `length` termination, and records finish reasons and provider
reasoning flags. It never substitutes reasoning text for a final answer.

Official API thinking control uses `thinking_mode`; local serving uses
`chat_template_kwargs.enable_thinking`. These are separate configuration
styles. See the official API documentation:
https://internlm.intern-ai.org.cn/docEn/docs/Chat/
and the model card:
https://huggingface.co/internlm/Intern-S2-Preview

## Added method coverage

| Method | Integration behavior |
| --- | --- |
| align3 | Preserves original images in four stages and records actual call count. |
| align3_specbench | Preserves images in behavior, safety and final-audit stages. |
| zero_think | Disables native thinking without appending an assistant prefill. |
| more_think | Enables native thinking and executes the configured answer-recheck rounds. |
| tpo | Executes configured candidate rounds with injected strict reward scoring. |
| vlm_safety_guard | Visual safety audit followed by a guarded final answer. |
| overrefusal_check | Initial response followed by safety-aware refusal correction. |
| user_conditioned_safety_critic | User-profile critique followed by a tailored final answer. |
| critic_revise_agent | Draft, critique, revision, all grounded in the original image. |
| risk_adaptive_agent | Parses a leading LOW/MEDIUM/HIGH label; invalid routing output fails. |
| multi_agent_consensus | Three distinct reviewer roles and final synthesis using the same backend. |

Use `AlignContext(reward_model=scorer, metadata={"require_reward_model": True})`
for strict TPO and Best-of-N scoring. Missing, failed, or nonfinite scores cause
failure. Without this context, the existing demonstration fallback remains for
backward compatibility; it is not a formally validated reward model.

The `more_think` API mode is repeated answer checking with native thinking
enabled. It does not expose or append hidden reasoning, and does not promise
the original token-level thought-continuation mechanism. Multi-agent roles
share one model; this is not evidence of independently trained agents.

## Reproduction

```bash
python scripts/verify_intern_remaining_methods.py \
  --credentials /secure/api_credentials.json \
  --images /path/to/control-images \
  --output /path/to/results \
  --concurrency 2
```

The credentials file contains `intern` and `judge` keys and must remain outside
the repository. Control images are `intern-red.png` (red square) and
`intern-green.png` (green circle). The script loads production methods from
`configs/methods.yaml`, runs every configured stage and round, verifies an image
is retained on each call, and checks final color/shape descriptions. TPO uses
an independent image-aware LLM scorer and validates both round and global
selection. Outputs include per-stage requests, responses, reward records,
partial failures and a resumable summary.

Live results on 2026-11-03: 22/22 visual contrast cases completed and passed,
covering 66 real Intern generations and 12 independent TPO reward calls.
All stages retained their input image. The official service reported
`reasoning_used=false` for zero-think and `reasoning_used=true` for each
more-think recheck round. Method tests (60) and backend tests (22) passed.

These are flow, image-retention and visual contrast checks. They do not measure
benchmark safety improvements, official task acceptance, or reproduction of
upstream training/checkpoints. See `intern_strict_algorithm_audit.md` for the
previous Best-of-N and IPR equivalence audit.

## Real-scene follow-up, November 3-4

`scripts/test_intern_real_scenarios.py --methods ...` also supports the remaining
eleven methods plus a freshly generated raw baseline. Four matched real-image
cases produced 48 completed evaluations and 137 generation calls in the final
records, with 24 strict TPO reward calls. Image retention held throughout.
These tests expose factual errors, over-refusal and visual-grounding issues;
flow completion is separate from judged answer quality.

Risk parsing now accepts explicit `Risk: LOW` / `Risk level: MEDIUM` labels,
while still failing on unknown labels. LOW and MEDIUM were exercised in the
main comparison. A separate sports-risk diagnostic exercised HIGH and its
three-stage delegate. A different diagnostic rejected by provider moderation
is retained separately and is not counted as model-level safety success.

The evaluator retries malformed JSON only, preserves original failures and
reuses an existing generated answer when possible. Valid negative judgments
are not rerun to select better results. `summarize_intern_real_scenarios.py`
exports paired statistics and a human-review CSV with blank human decisions.
TPO rewards and final judgments share an evaluator model, so apparent gains
need independent human review. Updated method/backend tests: 85 passed.

## Optional final-audit experiment

Five methods support `AlignContext(metadata={"enhanced_final_audit": True})`:
align3, vlm_safety_guard, user_conditioned_safety_critic, critic_revise_agent,
and multi_agent_consensus. It reinjects complete specifications into the final
stage, asks the model to recheck visual evidence and technical claims, and
adds factual correctness to the behavior reviewer. Default behavior remains
the previous prompt flow; the enhancement is opt-in, not enabled globally.

The twenty-case retest on the same four development samples completed without
execution failures, but automatic task-success counts fell from 15/20 to 9/20;
safety counts changed from 18/20 to 16/20. Some visual-grounding judgments
improved, while technical errors persisted. This is not evidence of efficacy,
nor a causal estimate given sampling and evaluator variance. Both improvements
and regressions are preserved for human review. No negative valid judgment was
retried to choose a better answer.

Pass `--enhanced-final-audit` to `test_intern_real_scenarios.py` to reproduce
the enhanced mode with current code, and use a separate output directory.
The first retest used the same enhanced prompts before they were wrapped in
the opt-in switch; its metadata identifies `specifications_and_visual_grounding_v1`.
`compare_intern_retest.py --before OLD --after NEW` verifies matching inputs
and serialized image hashes across stages, and exports paired judgments and
side-by-side review CSVs. Updated method/backend tests: 96 passed.
