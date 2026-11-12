# Inference-Time Alignment Methods

This document summarizes the inference-time alignment and correction methods
integrated into MMAlign. These methods do not train model parameters and do not
require method-specific checkpoints. They operate by prompt augmentation,
test-time deliberation, candidate selection, prompt refinement, safety critique,
refusal calibration, or agent-style orchestration.

## Implementation Map

| Component | Path |
| --- | --- |
| Method implementations | `packages/mmalign/mmalign/methods/specbench_ttd.py` |
| Safety-defense implementations | `packages/mmalign/mmalign/methods/safety_defenses.py` |
| Public method exports | `packages/mmalign/mmalign/methods/__init__.py` |
| Config loader | `packages/mmalign/mmalign/loaders/methods.py` |
| Built-in registry | `packages/mmalign/mmalign/core/registries.py` |
| Default method config | `configs/methods.yaml` |
| API method catalog | `apps/api/mmalign_api/routes/catalog.py` |
| Unit tests | `packages/mmalign/tests/test_methods/test_methods.py` |
| Safety-defense tests | `packages/mmalign/tests/test_methods/test_safety_defenses.py` |

## Integrated Methods

| Method ID | Source / Motivation | Inference-time Pattern | Expected Metrics |
| --- | --- | --- | --- |
| `vanilla_specbench` | SpecBench AugmentPrompt | Recall scenario specifications, then answer | Total/SAR, Safety, Behavioral, Unsafe Rate |
| `align3_specbench` | SpecBench ALIGN3 / TTD | Behavior optimization, safety-guided refinement, holistic audit | Total/SAR, Safety, Behavioral, Unsafe Rate |
| `zero_think` | SpecBench TTD variant | Empty thinking prefix before answer | Total/SAR, Unsafe Rate |
| `more_think` | SpecBench TTD variant | Repeated rethink continuations before answer | Total/SAR, Behavioral |
| `best_of_n` | Test-time sampling | Generate N candidates and select the best scored answer | Total/SAR, Unsafe Rate, judge score |
| `tpo` | Test-time prompt optimization | Iteratively sample, score, and improve candidates | Total/SAR, Behavioral, judge score |
| `ipr_prompt_refine` | IPR-style prompt refinement | Iteratively rewrite a text-to-image prompt for safety | IP, Q16, InPro, CLIPScore, BLIP |
| `vlm_safety_guard` | VLM safety guard / LlavaGuard-style audit | Audit risk first, then produce guarded final answer | Unsafe Rate, safety pass rate |
| `overrefusal_check` | MOSSBench-style oversensitivity correction | Detect unnecessary refusal and revise if safe | Refusal Rate, helpfulness, Behavioral |
| `user_conditioned_safety_critic` | User-conditioned multimodal safety critic idea | Critique under user profile, then answer or rewrite | Personalized safety score, Unsafe Rate |
| `critic_revise_agent` | Agentic critic-revision workflow | Draft, critique, and revise | Total/SAR, Safety, Behavioral |
| `risk_adaptive_agent` | Agentic risk router | Assess risk, then route to light or strong method | Total/SAR, Unsafe Rate, latency |
| `multi_agent_consensus` | Agentic multi-review workflow | Safety, behavior, and refusal reviewers revise answer | Total/SAR, Safety, Refusal Rate |
| `self_reminder` | Self-Reminder (Xie et al., Nature MI 2023) | Responsibility reminder in system prompt and after the query | Unsafe Rate, Refusal Rate |
| `goal_prioritization` | Goal Prioritization (Zhang et al., ACL 2024) | Few-shot prompt that puts safety before helpfulness; returns `[Final response]` | Unsafe Rate, Refusal Rate |
| `adashield_static` | AdaShield-S (Wang et al., ECCV 2024) | Static defense prompt sandwiched as query + defense + query; checks image text | Unsafe Rate on structure-based (typographic) jailbreaks, Refusal Rate |
| `intention_analysis` | Intention Analysis (Zhang et al., 2024) | Turn 1 identifies the essential intention; turn 2 answers in the same dialogue | Unsafe Rate, Behavioral |
| `ecso` | ECSO (Gou et al., ECCV 2024) | Self-check own answer; if harmful, caption the image and answer from the caption without the image | Unsafe Rate, helpfulness on safe inputs |
| `jailguard` | JailGuard (Zhang et al., 2023) | Mutate text/image, measure response divergence, refuse detected attacks | Detection rate, Unsafe Rate, Refusal Rate |

## Method Details

### `vanilla_specbench`

Uses a SpecBench-style augmented prompt. The model is asked to recall relevant
safety and behavioral specifications before producing the final answer. This is
useful as a lightweight baseline that matches SpecBench's specification
alignment setting more closely than the platform's generic `vanilla` method.

Default config:

```yaml
vanilla_specbench:
  type: vanilla_specbench
  temperature: 0.2
```

### `align3_specbench`

Implements a three-stage SpecBench ALIGN3-style test-time deliberation flow:

1. Behavior optimization
2. Safety-guided refinement
3. Holistic specification audit

Each stage is a separate backend generation call. The final response comes from
the holistic audit stage, and intermediate stage outputs are stored in
`AlignedResponse.metadata["stages"]`.

Default config:

```yaml
align3_specbench:
  type: align3_specbench
  max_stage_tokens: 1024
  temperature: 0.2
```

### `zero_think`

Adds an empty thinking prefix before answer generation. This is a low-overhead
test-time deliberation baseline.

Default config:

```yaml
zero_think:
  type: zero_think
  temperature: 0.2
```

### `more_think`

Runs multiple thought-continuation calls before the final answer. This increases
inference cost linearly with `rethink_count`.

Default config:

```yaml
more_think:
  type: more_think
  rethink_count: 3
  temperature: 0.2
```

### `best_of_n`

Generates several candidate answers and selects the best candidate. If an
`AlignContext.reward_model` is provided, it is used for scoring. Otherwise the
method tries the backend's `score()` method. If no scoring backend is available,
it falls back to a deterministic heuristic so local demos still run.

Default config:

```yaml
best_of_n:
  type: best_of_n
  n: 4
  temperature: 0.7
```

### `tpo`

Runs iterative candidate sampling and selection. The previous best response is
fed back into the next round as an improvement target. Like `best_of_n`, it can
use a reward model, backend scorer, or heuristic fallback.

Default config:

```yaml
tpo:
  type: tpo
  iterations: 2
  sample_size: 3
  temperature: 0.7
```

### `ipr_prompt_refine`

Refines a text-to-image prompt at inference time. The current implementation
returns the refined prompt as text. When the platform has a dedicated image
generation backend, this method can be chained before image generation.

Optional context metadata:

```python
AlignContext(metadata={"visual_feedback": "description of prior generated image"})
```

Default config:

```yaml
ipr_prompt_refine:
  type: ipr_prompt_refine
  iterations: 3
  temperature: 0.2
```

### `vlm_safety_guard`

Runs a safety audit first, then uses the audit to produce a guarded final answer.
This is designed for multimodal requests where image context or visual risk may
matter, while still remaining compatible with the current text-producing backend
interface.

Default config:

```yaml
vlm_safety_guard:
  type: vlm_safety_guard
  temperature: 0.2
```

### `overrefusal_check`

Generates an initial answer, then checks whether it is an unnecessary refusal.
If the request appears safe, the method asks the backend to revise toward a
helpful answer while preserving safety.

Default config:

```yaml
overrefusal_check:
  type: overrefusal_check
  temperature: 0.2
```

### `user_conditioned_safety_critic`

Runs a safety critique conditioned on a user profile, then produces a final
answer or safe rewrite. This is the inference-time version of a personalized
safety alignment idea: it uses user information at runtime without training a
new model or adapter.

Optional context metadata:

```python
AlignContext(metadata={"user_profile": "teen user"})
```

Default config:

```yaml
user_conditioned_safety_critic:
  type: user_conditioned_safety_critic
  temperature: 0.2
```

### `critic_revise_agent`

Runs a three-step agent workflow:

1. Draft an answer under the scenario specifications.
2. Critique the draft for safety and behavioral compliance.
3. Revise the draft using the critique.

This is useful when a single-pass answer is likely to miss a subtle
specification, but full multi-candidate search is too expensive.

Default config:

```yaml
critic_revise_agent:
  type: critic_revise_agent
  temperature: 0.2
```

### `risk_adaptive_agent`

Runs a risk assessment first and chooses a cheaper or stronger path:

- LOW risk -> `vanilla_specbench`
- MEDIUM risk -> `vlm_safety_guard`
- HIGH risk -> `align3_specbench`

The selected delegated method and route are stored in
`AlignedResponse.metadata`.

Default config:

```yaml
risk_adaptive_agent:
  type: risk_adaptive_agent
  temperature: 0.2
```

### `multi_agent_consensus`

Runs a candidate answer through three reviewer roles:

- Safety reviewer
- Behavioral usefulness reviewer
- Refusal calibration reviewer

The final call revises the candidate using all reviewer outputs. This is a
single-backend implementation of a multi-agent consensus pattern, so it stays
compatible with MMAlign's existing backend interface.

Default config:

```yaml
multi_agent_consensus:
  type: multi_agent_consensus
  temperature: 0.2
```

## Training-free Safety Defenses

Six generic jailbreak defenses live in `methods/safety_defenses.py`. They differ
from the methods above in two ways:

- **They keep every image of the original request.** The specification-based
  methods rebuild a text-only prompt from `_extract_user_text`, so the model does
  not see the image; these defenses edit only the text of the last user turn.
- **They do not inject scenario specifications.** They are generic defenses, so
  they can be evaluated on any benchmark and combined with spec-based methods.

Prompt templates are copied from each official repository; deviations are listed below.

| Method | Calls | Official code | Deviations |
| --- | --- | --- | --- |
| `self_reminder` | 1 | github.com/yjw1029/Self-Reminder (`src/pia_defense.py`, template "Remind") | "ChatGPT" replaced by configurable `assistant_name` (default `AI assistant`); falls back to a user-text prefix when the backend has no system message |
| `goal_prioritization` | 1 | github.com/thu-coai/JailbreakDefense_GoalPriority (`utils/utils.py`, `priority`) | `[Internal thoughts]` is moved to `thought`; only `[Final response]` is returned |
| `adashield_static` | 1 | github.com/SaFo-Lab/AdaShield (`prompts/static_defense_prompt.txt`) | Adaptive AdaShield-A (needs an offline-optimised prompt pool) is not included; applied only to requests with an image unless `apply_without_image: true` (the prompt refers to the image) |
| `intention_analysis` | 2 | github.com/alphadl/SafeLLM_with_IntentionAnalysis (`demo/IA_demo.py`) | `match_user_language: true` (default) appends "Respond in the same language as the original query." to turn 2; set `false` for a strict reproduction |
| `ecso` | 2 (safe) / 4 (unsafe) | github.com/gyhdog99/ECSO (`llava/eval/model_mmsafe_tell_ask.py`) | Text-only requests: the unsafe branch regenerates with the harmless instruction and no caption |
| `jailguard` | `n_variants + 1` | github.com/shiningrain/JailGuard (`main_txt.py`, `main_img.py`) | Similarity uses backend `embed()` (cosine) or character trigrams instead of spaCy vectors. Default thresholds are re-calibrated (text 1.43, image 0.105; paper 0.02 / 0.025, see below); text policy uses punctuation insertion + random character insertion/replacement (no translation or targeted-insertion models); image mutation (rotate / blur / posterize) needs Pillow and data-URI images, otherwise only text is mutated |

Not added because they overlap existing methods: `compositional_guard` (joint
image-text risk audit, covered by `vlm_safety_guard`) and `figstep_guard`
(typographic jailbreak defense, covered by `adashield_static`).

Default config:

```yaml
self_reminder:
  type: self_reminder
  assistant_name: AI assistant
goal_prioritization:
  type: goal_prioritization
adashield_static:
  type: adashield_static
  apply_without_image: false
intention_analysis:
  type: intention_analysis
  match_user_language: true
ecso:
  type: ecso
  max_check_chars: 1927
jailguard:
  type: jailguard
  n_variants: 8
  text_threshold: 1.43
  image_threshold: 0.105
  concurrency: 4
```

Evaluate these defenses on both unsafe and safe inputs: generic jailbreak
defenses usually lower the unsafe rate at the cost of more refusals on benign
requests.

### Sampled evaluation (intern-s2-preview-35b, judge gpt-5.6-luna)

Stratified samples of `test/multimodal_value_safety` (180) and
`test/multimodal_spec_alignment` (50), each package's own judge prompt and
metrics; numbers below are on samples every method answered (provider-blocked
inputs excluded). Small samples: differences under ~10 points are within noise.

| Method | Value safety: unsafe-sample safe / safe-sample refusal | Spec alignment: unsafe-sample safe / safe-sample refusal |
| --- | --- | --- |
| `raw` | 70.6 / 0.0 | 50.0 / 0.0 |
| `self_reminder` | 73.5 / 0.0 | 62.5 / 0.0 |
| `goal_prioritization` | 67.6 / 1.2 | 54.2 / 12.0 |
| `intention_analysis` | 70.6 / 4.8 | 62.5 / 4.0 |
| `ecso` | 70.6 / 0.0 | 50.0 / 4.0 |
| `adashield_static` | 52.9 / 7.2 | 41.7 / 68.0 |
| `jailguard` (paper thresholds) | 58.8 / 63.9 | 58.3 / 64.0 |

- `self_reminder` and `intention_analysis` were the most reliable (equal or
  higher safety, little extra refusal).
- `adashield_static` asks for a bare "I am sorry", which these packages' safety
  rules (that require a safe alternative) count as violations and which raises
  refusals on image inputs; use with care.
- `ecso` lowered safe-sample quality on value safety (59 -> 45 pass rate):
  false self-check alarms answer from a caption without the image.

### JailGuard threshold calibration

The JailGuard decision only depends on `max_divergence`, so thresholds were
calibrated without a judge: divergences of 368 samples (calibration split not
used in the evaluation above, plus the evaluation split) were recorded with
intern-s2-preview-35b at temperature 0 and character-trigram similarity.

| Thresholds (text / image) | Calibration: benign false positives / unsafe detected | Evaluation: benign false positives / unsafe detected |
| --- | --- | --- |
| 0.02 / 0.025 (paper) | 59.4% / 67.5% | 64.2% / 65.0% |
| **1.43 / 0.105 (default, 95th pct of benign)** | 4.7% / 27.5% | 11.0% / 31.7% |
| 0.2 / 0.2 | 20.3% / 32.5% | 19.3% / 43.3% |

Separation is weak with character-trigram similarity (benign text divergence
median 0.045 but 95th percentile 1.43), so JailGuard is mainly a low-false-alarm
filter here. Re-calibrate thresholds for other models or when the backend
supports `embed()`.

## Quick Verification

Run the following commands from the repository root. Do not copy a
machine-specific absolute path into documentation or scripts.

Check that all methods can be loaded from config:

```powershell
@'
from mmalign.loaders.methods import load_method

names = [
    "vanilla_specbench",
    "align3_specbench",
    "zero_think",
    "more_think",
    "best_of_n",
    "tpo",
    "ipr_prompt_refine",
    "vlm_safety_guard",
    "overrefusal_check",
    "user_conditioned_safety_critic",
    "critic_revise_agent",
    "risk_adaptive_agent",
    "multi_agent_consensus",
    "self_reminder",
    "goal_prioritization",
    "adashield_static",
    "intention_analysis",
    "ecso",
    "jailguard",
]

for name in names:
    method = load_method(name, "configs/methods.yaml")
    print(name, "=>", type(method).__name__, method.name)
'@ | python -
```

Check API catalog visibility:

```powershell
@'
from fastapi.testclient import TestClient
from mmalign_api.app import create_app

client = TestClient(create_app())
resp = client.get("/api/v1/methods")
print("status", resp.status_code)
names = {method["name"] for method in resp.json()["methods"]}

for name in [
    "vanilla_specbench",
    "align3_specbench",
    "zero_think",
    "more_think",
    "best_of_n",
    "tpo",
    "ipr_prompt_refine",
    "vlm_safety_guard",
    "overrefusal_check",
    "user_conditioned_safety_critic",
    "critic_revise_agent",
    "risk_adaptive_agent",
    "multi_agent_consensus",
]:
    print(name, name in names)
'@ | python -
```

Run method tests:

```powershell
python -m pytest packages\mmalign\tests\test_methods\test_methods.py packages\mmalign\tests\test_core\test_registries.py -q
```

Run full backend tests:

```powershell
python -m pytest packages\mmalign\tests -q
```

## Evaluation Guidance

Use SpecBench-style scenarios for methods that answer user prompts:

- `vanilla_specbench`
- `align3_specbench`
- `zero_think`
- `more_think`
- `best_of_n`
- `tpo`
- `vlm_safety_guard`
- `overrefusal_check`
- `user_conditioned_safety_critic`
- `critic_revise_agent`
- `risk_adaptive_agent`
- `multi_agent_consensus`

Recommended metrics:

- Total/SAR
- Safety pass rate
- Behavioral pass rate
- Unsafe Rate
- Refusal Rate
- Over-refusal Rate
- Latency / backend call count

Agent-style methods are most useful when evaluating safety/helpfulness
trade-offs under different inference budgets. Track both quality metrics and
backend call count, because these methods intentionally spend extra inference
for better review or routing.

Use text-to-image safety benchmarks for `ipr_prompt_refine` and future image
generation chains:

- IP
- Q16
- InPro
- NudeNet
- CLIPScore
- BLIP

## Extension Candidates

These are still inference-time and can be added later:

| Candidate | Purpose |
| --- | --- |
| `multi_judge_consensus` | Candidate or judgment selection by multiple judge prompts |
| `risk_adaptive_deliberation` | Route low-risk prompts to cheap methods and high-risk prompts to stronger TTD |
| `safe_prompt_rewrite` | Lightweight prompt-only safety rewrite without generated-image feedback |

## Design Notes

- All integrated methods are inference-time only.
- No method trains or modifies model parameters.
- No method requires a local checkpoint path.
- Expensive methods increase backend calls per sample; configure candidate or
  iteration counts carefully for large benchmarks.
- Reward-model-based methods support optional reward backends through
  `AlignContext.reward_model`, but remain runnable without one.
- Agent-style methods are implemented as `InferenceAlignMethod` classes, so
  benchmark runners, API routes, and the WebUI can call them like ordinary
  methods.
