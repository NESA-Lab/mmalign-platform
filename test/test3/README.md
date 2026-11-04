# Intern-S2 inference adaptation

This directory contains the MMAlign inference-time adaptation package for the
official `intern-s2-preview-35b` API. It covers the 17 inference entries:

`raw`, `vanilla`, `align3`, `self_refine`, `vanilla_specbench`,
`align3_specbench`, `zero_think`, `more_think`, `best_of_n`, `tpo`,
`ipr_prompt_refine`, `vlm_safety_guard`, `overrefusal_check`,
`user_conditioned_safety_critic`, `critic_revise_agent`,
`risk_adaptive_agent`, and `multi_agent_consensus`.

DPO and SFT are excluded because they require local weight training and are not
API-only adaptations.

## Adaptation points

- `source/mmalign/methods/`: image-preserving prompt reconstruction, four-stage
  deliberation, native thinking variants, candidate selection, safety guards,
  critic/revision agents and risk routing.
- `source/mmalign/backends/intern.py`: Intern API backend. It supports the
  official `thinking_mode` request field, local serving's
  `chat_template_kwargs.enable_thinking`, a minimum output budget of 16384,
  empty-answer rejection and truncated-answer detection.
- `configs/models_intern_s2.yaml`: API and local-serving model entries.
- `configs/methods.yaml`: method parameters used by the runner.

Best-of-N and TPO require an injected reward model with
`AlignContext(metadata={"require_reward_model": True})` for strict scoring.
IPR's closed image loop requires a real `image_generator(prompt, index)`
callback. Without those dependencies the methods fail explicitly in strict
mode; they do not silently claim a valid score or image review.

## Run the visual flow check

Keep credentials outside this repository. The JSON file must contain `intern`
and `judge` keys. The script expects two control images named
`intern-red.png` and `intern-green.png` in the image directory.

```bash
PYTHONPATH=/path/to/mmalign/packages/mmalign \
python test/test3/scripts/verify_intern_remaining_methods.py \
  --credentials /secure/api_credentials.json \
  --images /path/to/control-images \
  --output /tmp/intern-test3-results \
  --concurrency 2
```

The real-scene script accepts `--methods` to run a subset and
`--enhanced-final-audit` to opt into the experimental final review. Use a new
output directory when changing that flag. It preserves per-stage traces and
can retry malformed evaluator JSON without regenerating valid model outputs.

## Evidence

The implementation was tested with real Intern API calls and two control
images. The final adapter/backend regression suite passed 96 tests. This is
flow, image-retention and interface evidence; it is not a formal task metric,
not a reproduction of upstream training checkpoints, and not a DPO/SFT result.

No credentials, generated images, API responses or private result files are
included in this directory.
