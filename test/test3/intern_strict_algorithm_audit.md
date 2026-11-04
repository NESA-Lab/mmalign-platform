# Intern algorithm consistency audit

## Sources inspected

- Local `SpecBench.tar.zst`: `specbench/ttd.py`, `specbench/prompt_ttd.py`.
- IPR official repository: https://github.com/ku-dmlab/IPR
- Inspected IPR commit: `305036643eda2ce88f793456fabac64755a38eda`.
- IPR control flow: `eval.py`, prompt/image review, `keep`, regenerate changed prompts.

## Best-of-N

The supplied SpecBench code uses N=15 by default and a local
FsfairX-LLaMA3-RM-v0.1 reward model. It also contains a heuristic fallback
if reward weights cannot load. MMAlign defaults to N=4; earlier smoke checks
used that heuristic, not the reward model.

Strict adaptation uses `AlignContext(reward_model=scorer,
metadata={"require_reward_model": True})`. Every candidate is scored,
nonfinite scores fail, and scorer failure propagates instead of falling back.
The live verification script uses N=3 to bound cost and an independent
gpt-6.1-sol LLM judge returning validated scores in [0,1]. Thus it verifies
reward-scored selection structure but does not reproduce the original reward
checkpoint, candidate budget, or paper results. Full-budget testing can use N=15.

## IPR

Previously `ipr_prompt_refine` repeated text rewrites with an optional static
feedback string. That is only an IPR-inspired component.

The optional image-feedback controller is enabled by:

```python
AlignContext(metadata={
    "ipr_closed_loop": True,
    "image_generator": generate_image_async,
})
```

The callback receives `(prompt, index)` and returns a real image URL or path.
The controller generates the initial image, gives the prompt and image to the
VLM, stops on `keep` (also accepts `[keep]`), or regenerates from the changed
prompt and repeats. It fails if no generator exists. It retains original user
intent, prompts, decisions and image references in its result metadata.
If the iteration budget ends after regeneration, the new final image is
explicitly marked unreviewed; completion is not a safety pass.

Live verification uses the Intern API as reviewer and an external image API
as generator. Upstream uses trained checkpoints and Stable Diffusion, with
Q16/NudeNet/BLIP-related evaluation. Those weights, training, and metrics are
not reproduced by API workflow adaptation and must not be claimed as such.

## Other method equivalence limits

- `raw`: structurally direct backend generation.
- `vanilla`: generic scenario-system-prompt implementation, distinct from the
  upstream user-prompt template.
- `vanilla_specbench`: scenario rules plus a user request; current template is
  semantically related but not byte-identical to the archived Vanilla prompt.
- `self_refine`: critique/revise flow preserved; prompt templates, iteration
  count, API generation limits and model differ from upstream experiments.

Image-retention regression tests cover the prompt rebuilding paths. They do
not establish numerical reproducibility or safety effectiveness.
