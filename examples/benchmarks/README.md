# Example benchmark packages (tracked)

Self-contained layout used under `data/benchmarks/builtin/` after `./scripts/setup-data.sh`.
No parent-monorepo checkout is required.

Each package directory:

```text
<id>/
  metadata.yaml
  specifications.json
  prompts.json
  scenario.json          # optional (SpecBench-sourced packages)
```

| Package | Source |
|---------|--------|
| `education`, `privacy` | Native MMAlign |
| `biochem`, `child`, `code`, `health`, `travel` | SpecBench scene data ([zzzhr97/SpecBench](https://github.com/zzzhr97/SpecBench)) |
| `visual_safety` | Small native MMAlign image value-safety smoke-test set |
| `visual_privacy` | 300 synthetic samples for privacy, identity, credential, and location safety |
| `visual_health` | 300 synthetic samples for medical grounding, triage, and hazardous products |
| `visual_physical_safety` | 300 synthetic samples for tools, weapons, chemicals, and operational hazards |
| `visual_minor_safety` | 300 synthetic samples for child privacy, profiling, tracking, and safe guidance |

The four expanded visual benchmarks contain 1,200 samples backed by 240
deterministic synthetic SVG assets. They share 24 stable specifications
(`VS-S01`–`VS-S12`, `VS-B01`–`VS-B12`). Asset-level train/validation/test
splits prevent variants of the same image from leaking across splits. Run
`python scripts/generate_multimodal_benchmarks.py` to reproduce the corpus.

Multimodal entries may add either `image_url` or an `image_path` relative to
the package directory. Local files are encoded as data URLs at load time, so
the benchmark remains self-contained and works with OpenAI-compatible vision
backends without hosting the assets separately.

User-uploaded datasets use the same layout under `data/benchmarks/user/<id>/` with `origin: user` in metadata.
