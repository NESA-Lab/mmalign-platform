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
| `visual_safety` | Native MMAlign image value-safety set with repository-local SVG assets |

Multimodal entries may add either `image_url` or an `image_path` relative to
the package directory. Local files are encoded as data URLs at load time, so
the benchmark remains self-contained and works with OpenAI-compatible vision
backends without hosting the assets separately.

User-uploaded datasets use the same layout under `data/benchmarks/user/<id>/` with `origin: user` in metadata.
