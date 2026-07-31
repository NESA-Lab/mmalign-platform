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

User-uploaded datasets use the same layout under `data/benchmarks/user/<id>/` with `origin: user` in metadata.
