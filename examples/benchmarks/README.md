# Example benchmark packages (tracked)

Self-contained layout used under `data/benchmarks/builtin/` after `./scripts/setup-data.sh`.

Each package directory:

```text
<id>/
  metadata.yaml
  specifications.json
  prompts.json
```

User-uploaded datasets use the same layout under `data/benchmarks/user/<id>/` with `origin: user` in metadata.
