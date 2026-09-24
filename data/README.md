# Platform workspace (`data/`)

Gitignored runtime root. Layout:

```text
data/
├── benchmarks/
│   ├── builtin/          # shipped / setup-data examples (read-only via API)
│   │   └── <id>/
│   │       ├── metadata.yaml
│   │       ├── specifications.json
│   │       └── prompts.json
│   └── user/             # bundled SpecBench packages + platform user uploads
│       └── <id>/
│           └── …
└── results/              # ResultStore
```

Builtin and user packages share the same self-contained shape; **`metadata.yaml` → `origin`** must match the parent folder (`builtin` or `user`).

## Bootstrap

```bash
./scripts/setup-data.sh
```

The bootstrap script installs 14 tracked SpecBench packages from
`examples/benchmarks/user/`, preserving existing packages on repeat runs.
Images are synthetic substitutes for the original SIUO assets.

## Register a user dataset

```bash
curl -X POST http://localhost:8000/api/v1/benchmarks/user \
  -H 'Content-Type: application/json' \
  -d '{"name":"My set","safety_specifications":["…"],"behavioral_specifications":["…"],"prompts":[{"prompt":"hi","label":"safe"}]}'
```

## Overrides

| Variable | Effect |
|----------|--------|
| `MMALIGN_DATA_DIR` | Workspace root |
| `MMALIGN_RESULTS_DIR` | Only `results/` subtree |
| `MMALIGN_USE_XDG=1` | `~/.local/share/mmalign/data/` |
