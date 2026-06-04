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
│   └── user/             # platform user uploads only
│       └── <id>/
│           └── …
└── results/              # ResultStore
```

Builtin and user packages share the same self-contained shape; **`metadata.yaml` → `origin`** must match the parent folder (`builtin` or `user`).

## Bootstrap

```bash
./scripts/setup-data.sh
```

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
