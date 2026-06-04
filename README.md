# MMAlign Platform Monorepo

## Layout

```text
mmalign-platform/
├── configs/              # models, scenarios, methods, evaluators (tracked)
├── data/                 # platform workspace — gitignored except README
│   ├── benchmarks/
│   │   ├── builtin/      # self-contained packages (metadata.yaml each)
│   │   └── user/         # user-uploaded packages
│   └── results/
├── examples/benchmarks/builtin/  # tracked seed packages
├── packages/mmalign/
├── apps/{api,cli,webui}/
└── scripts/setup-data.sh
```

## Setup

```bash
pnpm install
uv sync --package mmalign --extra dev
./scripts/setup-data.sh
```

## Paths

| Purpose | Dev (default) | Deploy |
|---------|---------------|--------|
| Workspace | `./data/` | `MMALIGN_USE_XDG=1` → `~/.local/share/mmalign/data/` |
| Datasets | `data/benchmarks/` | under workspace |
| Results | `data/results/` | under workspace (or `MMALIGN_RESULTS_DIR`) |

## Commands

```bash
pnpm dev
pnpm test
uv run --package mmalign-cli mmalign seed --reset --scale large   # generate the mock corpus
uv run --package mmalign mmalign benchmark -m gpt-4.1 -s education
uv run --package mmalign mmalign serve
```

## Mock data

`mmalign seed` generates the entire demo corpus from a single seed: benchmark
packages (`data/benchmarks/builtin/`), runs / samples / review labels in the
`FileResultStore`, and `data/results/platform_stats.json`. Every dashboard
chart (overview metrics, heatmap, controller time-series, drift, cross-modal
consistency, review agreement) derives from this corpus — no endpoint hardcodes
its own numbers. Use `--scale small` for a quick lightweight set.
