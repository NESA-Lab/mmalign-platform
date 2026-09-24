# MMAlign Platform Monorepo

Licensed under the [Apache License 2.0](LICENSE).

## Layout

```text
mmalign-platform/
├── configs/              # models, scenarios, methods, evaluators (tracked)
├── data/                 # platform workspace — gitignored except README
│   ├── benchmarks/
│   │   ├── builtin/      # self-contained packages (metadata.yaml each)
│   │   └── user/         # user-uploaded packages
│   └── results/
├── examples/benchmarks/{builtin,user}/  # tracked seed packages
├── packages/mmalign/
├── apps/{api,cli,webui}/
└── scripts/setup-data.sh
```

## Setup

```bash
pnpm install
uv sync --all-packages --extra dev
./scripts/setup-data.sh
```

Requires [pnpm](https://pnpm.io/) and [uv](https://docs.astral.sh/uv/) on `PATH`.

## Paths

| Purpose   | Dev (default)      | Deploy                                               |
| --------- | ------------------ | ---------------------------------------------------- |
| Workspace | `./data/`          | `MMALIGN_USE_XDG=1` → `~/.local/share/mmalign/data/` |
| Datasets  | `data/benchmarks/` | under workspace                                      |
| Results   | `data/results/`    | under workspace (or `MMALIGN_RESULTS_DIR`)           |

## Commands

CLI entry point lives in the `mmalign-cli` package (not `mmalign`).

```bash
pnpm dev    # API (:8000) + WebUI (Vite)
pnpm test   # core pytest + webui typecheck

uv run --package mmalign-cli mmalign seed --reset --scale small   # mock corpus (use large for denser demo)
uv run --package mmalign-cli mmalign benchmark -m mock -s education -n 2 --evaluator mock_judge
uv run --package mmalign-cli mmalign serve                        # API only; docs at /api/docs
```

For a real model (needs `OPENAI_API_KEY` or the env named in `configs/models.yaml`):

```bash
uv run --package mmalign-cli mmalign benchmark -m gpt-4.1 -s education
```

## Bundled SpecBench datasets

`./scripts/setup-data.sh` installs 14 SpecBench packages into
`data/benchmarks/user/`: 5 text scenarios (1,500 samples) and 9 image
scenarios (167 samples). Each image sample has its own generated JPEG in its
package's `assets/` directory; `prompts.json` references it by relative
`image_path`. The images are synthetic substitutes, **not** the original SIUO
images referenced in the source prompts. Their original references are retained
in `image_original_path` for provenance. Existing user packages are not
replaced when setup is rerun. Source dataset: SpecBench (Apache-2.0; see
`examples/benchmarks/user/LICENSE`).

Choose `mock` and `mock_judge` to exercise the interface and evaluation
pipeline offline. Mock output and scores are illustrative only. For actual
model responses and evaluation results, configure an API-backed model and an
appropriate evaluator in `configs/models.yaml` and `configs/evaluators.yaml`.

## Mock data

`mmalign seed` generates the entire demo corpus from a single seed: benchmark
packages (`data/benchmarks/builtin/`), runs / samples / review labels in the
`FileResultStore`, and `data/results/platform_stats.json`. Every dashboard
chart (overview metrics, heatmap, controller time-series, drift, cross-modal
consistency, review agreement) derives from this corpus — no endpoint hardcodes
its own numbers. Use `--scale small` for a quick lightweight set.
