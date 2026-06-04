#!/usr/bin/env bash
# Populate gitignored data/ workspace: benchmarks/builtin + benchmarks/user + results.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$ROOT/data"
BUILTIN="$WORKSPACE/benchmarks/builtin"
USER="$WORKSPACE/benchmarks/user"
EXAMPLES_BUILTIN="$ROOT/examples/benchmarks/builtin"
SPECBENCH_REF="$ROOT/../../refs/repos/SpecBench/data"

mkdir -p "$BUILTIN" "$USER" "$WORKSPACE/results"/{runs,samples,reviews}
if [[ ! -f "$WORKSPACE/results/index.json" ]]; then
  echo '[]' > "$WORKSPACE/results/index.json"
fi

# ── Builtin: native packages (full copy from examples) ─────────────────────
if [[ -d "$EXAMPLES_BUILTIN" ]]; then
  for pkg in "$EXAMPLES_BUILTIN"/*/; do
    name="$(basename "$pkg")"
    if [[ ! -f "$pkg/metadata.yaml" ]]; then
      continue
    fi
    echo "Installing builtin/$name …"
    mkdir -p "$BUILTIN/$name"
    cp "$pkg/metadata.yaml" "$BUILTIN/$name/"
    for f in specifications.json prompts.json; do
      if [[ -f "$pkg/$f" ]]; then
        cp "$pkg/$f" "$BUILTIN/$name/"
      fi
    done
  done
else
  echo "warning: $EXAMPLES_BUILTIN not found" >&2
fi

# ── Builtin: SpecBench scenes (metadata from examples + data symlinks) ───────
if [[ -d "$SPECBENCH_REF" ]]; then
  for scene in Biochem Child Code Health Travel; do
    id="$(echo "$scene" | tr '[:upper:]' '[:lower:]')"
    meta_src="$EXAMPLES_BUILTIN/$id/metadata.yaml"
    if [[ ! -f "$meta_src" ]]; then
      echo "warning: no metadata for $id" >&2
      continue
    fi
    target="$BUILTIN/$id"
    mkdir -p "$target"
    cp "$meta_src" "$target/metadata.yaml"
    for f in specifications.json prompts.json scenario.json; do
      rm -f "$target/$f"
      ln -sf "$SPECBENCH_REF/$scene/$f" "$target/$f" 2>/dev/null || true
    done
    echo "Linked builtin/$id → SpecBench/$scene"
  done
else
  echo "warning: SpecBench not found at $SPECBENCH_REF" >&2
fi

echo "Done. Workspace: $WORKSPACE"
echo "  builtin: $BUILTIN"
echo "  user:    $USER (empty until uploads)"
