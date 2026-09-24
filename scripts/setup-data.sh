#!/usr/bin/env bash
# Populate gitignored data/ workspace: benchmarks/builtin + benchmarks/user + results.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$ROOT/data"
BUILTIN="$WORKSPACE/benchmarks/builtin"
USER="$WORKSPACE/benchmarks/user"
EXAMPLES_BUILTIN="$ROOT/examples/benchmarks/builtin"
EXAMPLES_USER="$ROOT/examples/benchmarks/user"

mkdir -p "$BUILTIN" "$USER" "$WORKSPACE/results"/{runs,samples,reviews}
if [[ ! -f "$WORKSPACE/results/index.json" ]]; then
  echo '[]' > "$WORKSPACE/results/index.json"
fi

if [[ ! -d "$EXAMPLES_BUILTIN" ]]; then
  echo "error: $EXAMPLES_BUILTIN not found" >&2
  exit 1
fi

for pkg in "$EXAMPLES_BUILTIN"/*/; do
  name="$(basename "$pkg")"
  if [[ ! -f "$pkg/metadata.yaml" ]]; then
    continue
  fi
  echo "Installing builtin/$name …"
  mkdir -p "$BUILTIN/$name"
  cp "$pkg/metadata.yaml" "$BUILTIN/$name/"
  for f in specifications.json prompts.json scenario.json; do
    if [[ -f "$pkg/$f" ]]; then
      cp "$pkg/$f" "$BUILTIN/$name/"
    fi
  done
  if [[ -d "$pkg/assets" ]]; then
    rm -rf "$BUILTIN/$name/assets"
    cp -R "$pkg/assets" "$BUILTIN/$name/"
  fi
done

# Seed bundled user benchmarks without replacing packages uploaded or edited locally.
if [[ -d "$EXAMPLES_USER" ]]; then
  for pkg in "$EXAMPLES_USER"/*/; do
    name="$(basename "$pkg")"
    if [[ ! -f "$pkg/metadata.yaml" || -e "$USER/$name" ]]; then
      continue
    fi
    echo "Installing user/$name …"
    cp -R "$pkg" "$USER/$name"
  done
fi

echo "Done. Workspace: $WORKSPACE"
echo "  builtin: $BUILTIN"
echo "  user:    $USER"
