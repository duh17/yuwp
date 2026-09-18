#!/bin/bash
# One-time PyTorch → MLX conversion for AuK-Flash. Runtime inference is Swift/MLX only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN=""
for candidate in \
  "${ROOT}/.build/out/Products/Release/yuwp-tts" \
  "${ROOT}/.build/out/Products/Debug/yuwp-tts" \
  "${ROOT}/.build/debug/yuwp-tts"
do
  if [[ -x "$candidate" ]]; then
    BIN="$candidate"
    break
  fi
done
if [[ -z "$BIN" ]]; then
  echo "Build yuwp-tts first: swift build --product yuwp-tts" >&2
  exit 1
fi
exec "$BIN" convert-auk "$@"
