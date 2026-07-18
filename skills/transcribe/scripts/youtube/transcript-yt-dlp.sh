#!/bin/bash
# Fetch YouTube (auto) subtitles via yt-dlp and convert to plain text transcript.
# Usage: transcript-yt-dlp.sh <video-url> <output-txt>

set -euo pipefail

VIDEO_URL="${1:-}"
OUT_FILE="${2:-}"

if [ -z "$VIDEO_URL" ] || [ -z "$OUT_FILE" ]; then
  echo "Usage: transcript-yt-dlp.sh <video-url> <output-txt>" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

VIDEO_ID=$(yt-dlp --no-playlist --print id "$VIDEO_URL" 2>/dev/null | head -1)
if [ -z "$VIDEO_ID" ]; then
  echo "Error: could not determine video id" >&2
  exit 1
fi

# Prefer auto-captions in any language. If those are missing, yt-dlp will fail and caller can fallback.
# We use a stable output template so we can locate the resulting .srt file.
# --remote-components ejs:github downloads latest JS challenge solver (required since ~March 2026)
yt-dlp --no-playlist --skip-download \
  --write-auto-subs --sub-lang "en,zh-Hans,zh-Hant,zh,ja,ko,es,fr,de,ar,ru" --sub-format ttml \
  --convert-subs srt \
  --remote-components ejs:github \
  --extractor-args "youtube:player_client=web" \
  -o "$TMP_DIR/%(id)s.%(ext)s" \
  "$VIDEO_URL" >/dev/null

SRT_FILE=$(ls -1 "$TMP_DIR/${VIDEO_ID}"*.srt 2>/dev/null | head -1 || true)
if [ -z "$SRT_FILE" ] || [ ! -s "$SRT_FILE" ]; then
  echo "Error: no subtitles downloaded" >&2
  exit 1
fi

python3 - "$SRT_FILE" > "$OUT_FILE" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text_lines = []

# Very small SRT-to-text conversion:
# - drop numeric counters
# - drop timestamp lines
# - keep spoken text lines
for raw in path.read_text(errors="ignore").splitlines():
    line = raw.strip()
    if not line:
        continue
    if line.isdigit():
        continue
    if re.match(r"^\d{2}:\d{2}:\d{2}[,\.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,\.]\d{3}", line):
        continue
    # strip simple markup
    line = re.sub(r"<[^>]+>", "", line)
    text_lines.append(line)

# De-dup adjacent identical lines (common in captions)
out = []
prev = None
for l in text_lines:
    if l == prev:
        continue
    out.append(l)
    prev = l

sys.stdout.write("\n".join(out).strip() + "\n")
PY

if [ ! -s "$OUT_FILE" ]; then
  echo "Error: transcript conversion produced empty output" >&2
  exit 1
fi
