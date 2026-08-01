#!/bin/bash
set -euo pipefail

# Autoresearch measure script for Qwen3-ASR inference speed.
# Outputs METRIC name=value lines for the autoresearch loop.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_BIN="$REPO_ROOT/.build/arm64-apple-macosx/release/yuwp-asr"
STREAM_BIN="$REPO_ROOT/.build/arm64-apple-macosx/release/asr-stream-test"
MODEL="$HOME/Library/Application Support/Yuwp/models/mlx-community--Qwen3-ASR-0.6B-bf16"
FIXTURES="$REPO_ROOT/Tests/fixtures"
REPS=3

# --- Pre-checks (fast fail) ---

if [[ ! -x "$SERVER_BIN" ]]; then
    echo "ERROR: yuwp-asr release binary not found. Run: swift build -c release --product yuwp-asr" >&2
    exit 1
fi
if [[ ! -x "$STREAM_BIN" ]]; then
    echo "ERROR: asr-stream-test release binary not found. Run: swift build -c release --product asr-stream-test" >&2
    exit 1
fi

while IFS= read -r source; do
    for binary in "$SERVER_BIN" "$STREAM_BIN"; do
        if [[ "$REPO_ROOT/$source" -nt "$binary" ]]; then
            echo "ERROR: $(basename "$binary") is stale relative to $source. Rebuild release products first." >&2
            exit 1
        fi
    done
done < <(git -C "$REPO_ROOT" ls-files 'Sources/NativeASR/*.swift')

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: Model not found at $MODEL" >&2
    exit 1
fi

for f in jfk.wav asr_en.wav asr_zh.wav asr_en_long.wav; do
    if [[ ! -f "$FIXTURES/$f" ]]; then
        echo "ERROR: Fixture not found: $FIXTURES/$f" >&2
        exit 1
    fi
done

# --- Benchmark ---

TMPDIR_BENCH=$(mktemp -d /tmp/autoresearch-asr.XXXXXX)
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

# Warmup: compile Metal shaders (discard result)
"$SERVER_BIN" transcribe "$FIXTURES/jfk.wav" --model "$MODEL" --format json --language English >/dev/null 2>&1 || true

median_values() {
    sort -g | python3 -c '
import sys
values = [float(line) for line in sys.stdin]
mid = len(values) // 2
print(values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2)
'
}

run_fixture() {
    local name="$1" file="$2" lang="$3"
    local rtfs=() texts=()

    for rep in $(seq 1 $REPS); do
        local json
        json=$("$SERVER_BIN" transcribe "$file" --model "$MODEL" --format json --language "$lang" 2>/dev/null)

        local rtf text
        rtf=$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin)['rtf'])")
        text=$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")

        rtfs+=("$rtf")
        if [[ $rep -eq 1 ]]; then
            texts+=("$text")
        fi
    done

    # Median of REPS values
    local median_rtf
    median_rtf=$(printf '%s\n' "${rtfs[@]}" | median_values)

    echo "${name}_rtf=${median_rtf}"
    echo "${name}_text=${texts[0]}"
}

run_stream_fixture() {
    local file="$1"
    local rtfs=() prefill_ms=() reuse_pct=()

    for rep in $(seq 1 $REPS); do
        local report metrics
        report=$("$STREAM_BIN" "$file" --model "$MODEL" \
            --no-batch-retranscribe --compact 2>/dev/null)
        metrics=$(STREAM_REPORT="$report" python3 - <<'PY'
import json
import os
import statistics

report = json.loads(os.environ["STREAM_REPORT"])
streaming = report["streaming"]
duration = report["batch"]["audioDurationSec"]
chunks = streaming["chunks"]
print(streaming["elapsedSec"] / duration)
print(statistics.median(chunk["timing"]["prefillMs"] for chunk in chunks))
print(statistics.median(chunk["timing"]["reusePct"] for chunk in chunks))
PY
)
        rtfs+=("$(printf '%s\n' "$metrics" | sed -n '1p')")
        prefill_ms+=("$(printf '%s\n' "$metrics" | sed -n '2p')")
        reuse_pct+=("$(printf '%s\n' "$metrics" | sed -n '3p')")
    done

    echo "stream_rtf=$(printf '%s\n' "${rtfs[@]}" | median_values)"
    echo "stream_prefill_ms=$(printf '%s\n' "${prefill_ms[@]}" | median_values)"
    echo "stream_reuse_pct=$(printf '%s\n' "${reuse_pct[@]}" | median_values)"
}

# Run all fixtures
EN_RESULT=$(run_fixture "en" "$FIXTURES/jfk.wav" "English")
LONG_RESULT=$(run_fixture "long" "$FIXTURES/asr_en.wav" "English")
ZH_RESULT=$(run_fixture "zh" "$FIXTURES/asr_zh.wav" "Chinese")
XL_RESULT=$(run_fixture "xl" "$FIXTURES/asr_en_long.wav" "English")
STREAM_RESULT=$(run_stream_fixture "$FIXTURES/asr_en.wav")

# Parse results
EN_RTF=$(echo "$EN_RESULT" | grep "^en_rtf=" | cut -d= -f2)
EN_TEXT=$(echo "$EN_RESULT" | grep "^en_text=" | cut -d= -f2-)
LONG_RTF=$(echo "$LONG_RESULT" | grep "^long_rtf=" | cut -d= -f2)
ZH_RTF=$(echo "$ZH_RESULT" | grep "^zh_rtf=" | cut -d= -f2)
ZH_TEXT=$(echo "$ZH_RESULT" | grep "^zh_text=" | cut -d= -f2-)
XL_RTF=$(echo "$XL_RESULT" | grep "^xl_rtf=" | cut -d= -f2)
STREAM_RTF=$(echo "$STREAM_RESULT" | grep "^stream_rtf=" | cut -d= -f2)
STREAM_PREFILL_MS=$(echo "$STREAM_RESULT" | grep "^stream_prefill_ms=" | cut -d= -f2)
STREAM_REUSE_PCT=$(echo "$STREAM_RESULT" | grep "^stream_reuse_pct=" | cut -d= -f2)

# Compute overall median RTF across all batch fixtures
MEDIAN_RTF=$(printf '%s\n' "$EN_RTF" "$LONG_RTF" "$ZH_RTF" "$XL_RTF" | sort -g | python3 -c "
import sys
vals = [float(l) for l in sys.stdin]
n = len(vals)
mid = n // 2
if n % 2 == 0:
    print((vals[mid-1] + vals[mid]) / 2)
else:
    print(vals[mid])
")

# --- Output metrics ---

echo "METRIC median_rtf=$MEDIAN_RTF"
echo "METRIC en_rtf=$EN_RTF"
echo "METRIC long_rtf=$LONG_RTF"
echo "METRIC zh_rtf=$ZH_RTF"
echo "METRIC xl_rtf=$XL_RTF"
echo "METRIC stream_rtf=$STREAM_RTF"
echo "METRIC stream_prefill_ms=$STREAM_PREFILL_MS"
echo "METRIC stream_reuse_pct=$STREAM_REUSE_PCT"

# Quality: save transcripts for checks.sh comparison
echo "$EN_TEXT" > "$TMPDIR_BENCH/en_transcript.txt"
echo "$ZH_TEXT" > "$TMPDIR_BENCH/zh_transcript.txt"

# Emit transcripts as info (not metrics)
echo "INFO en_text=$EN_TEXT"
echo "INFO zh_text=$ZH_TEXT"
