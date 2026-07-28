#!/bin/bash
set -euo pipefail

# Autoresearch measure script for Qwen3-ASR inference speed.
# Outputs METRIC name=value lines for the autoresearch loop.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_BIN="$REPO_ROOT/.build/arm64-apple-macosx/release/yuwp-asr"
MODEL="$HOME/Library/Application Support/Yuwp/models/mlx-community--Qwen3-ASR-0.6B-bf16"
FIXTURES="$REPO_ROOT/Tests/fixtures"
REPS=3

# --- Pre-checks (fast fail) ---

if [[ ! -x "$SERVER_BIN" ]]; then
    echo "ERROR: yuwp-asr release binary not found. Run: swift build -c release --product yuwp-asr" >&2
    exit 1
fi

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
    median_rtf=$(printf '%s\n' "${rtfs[@]}" | sort -g | python3 -c "
import sys
vals = [float(l) for l in sys.stdin]
n = len(vals)
mid = n // 2
if n % 2 == 0:
    print((vals[mid-1] + vals[mid]) / 2)
else:
    print(vals[mid])
")

    echo "${name}_rtf=${median_rtf}"
    echo "${name}_text=${texts[0]}"
}

# Run all fixtures
EN_RESULT=$(run_fixture "en" "$FIXTURES/jfk.wav" "English")
LONG_RESULT=$(run_fixture "long" "$FIXTURES/asr_en.wav" "English")
ZH_RESULT=$(run_fixture "zh" "$FIXTURES/asr_zh.wav" "Chinese")
XL_RESULT=$(run_fixture "xl" "$FIXTURES/asr_en_long.wav" "English")

# Parse results
EN_RTF=$(echo "$EN_RESULT" | grep "^en_rtf=" | cut -d= -f2)
EN_TEXT=$(echo "$EN_RESULT" | grep "^en_text=" | cut -d= -f2-)
LONG_RTF=$(echo "$LONG_RESULT" | grep "^long_rtf=" | cut -d= -f2)
ZH_RTF=$(echo "$ZH_RESULT" | grep "^zh_rtf=" | cut -d= -f2)
ZH_TEXT=$(echo "$ZH_RESULT" | grep "^zh_text=" | cut -d= -f2-)
XL_RTF=$(echo "$XL_RESULT" | grep "^xl_rtf=" | cut -d= -f2)

# Compute overall median RTF across all fixtures
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

# Quality: save transcripts for checks.sh comparison
echo "$EN_TEXT" > "$TMPDIR_BENCH/en_transcript.txt"
echo "$ZH_TEXT" > "$TMPDIR_BENCH/zh_transcript.txt"

# Emit transcripts as info (not metrics)
echo "INFO en_text=$EN_TEXT"
echo "INFO zh_text=$ZH_TEXT"
