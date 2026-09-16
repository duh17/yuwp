#!/bin/bash
# Stress-test AudioCapture start/stop and input-selection switching.
#
# Usage:
#   scripts/stress_audio_capture.sh [rounds]
#
# Environment overrides:
#   AUDIO_STRESS_ITERATIONS        (default: 60)
#   AUDIO_STRESS_HOLD_MS           (default: 250)
#   AUDIO_STRESS_SETTLE_MS         (default: 50)
#   AUDIO_STRESS_START_TIMEOUT_MS  (default: 4000)
#   AUDIO_STRESS_STOP_TIMEOUT_MS   (default: 4000)
set -euo pipefail

cd "$(dirname "$0")/.."

ROUNDS="${1:-1}"
: "${AUDIO_STRESS_ITERATIONS:=60}"
: "${AUDIO_STRESS_HOLD_MS:=250}"
: "${AUDIO_STRESS_SETTLE_MS:=50}"
: "${AUDIO_STRESS_START_TIMEOUT_MS:=4000}"
: "${AUDIO_STRESS_STOP_TIMEOUT_MS:=4000}"

echo "[audio-stress] rounds=$ROUNDS"
echo "[audio-stress] iterations=$AUDIO_STRESS_ITERATIONS hold_ms=$AUDIO_STRESS_HOLD_MS settle_ms=$AUDIO_STRESS_SETTLE_MS"
echo "[audio-stress] start_timeout_ms=$AUDIO_STRESS_START_TIMEOUT_MS stop_timeout_ms=$AUDIO_STRESS_STOP_TIMEOUT_MS"
echo "[audio-stress] Note: requires microphone permission for swift test"

if ! swift -e 'import Testing; print("ok")' >/dev/null 2>&1; then
    echo "[audio-stress] Swift Testing module is unavailable in this toolchain."
    echo "[audio-stress] Use scripts/stress_dictation_toggle.sh instead (app-level stress)."
    exit 2
fi

for round in $(seq 1 "$ROUNDS"); do
    echo "[audio-stress] round $round/$ROUNDS"
    AUDIO_STRESS_TEST=1 \
    AUDIO_STRESS_ITERATIONS="$AUDIO_STRESS_ITERATIONS" \
    AUDIO_STRESS_HOLD_MS="$AUDIO_STRESS_HOLD_MS" \
    AUDIO_STRESS_SETTLE_MS="$AUDIO_STRESS_SETTLE_MS" \
    AUDIO_STRESS_START_TIMEOUT_MS="$AUDIO_STRESS_START_TIMEOUT_MS" \
    AUDIO_STRESS_STOP_TIMEOUT_MS="$AUDIO_STRESS_STOP_TIMEOUT_MS" \
    swift test --build-system native --filter AudioCaptureStressTests

done

echo "[audio-stress] PASS ($ROUNDS rounds)"
