#!/bin/bash
# Stress-test dictation start/stop via the real Yuwp hotkey path.
#
# This exercises:
#   hotkey -> DictationSession.start -> AudioCapture.start
#   hotkey -> DictationSession.stop  -> AudioCapture.stop
#
# Usage:
#   scripts/stress_dictation_toggle.sh [iterations]
#
# Environment:
#   STRESS_HOLD_MS=250                # keep capture active per cycle
#   STRESS_START_TIMEOUT_SEC=6        # wait for "Audio capture started"
#   STRESS_STOP_TIMEOUT_SEC=15        # wait for stop/finalization lines
#   STRESS_LOG_FILE=/tmp/yuwp.stress.log
#   STRESS_HOTKEY_OSA='key code 50 using control down'
#       AppleScript statement used to toggle dictation.
#       Default matches Yuwp's Ctrl+` binding.
#
# Optional input-selection flip (to stress selection path too):
#   STRESS_ALT_INPUT_UID=<CoreAudio UID>
#   (alternates system-default <-> device:<uid> each iteration)
set -euo pipefail

cd "$(dirname "$0")/.."

ITERATIONS="${1:-40}"
: "${STRESS_HOLD_MS:=250}"
: "${STRESS_START_TIMEOUT_SEC:=6}"
: "${STRESS_STOP_TIMEOUT_SEC:=15}"
: "${STRESS_LOG_FILE:=/tmp/yuwp.stress.log}"
: "${STRESS_HOTKEY_OSA:=key code 50 using control down}"

APP_BIN="/Applications/Yuwp.app/Contents/MacOS/Yuwp"

if [ ! -x "$APP_BIN" ]; then
    echo "[stress] Yuwp binary not found at $APP_BIN"
    exit 1
fi

send_hotkey() {
    osascript >/dev/null <<APPLESCRIPT
        tell application "System Events"
            $STRESS_HOTKEY_OSA
        end tell
APPLESCRIPT
}

log_lines() {
    if [ -f "$STRESS_LOG_FILE" ]; then
        wc -l < "$STRESS_LOG_FILE" | tr -d ' '
    else
        echo 0
    fi
}

wait_for_pattern_since() {
    local pattern="$1"
    local since_line="$2"
    local timeout_sec="$3"

    local deadline=$(( $(date +%s) + timeout_sec ))
    while [ "$(date +%s)" -le "$deadline" ]; do
        if [ -f "$STRESS_LOG_FILE" ]; then
            if tail -n +"$((since_line + 1))" "$STRESS_LOG_FILE" | rg -q "$pattern"; then
                return 0
            fi
        fi
        sleep 0.1
    done
    return 1
}

set_input_selection() {
    local iteration="$1"
    local alt_uid="${STRESS_ALT_INPUT_UID:-}"

    if [ -z "$alt_uid" ]; then
        return 0
    fi

    if (( iteration % 2 == 0 )); then
        defaults write com.yuwp.app audioInputSelection -string "system-default"
    else
        defaults write com.yuwp.app audioInputSelection -string "device:${alt_uid}"
    fi
}

echo "[stress] iterations=$ITERATIONS hold_ms=$STRESS_HOLD_MS"
echo "[stress] start_timeout_sec=$STRESS_START_TIMEOUT_SEC stop_timeout_sec=$STRESS_STOP_TIMEOUT_SEC"
echo "[stress] hotkey_osa=$STRESS_HOTKEY_OSA"
echo "[stress] log=$STRESS_LOG_FILE"

# Clean restart so log parsing is deterministic.
pkill -f '/Applications/Yuwp.app/Contents/MacOS/Yuwp' >/dev/null 2>&1 || true
pkill -f '/Applications/Yuwp.app/Contents/MacOS/yuwp-asr' >/dev/null 2>&1 || true
: > "$STRESS_LOG_FILE"
nohup "$APP_BIN" > "$STRESS_LOG_FILE" 2>&1 &

# Wait for provider ready.
base_line=$(log_lines)
if ! wait_for_pattern_since "STT provider ready" "$base_line" 30; then
    echo "[stress] Yuwp did not become ready"
    tail -n 120 "$STRESS_LOG_FILE" || true
    exit 1
fi

echo "[stress] Yuwp ready; starting loop"

for i in $(seq 1 "$ITERATIONS"); do
    set_input_selection "$i"

    start_line=$(log_lines)
    send_hotkey

    if ! wait_for_pattern_since "Audio capture started" "$start_line" "$STRESS_START_TIMEOUT_SEC"; then
        echo "[stress] FAIL iteration=$i: start timeout"
        tail -n 160 "$STRESS_LOG_FILE" || true
        exit 1
    fi

    sleep_sec=$(awk -v ms="$STRESS_HOLD_MS" 'BEGIN{printf "%.3f", ms/1000.0}')
    sleep "$sleep_sec"

    stop_line=$(log_lines)
    send_hotkey

    if ! wait_for_pattern_since "Audio capture stopped" "$stop_line" "$STRESS_STOP_TIMEOUT_SEC"; then
        echo "[stress] FAIL iteration=$i: stop timeout"
        tail -n 160 "$STRESS_LOG_FILE" || true
        exit 1
    fi

    if ! wait_for_pattern_since "Session stopped" "$stop_line" "$STRESS_STOP_TIMEOUT_SEC"; then
        echo "[stress] FAIL iteration=$i: no server finalization signal"
        tail -n 160 "$STRESS_LOG_FILE" || true
        exit 1
    fi

    if (( i % 10 == 0 )); then
        echo "[stress] progress: $i/$ITERATIONS"
    fi
done

if rg -q "Final result timeout|yuwp-asr serve exited unexpectedly|Failed to start audio engine|Audio capture failed to start" "$STRESS_LOG_FILE"; then
    echo "[stress] FAIL: detected error signatures in log"
    rg -n "Final result timeout|yuwp-asr serve exited unexpectedly|Failed to start audio engine|Audio capture failed to start" "$STRESS_LOG_FILE" || true
    exit 1
fi

echo "[stress] PASS: $ITERATIONS iterations"
