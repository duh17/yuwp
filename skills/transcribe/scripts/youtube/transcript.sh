#!/bin/bash
# Get YouTube transcript (captions → yt-dlp → audio SRT fallback)
# Usage: transcript.sh <video-url> [--hq] [--srt]
#
# Output: Plain text transcript (or SRT with --srt) to stdout
# Cached in /tmp/youtube-transcripts/

set -e

VIDEO_URL=""
OUTPUT_DIR="/tmp/youtube-transcripts"
FORCE_HQ=false
OUTPUT_SRT=false
SKILL_DIR="$(cd -- "$(dirname "$0")" && pwd)"
MLX_SERVER="${MLX_SERVER:-http://localhost:9847}"
YUWP_ASR_BIN="${YUWP_ASR_BIN:-}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hq|--high-quality)
            FORCE_HQ=true
            shift
            ;;
        --srt)
            OUTPUT_SRT=true
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            VIDEO_URL="$1"
            shift
            ;;
    esac
done

if [ -z "$VIDEO_URL" ]; then
    echo "Usage: transcript.sh <video-url> [--hq] [--srt]" >&2
    echo "" >&2
    echo "Options:" >&2
    echo "  --hq   Force audio transcription (slower, better quality)" >&2
    echo "  --srt  Output SRT with timestamps (uses yuwp-asr, then MLX fallback)" >&2
    exit 1
fi

# Extract video ID
VIDEO_ID=$(echo "$VIDEO_URL" | grep -oE '[a-zA-Z0-9_-]{11}' | head -1)
if [ -z "$VIDEO_ID" ]; then
    VIDEO_ID="video_$(date +%s)"
fi

mkdir -p "$OUTPUT_DIR"
TRANSCRIPT_FILE="$OUTPUT_DIR/${VIDEO_ID}_transcript.txt"
SRT_FILE="$OUTPUT_DIR/${VIDEO_ID}_transcript.srt"

# Check cache first
if [ "$OUTPUT_SRT" = true ]; then
    if [ -f "$SRT_FILE" ] && [ -s "$SRT_FILE" ] && [ "$FORCE_HQ" = false ]; then
        cat "$SRT_FILE"
        exit 0
    fi
else
    if [ -f "$TRANSCRIPT_FILE" ] && [ -s "$TRANSCRIPT_FILE" ] && [ "$FORCE_HQ" = false ]; then
        cat "$TRANSCRIPT_FILE"
        exit 0
    fi
fi

# Ensure audio is downloaded (shared by both text and SRT paths)
ensure_audio() {
    AUDIO_FILE="$OUTPUT_DIR/${VIDEO_ID}_audio.wav"
    if [ ! -f "$AUDIO_FILE" ]; then
        echo "Downloading audio..." >&2
        if ! "$SKILL_DIR/download-audio.sh" "$VIDEO_URL" "$AUDIO_FILE" >/dev/null 2>&1; then
            echo "Failed to download audio" >&2
            return 1
        fi
    fi
    echo "$AUDIO_FILE"
}

resolve_yuwp_asr() {
    if [ -n "$YUWP_ASR_BIN" ] && [ -x "$YUWP_ASR_BIN" ]; then
        echo "$YUWP_ASR_BIN"
        return 0
    fi
    if command -v yuwp-asr >/dev/null 2>&1; then
        command -v yuwp-asr
        return 0
    fi
    if [ -x "/Applications/Yuwp.app/Contents/MacOS/yuwp-asr" ]; then
        echo "/Applications/Yuwp.app/Contents/MacOS/yuwp-asr"
        return 0
    fi
    if [ -x "$HOME/workspace/yuwp/.build/arm64-apple-macosx/release/yuwp-asr" ]; then
        echo "$HOME/workspace/yuwp/.build/arm64-apple-macosx/release/yuwp-asr"
        return 0
    fi
    echo "Error: yuwp-asr not found. Set YUWP_ASR_BIN or install Yuwp.app." >&2
    return 1
}

clean_srt_spacing() {
    # Clean up Chinese character spacing (SRT word-level alignment can add spaces between CJK chars)
    # "中 东 变 局" -> "中东变局" but keep spaces around non-CJK text
    python3 -c "
import sys, re
for line in sys.stdin:
    line = line.rstrip('\n')
    # Only process text lines (not timestamps or sequence numbers)
    if re.match(r'^\d+$', line.strip()) or '-->' in line or line.strip() == '':
        print(line)
    else:
        # Remove spaces between CJK characters
        cleaned = re.sub(r'([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])\s+([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])', r'\1\2', line)
        # May need multiple passes for consecutive CJK chars
        while re.search(r'([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])\s+([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])', cleaned):
            cleaned = re.sub(r'([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])\s+([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])', r'\1\2', cleaned)
        print(cleaned)
"
}

# Transcribe via Yuwp ASR — SRT with timestamps, falling back to the legacy MLX subtitles endpoint.
transcribe_srt() {
    local audio_file="$1"
    local asr_bin
    local tmp_srt

    if asr_bin=$(resolve_yuwp_asr 2>/dev/null); then
        echo "Generating SRT with yuwp-asr..." >&2
        tmp_srt=$(mktemp /tmp/youtube-transcript-srt.XXXXXX)
        if "$asr_bin" transcribe "$audio_file" --format srt --output "$tmp_srt" 2>/dev/null && [ -s "$tmp_srt" ]; then
            cat "$tmp_srt" | clean_srt_spacing
            rm -f "$tmp_srt"
            return 0
        fi
        rm -f "$tmp_srt"
        echo "yuwp-asr SRT failed; trying MLX subtitles endpoint..." >&2
    fi

    if ! curl -s "$MLX_SERVER/health" > /dev/null 2>&1; then
        echo "Error: yuwp-asr failed and MLX server is not running at $MLX_SERVER" >&2
        return 1
    fi

    echo "Generating SRT via MLX subtitles endpoint..." >&2
    local raw
    raw=$(curl -s -X POST "$MLX_SERVER/v1/audio/subtitles" \
        -F "file=@$audio_file" \
        -F "response_format=srt" \
        -F "llm_correct=false")

    if [ -z "$raw" ] || echo "$raw" | grep -q '"detail"'; then
        echo "SRT generation failed: $raw" >&2
        return 1
    fi

    echo "$raw" | clean_srt_spacing
}

# Transcribe via Yuwp ASR — plain text
transcribe_text() {
    local audio_file="$1"
    local asr_bin
    asr_bin=$(resolve_yuwp_asr) || return 1

    echo "Transcribing with yuwp-asr..." >&2
    if "$asr_bin" transcribe "$audio_file" 2>/dev/null; then
        return 0
    else
        echo "Transcription failed" >&2
        return 1
    fi
}

# --- SRT output path ---
if [ "$OUTPUT_SRT" = true ]; then
    # For SRT, always need audio
    AUDIO_FILE=$(ensure_audio) || exit 1
    transcribe_srt "$AUDIO_FILE" | tee "$SRT_FILE"
    exit 0
fi

# --- Text output path (with fallback chain) ---
if [ "$FORCE_HQ" = true ]; then
    AUDIO_FILE=$(ensure_audio) || exit 1
    transcribe_text "$AUDIO_FILE" | tee "$TRANSCRIPT_FILE"
elif "$SKILL_DIR/transcript.js" "$VIDEO_URL" 2>/dev/null | tee "$TRANSCRIPT_FILE" && [ -s "$TRANSCRIPT_FILE" ]; then
    # Got captions from YouTube API
    :
elif "$SKILL_DIR/transcript-yt-dlp.sh" "$VIDEO_URL" "$TRANSCRIPT_FILE" >/dev/null 2>&1 && [ -s "$TRANSCRIPT_FILE" ]; then
    # Got auto-subs via yt-dlp
    cat "$TRANSCRIPT_FILE"
else
    # Fall back to local audio transcription
    echo "Captions unavailable, using audio transcription..." >&2
    AUDIO_FILE=$(ensure_audio) || exit 1
    transcribe_text "$AUDIO_FILE" | tee "$TRANSCRIPT_FILE"
fi
