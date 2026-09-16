#!/bin/bash
# Download YouTube transcript and summarize
# Usage: summarize.sh <video-url> [output-dir] [--hq]

set -e

VIDEO_URL=""
OUTPUT_DIR="/tmp/youtube-transcripts"
FORCE_HQ=false
SKILL_DIR="$(cd -- "$(dirname "$0")" && pwd)"
YUWP_ASR_BIN="${YUWP_ASR_BIN:-}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hq|--high-quality)
            FORCE_HQ=true
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [ -z "$VIDEO_URL" ]; then
                VIDEO_URL="$1"
            else
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$VIDEO_URL" ]; then
    echo "Usage: summarize.sh <video-url> [output-dir] [--hq]" >&2
    echo "" >&2
    echo "Options:" >&2
    echo "  --hq, --high-quality  Force Qwen-ASR transcription (slower, better quality)" >&2
    exit 1
fi

# Extract video ID for filename
VIDEO_ID=$(echo "$VIDEO_URL" | grep -oE '[a-zA-Z0-9_-]{11}' | head -1)
if [ -z "$VIDEO_ID" ]; then
    VIDEO_ID="video_$(date +%s)"
fi

mkdir -p "$OUTPUT_DIR"
TRANSCRIPT_FILE="$OUTPUT_DIR/${VIDEO_ID}_transcript.txt"
SUMMARY_FILE="$OUTPUT_DIR/${VIDEO_ID}_summary.md"

echo "=== YouTube Video Summarizer ===" >&2
echo "Video: $VIDEO_URL" >&2
echo "Output: $OUTPUT_DIR" >&2
if [ "$FORCE_HQ" = true ]; then
    echo "Mode: High-quality (Qwen-ASR)" >&2
fi
echo "" >&2

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
    if [ -x "$HOME/workspace/yuwp/.build/out/Products/Release/yuwp-asr" ]; then
        echo "$HOME/workspace/yuwp/.build/out/Products/Release/yuwp-asr"
        return 0
    fi
    if [ -x "$HOME/workspace/yuwp/.build/arm64-apple-macosx/release/yuwp-asr" ]; then
        echo "$HOME/workspace/yuwp/.build/arm64-apple-macosx/release/yuwp-asr"
        return 0
    fi
    echo "Error: yuwp-asr not found. Set YUWP_ASR_BIN or install Yuwp.app." >&2
    return 1
}

# Function to transcribe via audio
transcribe_audio() {
    echo "  Downloading audio..." >&2

    # Use output dir for audio to allow reuse
    AUDIO_FILE="$OUTPUT_DIR/${VIDEO_ID}_audio.wav"

    # Download audio if not already cached
    if [ ! -f "$AUDIO_FILE" ]; then
        if ! "$SKILL_DIR/download-audio.sh" "$VIDEO_URL" "$AUDIO_FILE" >/dev/null 2>&1; then
            echo "  ✗ Failed to download audio" >&2
            return 1
        fi
    else
        echo "  ✓ Using cached audio" >&2
    fi
    echo "  ✓ Audio ready" >&2

    echo "  Transcribing with yuwp-asr (~5 min for 1hr video)..." >&2
    local asr_bin
    asr_bin=$(resolve_yuwp_asr) || return 1
    if "$asr_bin" transcribe "$AUDIO_FILE" > "$TRANSCRIPT_FILE" 2>/dev/null; then
        echo "  ✓ Transcription complete" >&2
        return 0
    else
        echo "  ✗ Transcription failed" >&2
        return 1
    fi
}

# Step 1: Get transcript
echo "Step 1: Fetching transcript..." >&2

if [ "$FORCE_HQ" = true ]; then
    # Skip captions, go straight to audio
    if ! transcribe_audio; then
        exit 1
    fi
elif "$SKILL_DIR/transcript.js" "$VIDEO_URL" > "$TRANSCRIPT_FILE" 2>/dev/null && [ -s "$TRANSCRIPT_FILE" ]; then
    echo "  ✓ Got transcript from captions" >&2
elif "$SKILL_DIR/transcript-yt-dlp.sh" "$VIDEO_URL" "$TRANSCRIPT_FILE" >/dev/null 2>&1 && [ -s "$TRANSCRIPT_FILE" ]; then
    echo "  ✓ Got transcript from yt-dlp auto-captions" >&2
else
    echo "  ✗ Captions unavailable, falling back to audio..." >&2
    if ! transcribe_audio; then
        exit 1
    fi
fi

if [ ! -s "$TRANSCRIPT_FILE" ]; then
    echo "Error: Transcript is empty" >&2
    exit 1
fi

WORD_COUNT=$(wc -w < "$TRANSCRIPT_FILE" | tr -d ' ')
echo "  Transcript: $WORD_COUNT words" >&2
echo "  Saved to: $TRANSCRIPT_FILE" >&2
echo "" >&2

# Step 2: Summarize using the local ds4 text endpoint
echo "Step 2: Generating summary with DeepSeek V4 Flash..." >&2

PROMPT="Please provide a comprehensive summary of the following transcript. Include:

1. **Overview**: A brief 2-3 sentence overview of the content
2. **Key Topics**: Main topics/themes discussed with bullet points
3. **Key Points**: Important facts, statistics, or statements mentioned
4. **Notable Quotes**: Any significant quotes (if applicable)
5. **Conclusion**: Main takeaways

Format the output in clean Markdown.

---

TRANSCRIPT:

$(cat "$TRANSCRIPT_FILE")"

pi --provider ds4 --model deepseek-v4-flash \
   --no-tools \
   --print \
   --system-prompt "You are a professional content summarizer. Summarize content clearly and accurately, preserving key information. Respond in the same language as the transcript." \
   "$PROMPT" > "$SUMMARY_FILE" 2>/dev/null

if [ -s "$SUMMARY_FILE" ]; then
    echo "  ✓ Summary generated" >&2
    echo "  Saved to: $SUMMARY_FILE" >&2
    echo "" >&2
    echo "=== Summary ===" >&2
    cat "$SUMMARY_FILE"
else
    echo "  ✗ Failed to generate summary" >&2
    exit 1
fi
