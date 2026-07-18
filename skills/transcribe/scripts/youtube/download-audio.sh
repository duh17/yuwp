#!/bin/bash
# Download audio from YouTube video
# Usage: download-audio.sh <video-url> [output-file]

set -e

VIDEO_URL="$1"
OUTPUT_FILE="${2:-}"

if [ -z "$VIDEO_URL" ]; then
    echo "Usage: download-audio.sh <video-url> [output-file]" >&2
    echo "Example: download-audio.sh https://www.youtube.com/watch?v=VIDEO_ID output.wav" >&2
    exit 1
fi

# Generate output filename if not provided
if [ -z "$OUTPUT_FILE" ]; then
    VIDEO_ID=$(echo "$VIDEO_URL" | grep -oE '[a-zA-Z0-9_-]{11}' | head -1)
    OUTPUT_FILE="${VIDEO_ID:-video}.wav"
fi

echo "Downloading audio from: $VIDEO_URL" >&2

# Download audio with yt-dlp
# Use --remote-components to download latest JS challenge solver (required since ~March 2026)
# Prefer m4a when available, then fall back to any bestaudio, then any format with audio.
yt-dlp -f "bestaudio[ext=m4a]/bestaudio/best" -x --audio-format wav --audio-quality 0 \
    -o "${OUTPUT_FILE%.wav}.%(ext)s" \
    --no-playlist \
    --remote-components ejs:github \
    --extractor-args "youtube:player_client=web" \
    "$VIDEO_URL" 2>&1 | grep -E "^(Downloading|ERROR|\[download\]|\[info\]|WARNING:)" >&2 || true

# Handle case where yt-dlp outputs different extension
if [ ! -f "$OUTPUT_FILE" ]; then
    # Find the downloaded file and convert if needed
    BASE="${OUTPUT_FILE%.wav}"
    DOWNLOADED=$(ls "${BASE}".* 2>/dev/null | head -1)
    if [ -n "$DOWNLOADED" ] && [ "$DOWNLOADED" != "$OUTPUT_FILE" ]; then
        echo "Converting to WAV..." >&2
        ffmpeg -i "$DOWNLOADED" -ar 16000 -ac 1 "$OUTPUT_FILE" -y -loglevel error
        rm -f "$DOWNLOADED"
    fi
fi

if [ -f "$OUTPUT_FILE" ]; then
    echo "Saved to: $OUTPUT_FILE" >&2
    echo "$OUTPUT_FILE"
else
    echo "Error: Failed to download audio" >&2
    exit 1
fi
