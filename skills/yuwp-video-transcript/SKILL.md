---
name: yuwp-video-transcript
description: Fetch YouTube transcripts through yt-dlp or the canonical yuwp-asr CLI. This skill should be used when asked to transcribe or summarize a YouTube video, or when testing Yuwp on long-form video audio.
container: false
---

# Yuwp Video Transcript

Transcribe a YouTube video with one repo-local script.

Prefer English captions for speed. Fall back to Yuwp's canonical `yuwp-asr` flow when captions are missing or when `--hq` is requested.

## Entry point

```bash
{baseDir}/transcript.py <video-url>
```

## Usage

```bash
{baseDir}/transcript.py <video-url>                          # captions first, then Yuwp batch fallback
{baseDir}/transcript.py <video-url> --hq                     # force local Yuwp audio transcription
{baseDir}/transcript.py <video-url> --srt                    # generate SRT via Yuwp batch transcription
{baseDir}/transcript.py <video-url> --model ~/models/Qwen3-ASR-1.7B-bf16
YUWP_ASR_BIN=/path/to/yuwp-asr {baseDir}/transcript.py <video-url>
```

Write transcript output to stdout. Reuse cached files from `/tmp/yuwp-video-transcripts/`.

## Requirements

Install these host tools:

- `uv`
- `yt-dlp`
- `ffmpeg`
- `swift`
- `bash`

The script uses the repo-local `yuwp-asr` binary plus sibling `mlx.metallib`.
When it needs local ASR, it will reuse a healthy local server if one is already running at `http://127.0.0.1:9748`; otherwise it auto-starts `yuwp-asr serve` on a temporary localhost port.
If the binary is missing, it auto-builds the canonical release CLI:

```bash
cd /path/to/yuwp
swift build -c release --product yuwp-asr
bash scripts/build_mlx_metallib.sh release
```

Keep the script standalone. Do not add `node_modules`, local Python venvs, or extra package scaffolding.

## Fallback chain

For plain text output:

1. Try English YouTube subtitles with `yt-dlp`
2. Fall back to Yuwp batch transcription via `yuwp-asr serve` + `/v1/audio/transcriptions`

Use `--hq` to skip subtitles and force local audio transcription.

For `--srt`, always download audio and request `response_format=srt` from the batch endpoint.
Timed output requires Yuwp's default forced aligner model to be present locally.

## Typical workflow

```bash
transcript=$({baseDir}/transcript.py "https://youtube.com/watch?v=VIDEO_ID")
# Summarize or analyze the transcript directly
```

## Cache reset

```bash
rm -rf /tmp/yuwp-video-transcripts/
```

## Notes

Prefer `--hq` for technical videos when auto-captions are sketchy.
Prefer `--srt` when timestamps matter.
`YUWP_ASR_BIN` overrides the binary path when you want to point at a different build.
`YUWP_SERVER_URL` lets you point the skill at an already-running remote or local Yuwp server instead of auto-starting one.
