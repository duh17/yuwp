---
name: yuwp-video-transcript
description: Fetch YouTube transcripts through yt-dlp or Yuwp. This skill should be used when asked to transcribe or summarize a YouTube video, or when testing Yuwp on long-form video audio.
container: false
---

# Yuwp Video Transcript

Transcribe a YouTube video with one repo-local script.

Prefer English captions for speed. Fall back to Yuwp audio transcription when captions are missing or when `--hq` is requested.

## Entry point

```bash
{baseDir}/transcript.py <video-url>
```

## Usage

```bash
{baseDir}/transcript.py <video-url>                # captions first, then Yuwp fallback
{baseDir}/transcript.py <video-url> --hq           # force Yuwp audio transcription
{baseDir}/transcript.py <video-url> --srt          # generate SRT via Yuwp
YUWP_SERVER_URL=http://192.168.1.20:9748 {baseDir}/transcript.py <video-url>
```

Write transcript output to stdout. Reuse cached files from `/tmp/yuwp-video-transcripts/`.

## Requirements

Install these host tools:

- `uv`
- `yt-dlp`
- `ffmpeg`
- a running Yuwp server (`Yuwp.app` or standalone `asr-server`)

Keep the script standalone. Do not add `node_modules`, local Python venvs, or extra package scaffolding.

## Fallback chain

For plain text output:

1. Try English YouTube subtitles with `yt-dlp`
2. Fall back to Yuwp `/v1/audio/transcriptions`

Use `--hq` to skip subtitles and force Yuwp audio transcription.

For `--srt`, always download audio and call Yuwp `/v1/audio/subtitles`.

## Server behavior

Default server URL:

```bash
http://localhost:9748
```

Override with:

```bash
YUWP_SERVER_URL=http://host:9748
```

Check `GET /v1/info` before sending audio.

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
