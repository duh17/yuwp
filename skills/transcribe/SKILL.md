---
name: transcribe
description: Transcribe local audio, video files, or YouTube audio with Yuwp ASR. Use for plain text, JSON, SRT, VTT, or YouTube transcript fallback chains.
container: false
---

# Transcribe

Single transcription lane for local audio/video and YouTube. This skill owns the old `youtube-transcript` workflow.

## Choose the path

| Source | Command | Notes |
|---|---|---|
| Local audio/video | `yuwp-asr transcribe <file>` | Plain text by default; supports JSON/SRT/VTT |
| YouTube quick transcript | `{baseDir}/scripts/youtube/transcript.sh <url>` | Captions → yt-dlp captions → Yuwp audio ASR |
| YouTube high quality | `{baseDir}/scripts/youtube/transcript.sh <url> --hq` | Skips captions and uses local ASR |
| YouTube subtitles | `{baseDir}/scripts/youtube/transcript.sh <url> --srt` | Emits SRT with timestamps |

YouTube transcripts are cached in `/tmp/youtube-transcripts/`.

## Local audio happy path

```bash
yuwp-asr transcribe recording.m4a
```

Outputs plain text to stdout. Add `--format json`, `srt`, or `vtt` when structured output is needed.

If the binary is not on `PATH`, use the full path.

Fresh DMG install:

```bash
/Applications/Yuwp.app/Contents/MacOS/yuwp-asr transcribe recording.m4a
```

Repo checkout:

```bash
~/workspace/yuwp/.build/arm64-apple-macosx/release/yuwp-asr transcribe recording.m4a
```

Build it first if the repo-local binary is missing:

```bash
cd ~/workspace/yuwp
swift build --build-system native -c release --product yuwp-asr
bash scripts/build_mlx_metallib.sh release
```

## Formats

| Flag | Output | When to use |
|------|--------|-------------|
| *(default)* | plain text on stdout | quick read, piping into notes |
| `--format json` | JSON with segments | programmatic use, alignment metadata |
| `--format srt --output out.srt` | SRT subtitles | video subtitles, caption upload |
| `--format vtt --output out.vtt` | VTT subtitles | web video, HTML5 `<track>` |

JSON includes subtitle segments when the forced aligner model is installed locally. SRT and VTT require the aligner.

Add `--debug` to JSON output for per-chunk confidence scores and alignment detail.

## Overrides

```bash
# Specific model
yuwp-asr transcribe clip.m4a --model ~/models/Qwen3-ASR-0.6B-4bit

# Language hint (ISO 639-1)
yuwp-asr transcribe interview.wav --language en

# Write to file instead of stdout
yuwp-asr transcribe talk.mp3 --format srt --output /tmp/talk.srt
```

Without `--model`, yuwp-asr falls back to the saved app model, then its built-in default.

## YouTube workflow

Use the bundled wrapper instead of hand-rolling `yt-dlp` commands:

```bash
{baseDir}/scripts/youtube/transcript.sh "https://youtube.com/watch?v=VIDEO_ID"
{baseDir}/scripts/youtube/transcript.sh "https://youtube.com/watch?v=VIDEO_ID" --hq
{baseDir}/scripts/youtube/transcript.sh "https://youtube.com/watch?v=VIDEO_ID" --srt
```

Fallback order:

1. YouTube captions via `youtube-transcript-plus`.
2. `yt-dlp` auto-captions.
3. Audio download with `yt-dlp`, then local `yuwp-asr` transcription.

Set `YUWP_ASR_BIN` to override the ASR binary. Set `MLX_SERVER` only for the legacy SRT fallback endpoint.

## Quick clip from a long downloaded file

```bash
ffmpeg -y -ss 00:05:00 -t 00:00:30 \
  -i /tmp/yuwp-video-transcripts/<id>_audio.m4a \
  -vn -c:a copy /tmp/<id>_clip.m4a

yuwp-asr transcribe /tmp/<id>_clip.m4a --language en --format json
```

## Decision tree

```text
Audio source?
├── Local file (.m4a, .wav, .mp3, video with audio)
│   └── yuwp-asr transcribe <file> [--format json|srt|vtt]
└── YouTube URL
    └── scripts/youtube/transcript.sh <url> [--hq|--srt]
```

## Notes

- The binary must live alongside `mlx.metallib`; if one moves, move both.
- If `yt-dlp` hits `429`, `403`, or missing-format errors, upgrade `yt-dlp` first, then retry.
- First local ASR run may download the default model and take minutes.
- `yuwp-asr serve` runs a persistent server for streaming/HTTP ASR. Use it only for iterative work on the same audio.
