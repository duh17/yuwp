---
name: audio-transcribe
description: Transcribe local audio files or YouTube videos with Yuwp's canonical `yuwp-asr` CLI and helper scripts. Use this skill for local audio transcription, JSON/SRT/VTT generation, YouTube transcript fetches, or long-form batch transcription checks outside the app.
container: false
---

# Yuwp Transcribe

Transcribe local audio files or YouTube videos with the repo-local Yuwp tools.

Prefer this skill over older ad hoc MLX server flows. For summaries, transcribe first, then summarize the transcript directly in chat instead of adding another summarizer layer.

## Entry Points

```bash
{baseDir}/transcribe.py <audio-file>
{baseDir}/transcript.py <video-url>
```

## Local Audio Usage

```bash
# Plain text to stdout
{baseDir}/transcribe.py note.m4a

# JSON
{baseDir}/transcribe.py interview.wav --format json

# SRT / VTT
{baseDir}/transcribe.py talk.mp3 --format srt --output /tmp/talk.srt
{baseDir}/transcribe.py talk.mp3 --format vtt --output /tmp/talk.vtt

# Force a model or language hint
{baseDir}/transcribe.py clip.m4a --model ~/models/Qwen3-ASR-0.6B-4bit --language en

# Include chunk / alignment debug metadata in JSON
{baseDir}/transcribe.py sample.wav --format json --debug
```

## YouTube / Video Usage

```bash
{baseDir}/transcript.py <video-url>
{baseDir}/transcript.py <video-url> --hq
{baseDir}/transcript.py <video-url> --srt
{baseDir}/transcript.py <video-url> --model ~/models/Qwen3-ASR-1.7B-bf16
YUWP_ASR_BIN=/path/to/yuwp-asr {baseDir}/transcript.py <video-url>
```

## Requirements

Install these host tools:

- `swift`
- `bash`
- `uv`
- `yt-dlp` for YouTube transcript fetches
- `ffmpeg` for YouTube audio fallback / subtitle generation

The scripts use the repo-local release binary plus sibling `mlx.metallib`.
If either artifact is missing, they auto-build them with:

```bash
cd /path/to/yuwp
swift build -c release --product yuwp-asr
bash scripts/build_mlx_metallib.sh release
```

## Output Rules

- default output is plain text on stdout
- `--format` supports `text`, `json`, `srt`, `vtt`
- `json` includes subtitle segments when the forced aligner is installed locally
- `srt` and `vtt` require the default forced aligner model to be installed locally
- `--output <path>` writes the result to a file instead of stdout
- YouTube transcript output prefers English captions first, then falls back to Yuwp local ASR
- `--hq` skips captions and forces local audio transcription
- `--srt` on YouTube always goes through the batch endpoint and requires the aligner locally

## Notes

- `--model` is optional; Yuwp falls back to the saved app model, then its built-in default model spec
- `--language` is useful when the language is known and the audio is short or noisy
- `YUWP_ASR_BIN=/path/to/yuwp-asr` overrides the binary path
- if the binary is moved out of `.build/.../release/`, move `mlx.metallib` with it too
- `transcript.py` reuses a healthy local server if one is already running; otherwise it auto-starts `yuwp-asr serve` on a temporary localhost port
- YouTube transcript cache lives at `/tmp/yuwp-video-transcripts/`
