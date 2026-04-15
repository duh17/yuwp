---
name: audio-transcribe
description: Transcribe local audio files with Yuwp's canonical `yuwp-asr` CLI. This skill should be used when transcribing audio files (wav, mp3, m4a, flac, webm, etc.), generating JSON/SRT/VTT output, or checking Yuwp batch transcription quality outside the app.
container: false
---

# Yuwp Transcribe

Transcribe local audio files with the repo-local `yuwp-asr` CLI.

Prefer this skill over the old MLX server flow. For summaries, transcribe first, then summarize the transcript directly in chat instead of calling a separate summarizer script.

## Entry point

```bash
{baseDir}/transcribe.py <audio-file>
```

## Usage

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

## Requirements

Install these host tools:

- `swift`
- `bash`
- `uv`

The script uses the repo-local release binary plus sibling `mlx.metallib`.
If either artifact is missing, it auto-builds them with:

```bash
cd /path/to/yuwp
swift build -c release --product yuwp-asr
bash scripts/build_mlx_metallib.sh release
```

## Output rules

- default output is plain text on stdout
- `--format` supports `text`, `json`, `srt`, `vtt`
- `json` includes subtitle segments when the forced aligner is installed locally
- `srt` and `vtt` require the default forced aligner model to be installed locally
- `--output <path>` writes the result to a file instead of stdout

## Notes

- `--model` is optional; Yuwp falls back to the saved app model, then its built-in default model spec
- `--language` is useful when the language is known and the audio is short or noisy
- `YUWP_ASR_BIN=/path/to/yuwp-asr` overrides the binary path
- if the binary is moved out of `.build/.../release/`, move `mlx.metallib` with it too
