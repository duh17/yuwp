---
name: transcribe
description: Transcribe local audio or YouTube/video audio with Yuwp's native `yuwp-asr` CLI. Use for plain text, JSON, SRT, or VTT output.
container: false
---

# Transcribe

Transcribe audio files with a single command. No wrappers, no API keys.

## Happy path

```bash
yuwp-asr transcribe recording.m4a
```

Outputs plain text to stdout. Add `--format json`, `srt`, or `vtt` when you need structured output.

If the binary isn't on your PATH, use the full path.

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
swift build -c release --product yuwp-asr
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

## YouTube and video

Fetch captions first. Fall back to audio download when captions are missing or the user wants higher-quality transcription.

### Captions (fast, existing)

```bash
mkdir -p /tmp/yuwp-video-transcripts
yt-dlp --skip-download --write-subs --write-auto-subs \
  --sub-lang en --sub-format srt --convert-subs srt \
  --remote-components ejs:github \
  --extractor-args 'youtube:player_client=android' \
  -o '/tmp/yuwp-video-transcripts/%(id)s.%(ext)s' '<url>'
```

### Audio download + Yuwp ASR (higher quality)

```bash
yt-dlp -f 'bestaudio[ext=m4a]/bestaudio/best' -x \
  --audio-format m4a --audio-quality 0 \
  --no-playlist --no-progress \
  --remote-components ejs:github \
  --extractor-args 'youtube:player_client=web' \
  -o '/tmp/yuwp-video-transcripts/%(id)s_audio.%(ext)s' '<url>'

yuwp-asr transcribe /tmp/yuwp-video-transcripts/<id>_audio.m4a --language en
```

Replace `<id>` with the YouTube video ID (the string after `?v=` in the URL).

### Quick clip from a long video

```bash
ffmpeg -y -ss 00:05:00 -t 00:00:30 \
  -i /tmp/yuwp-video-transcripts/<id>_audio.m4a \
  -vn -c:a copy /tmp/<id>_clip.m4a

yuwp-asr transcribe /tmp/<id>_clip.m4a --language en --format json
```

## Decision tree

```
Audio source?
├── Local file (.m4a, .wav, .mp3)
│   └── yuwp-asr transcribe <file> [--format json|srt|vtt]
└── YouTube / online video
    ├── Captions exist? → yt-dlp captions (fast)
    └── No captions / want HQ? → yt-dlp audio → yuwp-asr transcribe
```

## Notes

- The binary must live alongside `mlx.metallib` — if you move one, move both.
- If `yt-dlp` hits `429`, `403`, or missing-format errors, upgrade `yt-dlp` first, then retry.
- First run downloads the default model if none is saved. This can take minutes and produces no progress output.
- `yuwp-asr serve` runs a persistent server for streaming / HTTP ASR. Only reach for this when doing iterative work on the same audio.
