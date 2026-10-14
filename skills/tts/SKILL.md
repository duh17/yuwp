---
name: tts
description: "Local speech with Yuwp TTS (Qwen3-TTS speak; AuK clone, instruct, edit) to WAV/PCM on Apple Silicon. Not Grok cloud TTS (grok-media) or transcription."
container: false
---

# TTS

Local speech from `yuwp-tts`. Qwen3-TTS is the ordinary low-latency lane. AuK is the expressive create/clone/edit backend. One process loads one model directory.

## Choose the path

| Ask | Backend | Model dir |
|---|---|---|
| Speak this / named speaker | Qwen3-TTS CustomVoice | HF snapshot of `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Describe a voice, no clip | Qwen3-TTS VoiceDesign | HF snapshot of `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Clone, emotion, or edit existing audio | AuK-Flash (faster) or AuK Base (quality) | `~/Library/Application Support/Yuwp/models/auk-flash-mlx` or `auk-base-mlx` |
| Directed, expressive speech or multi-voice scenes (`[laughs]`, `<whisper>…</whisper>`, `<\|speaker:N\|>`) | Fish Audio Drama 3 | `~/Library/Application Support/Yuwp/models/drama3-mlx-8bit` (38 GB) |
| Cloud Grok / xAI TTS | stop — use `grok-media` | |

## Binary

Prefer the repo Release binary. The app-bundle copy can lag (no AuK until the next DMG).

```bash
YUWP_TTS=~/workspace/yuwp/.build/out/Products/Release/yuwp-tts
# fallbacks: `yuwp-tts` on PATH, then /Applications/Yuwp.app/Contents/MacOS/yuwp-tts
```

Build if missing:

```bash
cd ~/workspace/yuwp
swift build -c release --product yuwp-tts
bash scripts/build_mlx_metallib.sh release
```

`--model` must be a directory that contains `config.json` (an HF snapshot, not the hub repo root). The binary must sit next to `mlx.metallib`.

## Ordinary speak (Qwen)

```bash
QWEN_CV=$(ls -td "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots"/*/ | head -1)
mkdir -p .pi/tts
"$YUWP_TTS" \
  --model "$QWEN_CV" \
  --text "Hello from Yuwp." \
  --voice ryan \
  --language English \
  --out .pi/tts/hello.wav
```

CustomVoice speakers (matched case-insensitively): `ryan`, `serena`, `vivian`, `uncle_fu`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`.

`--language` is matched lowercased. Keys include `english`, `chinese`, `japanese`, `korean`, `french`, `german`, `spanish`. Default `English`.

VoiceDesign uses the same CLI: `--voice` is a free-text description, and `--model` is the VoiceDesign snapshot. Qwen clone uses `--ref-audio` plus `--ref-text`.

## Clone / instruct / edit (AuK)

Instruct TTS without a reference needs `--gen-seconds`. Clone or edit pass `--ref-audio` (alias `--audio`).

```bash
"$YUWP_TTS" \
  --model "$HOME/Library/Application Support/Yuwp/models/auk-base-mlx" \
  --instruction "Say the following with the same voice: 'Hello from Yuwp.'" \
  --ref-audio ref.wav \
  --gen-seconds 4 \
  --seed 11 \
  --out .pi/tts/clone.wav
```

Designed voice, no reference:

```bash
"$YUWP_TTS" \
  --model "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx" \
  --instruction "A warm, calm male narrator. Welcome home, how was work today?" \
  --gen-seconds 4 \
  --out .pi/tts/instruct.wav
```

Edit existing audio:

```bash
"$YUWP_TTS" \
  --model "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx" \
  --audio original.wav \
  --instruction "Replace 'Tuesday' with 'Thursday' while preserving the speaker and delivery." \
  --task edit \
  --gen-seconds 8 \
  --out .pi/tts/edited.wav
```

Base-only knobs (ignored on Flash): `--nfe` default 32, `--cfg` 2.0, `--sway` -1.0. Flash is fixed at four steps, CFG off.

`--task tts` enables sentence-preserving auto-chunk. Edits reject `--auto-chunk`. Default inter-chunk silence is 200 ms (`--chunk-pause-ms`).

If converted weights are missing:

```bash
"$YUWP_TTS" convert-auk \
  --src "$HOME/Library/Application Support/Yuwp/models/AuK-Flash" \
  --thinker-src "$HOME/Library/Application Support/Yuwp/models/Qwen2.5-Omni-3B" \
  --out "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx"
```

Base can reuse Flash's thinker as `--thinker-src`. Runtime inference is Swift/MLX and does not call Python.

## HTTP (only when iterating)

Default `127.0.0.1:7937`. Do not kill a server you did not start. Do not use 7936 (ASR). For tests, pick an isolated port.

```bash
"$YUWP_TTS" serve --transport http --model <model-dir> --host 127.0.0.1 --port 7937
curl -sf http://127.0.0.1:7937/v1/info
# POST /v1/audio/speech → WAV
# POST /v1/audio/speech/stream → NDJSON metadata/audio/done/error, pcm_s16le base64
```

`GET /v1/info` reports `backend` `auk-flash` or `auk-base` plus `variant` and sampling defaults.

Drama 3 serves on its own port (convention `7938`) with `--transport http`. Requests name saved voices (`voice`, or `voices` where voice *i* speaks `<|speaker:i|>`); a voice needs reference audio plus `reference_text`, which Qwen VoiceDesign previews provide. `/v1/info` reports `backend: drama3`, 44.1 kHz.

## Output

Write WAV under the current workspace `.pi/tts/` (gitignored). Render with an Oppi audio embed:

```text
![[.pi/tts/hello.wav]]
```

Do not `--play` unless the user asked to hear it at the desk. Headless/away: file + embed only.

Do not run ASR on the result unless the user asked to check the words.

## Decision tree

```text
Speak?
├── Cloud Grok/xAI → grok-media
├── Ordinary read-aloud, named speaker → Qwen CustomVoice
├── Invented voice, no clip → Qwen VoiceDesign (or AuK instruct + --gen-seconds)
└── Clone / emotion / edit existing audio → AuK
    ├── Faster / 4-step → auk-flash-mlx
    └── Quality / NFE-CFG-sway → auk-base-mlx
```

## Notes

- Sampling is stochastic at default temperature/top-p. Repeat with `--seed` (AuK) or `--temperature 0` (Qwen).
- Qwen is ~5 GB. AuK-Flash fp32 is ~20 GB. Do not load both at once.
- App-bundle `yuwp-tts` can be older than repo Release; AuK needs the post-`feat/auk-*` binary.
- Long-form auto-chunk is TTS only. Edit/enhance/separate stays one instruction plus source audio. Forcing auto-chunk on those tasks is HTTP 400 `auto_chunk_unsupported`.
- Do not classify AuK Base as Qwen3-TTS. Backend is detected from the model directory.
- First Qwen run may download weights. AuK conversion is one-time from official PyTorch dirs.
