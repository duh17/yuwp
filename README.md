# Yuwp

System-wide voice dictation for macOS. Press a hotkey, talk, see your words appear in any text field.

Runs entirely on-device once a model is installed — no cloud APIs, no network requests during dictation. Audio never leaves your machine.

## How It Works

A Swift menu bar app captures microphone audio and streams it to a native ASR server running [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) via [MLX](https://github.com/ml-explore/mlx). Transcription results are injected into the focused text field in real time using the macOS Accessibility API.

```
Hotkey → Mic Capture → asr-server → Text Injection
         (16kHz PCM)   (MLX/Metal)   (AX API)
```

The server implements streaming optimizations — encoder window caching, decoder KV cache reuse, and prefix rollback — to achieve O(1) latency per audio chunk instead of reprocessing the entire recording each time.

## Requirements

- macOS 14+
- Apple Silicon (M1 or later)

Yuwp does not auto-download models. On a fresh Mac, download one explicitly from the menu bar app when you're ready (~1GB for the default 0.6B-4bit model).

## Install & Run

```bash
git clone https://github.com/duh17/yuwp.git
cd yuwp
scripts/run.sh
```

This builds the app and asr-server, creates a proper .app bundle with stable code signing, and launches it. Stable signing means Accessibility and Microphone permissions persist across rebuilds.

`scripts/run.sh` embeds Sparkle when available, but automatic updates stay disabled until you provide a real `SUPublicEDKey` in the generated Info.plist.

Grant **Accessibility** and **Microphone** permissions when prompted.

On a fresh Mac, Yuwp will launch without a model and wait for you to choose one. Use **Model → Download Streaming Model** from the menu bar app to download the default small model. No downloads start until you explicitly choose that action.

> **Note:** `scripts/run.sh` auto-detects a local `Developer ID Application` signing identity when one is available. Override it with `YUWP_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"`, or force ad-hoc signing with `YUWP_SIGN_IDENTITY=-`.

### Alternative: direct build

```bash
swift build
swift run Yuwp
```

Note: without `scripts/run.sh`, macOS ties permissions to the binary hash — every rebuild requires re-granting Accessibility and Microphone access.

## Usage

**Ctrl+`** toggles dictation by default. Talk normally — text appears in whatever field has focus. Press the shortcut again to stop.

Open **Settings…** from the menu bar to configure:
- **General**
  - **Dictation Mode** — Toggle or Push to Talk
  - **Shortcut** — recorded directly in the settings window
  - current presets: Ctrl+`, ⌥+Space, ⌘+⇧+D
- **Models**
  - pick a preset (**Small**, **Large**, or **Custom**)
  - set separate streaming and final-pass model ids or local folder paths
  - enable or disable the final accuracy pass after stop
  - download supported models directly from the settings window
- **Recordings**
  - **Save Recordings** defaults to **off**
  - choose a custom save location or reset to the default app support folder
- **Server**
  - **Off** — don't run the bundled ASR server
  - **Localhost** — run the server on `127.0.0.1:<port>`
  - **0.0.0.0** — expose the server on all interfaces for LAN clients
  - **Server Port** defaults to `9748`

### Text Injection Strategies

Yuwp picks the best injection method for the focused app:

| Strategy | How | Works in |
|----------|-----|----------|
| **AX API** (default) | Reads/writes the focused text element via Accessibility | Most apps (Safari, Notes, TextEdit, etc.) |
| **CGEvent** | Simulates keyboard events | Terminal, iTerm2, other apps that block AX writes |
| **Clipboard** | Saves clipboard → paste → restore | Fallback when both AX and CGEvent fail |

Live streaming (seeing words appear as you speak) works with AX API and CGEvent. Clipboard fallback commits the final text only.

### Model Presets

Settings offers two built-in presets:

| Preset | Model | Use Case |
|--------|-------|----------|
| **Small** | Qwen3-ASR-0.6B-4bit | Low latency, lower memory (~900MB) |
| **Large** | Qwen3-ASR-1.7B-bf16 | Best accuracy, higher latency (~4GB) |

Both presets use the same model for streaming and the final accuracy pass.

Models are only downloaded after explicit user action from Settings. Yuwp never starts a model download on launch by itself.

## Standalone CLI Transcription

Canonical CLI:

```bash
swift build -c release --product yuwp-asr
bash scripts/build_mlx_metallib.sh release

.build/arm64-apple-macosx/release/yuwp-asr transcribe sample.m4a \
  [--model /path/to/model-dir] \
  [--format text|json|srt|vtt] \
  [--output out.txt]
```

Legacy compatibility:

```bash
swift build -c release --product yuwp-transcribe
.build/arm64-apple-macosx/release/yuwp-transcribe sample.m4a [options]
```

Notes:
- if `--model` is omitted, the CLI uses Yuwp's saved transcription model, then falls back to the built-in default model spec
- `json` includes `text`, `language`, `duration`, and `segments` when the forced aligner is available locally
- `srt` and `vtt` require the default forced aligner model to be present locally
- the CLI accepts any audio format `AVAudioFile` can decode (`wav`, `m4a`, `mp3`, etc.) and resamples to 16kHz mono internally
- if you move the binary out of `.build/.../release/`, move `mlx.metallib` with it too

Examples:

```bash
# Plain transcript using Yuwp's saved model
.build/arm64-apple-macosx/release/yuwp-asr transcribe note.m4a

# Explicit model path
.build/arm64-apple-macosx/release/yuwp-asr transcribe note.m4a \
  --model ~/models/Qwen3-ASR-0.6B-4bit

# Rich JSON
.build/arm64-apple-macosx/release/yuwp-asr transcribe note.m4a \
  --format json

# Timed subtitles
.build/arm64-apple-macosx/release/yuwp-asr transcribe note.m4a \
  --format srt \
  --output note.srt
```

### CLI benchmark comparison

```bash
uv run scripts/benchmark.py \
  --audio ~/workspace/qwen-asr/samples/jfk.wav \
  --audio /tmp/yuwp-subtitle-bench/video-4m.m4a \
  --audio /tmp/yuwp-subtitle-bench/video-32k.m4a \
  --tool yuwp \
  --tool mlx-audio \
  --tool qwen-asr \
  --qwen-args '-S 30 -W 3'
```

Measured on an **Apple M3 Ultra**.

- **Yuwp**: `asr-server` batch endpoint
- **mlx-audio**: load-once Python batch reference
- **qwen_asr**: normal mode for short audio, segmented mode (`-S 30 -W 3`) for medium and long audio

### Offline Mode

| Setup | Audio | Yuwp (`wall`, realtime) | mlx-audio (`wall`, realtime) | qwen_asr (`wall`, realtime) |
|-------|-------|--------------------------|-------------------------------|------------------------------|
| `jfk.wav` | `11.0s` | `0.181s`, `60.78x` | `0.643s`, `17.11x` | `1.089s`, `10.10x` |
| `video-4m.m4a` | `240.0s` | `6.179s`, `38.84x` | `25.158s`, `9.54x` | `26.703s`, `8.99x` |
| `video-32k.m4a` | `3526.0s` | `80.009s`, `44.07x` | `76.642s*`, `46.01x*` | `384.467s`, `9.17x` |
| **weighted total** | **`3777.0s`** | **`86.369s`, `43.73x`** | **`102.443s*`, `36.87x*`** | **`412.260s`, `9.16x`** |

`*` On `video-32k.m4a`, `mlx-audio` returned a much shorter transcript (`39,424` chars) than Yuwp (`68,315`) and `qwen_asr` (`68,411`). The tail also ended early instead of reaching the episode outro. Treat that row as incomplete output, not a clean full-transcript win.

## Standalone ASR Server

Canonical CLI:

```bash
swift build -c release --product yuwp-asr
.build/arm64-apple-macosx/release/yuwp-asr serve \
  [--model /path/to/model-dir-or-repo-id] \
  [--batch-model <dir>] \
  [--aligner-model <dir>] \
  [--disable-vad] \
  [--disable-batch-retranscribe] \
  [--port 9748] \
  [--host 127.0.0.1] \
  [--warmup]
```

Legacy compatibility:

```bash
swift build -c release --product asr-server
.build/arm64-apple-macosx/release/asr-server [--model /path/to/model-dir-or-repo-id] [other options]
.build/arm64-apple-macosx/release/asr-server <streaming-model-dir> [other options]
```

Treat the positional model arg as legacy compatibility only. `--model` is the canonical flag vocabulary across `yuwp-asr serve`, `yuwp-asr transcribe`, `asr-server`, and `yuwp-transcribe`.

Use `--host 0.0.0.0` only when you explicitly want LAN clients to connect.

If `--model` is omitted, local tooling should resolve the model from Yuwp's saved app config first, then the built-in default model spec. Timed `json` / `srt` / `vtt` output uses the default aligner model automatically when it is already present locally. Pass `--aligner-model` only to override it.

### HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/info` | Model info and server status (`aligner` / `vad` included) |
| `POST` | `/v1/audio/transcriptions` | OpenAI-style batch transcription (`text`, `json`, `srt`, `vtt`) |
| `POST` | `/audio/transcriptions` | Alias for `/v1/audio/transcriptions` |
| `POST` | `/v1/audio/subtitles` | Deprecated legacy subtitle alias |
| `POST` | `/v1/audio/transcriptions/stream` | Create a new streaming session |
| `POST` | `/v1/audio/transcriptions/stream/:id` | Feed audio chunk (raw 16kHz mono s16le PCM) |
| `DELETE` | `/v1/audio/transcriptions/stream/:id` | Stop session, returns final transcription |

#### Streaming

```bash
# Create session
ID=$(curl -s -X POST http://localhost:9748/v1/audio/transcriptions/stream | jq -r .session_id)

# Feed audio (raw 16kHz mono s16le PCM)
curl -X POST --data-binary @audio.pcm http://localhost:9748/v1/audio/transcriptions/stream/$ID

# Stop and get final text
curl -s -X DELETE http://localhost:9748/v1/audio/transcriptions/stream/$ID
```

#### Batch transcription

```bash
curl http://localhost:9748/v1/audio/transcriptions \
  -F file=@audio.m4a \
  -F model=qwen3-asr-0.6b \
  -F response_format=json
```

Supported `response_format` values:
- `text`
- `json`
- `srt`
- `vtt`

Example JSON response:

```json
{
  "text": "full transcript",
  "language": "English",
  "duration": 123.45,
  "segments": [
    { "start": 0.88, "end": 5.28, "text": "first subtitle" }
  ]
}
```

`json` includes `segments` when the aligner is loaded. `srt` and `vtt` require the aligner.

#### Long-audio behavior

The server keeps the streaming path unchanged. Batch endpoints share the same chunking behavior:

- when built-in Silero VAD is available, audio is chunked on speech/silence boundaries
- otherwise, the server falls back to low-energy chunking, following the same basic strategy used by `mlx-audio`
- chunking still targets roughly `120s` max chunks, but cuts move to local low-energy boundaries instead of hard time splits
- short files naturally stay as a single chunk
- timed `json` / `srt` / `vtt` output requires the aligner

Other batch limits and notes:
- request body limit: `100 MB`
- oversized uploads return JSON `413` instead of a dropped connection
- use compressed uploads (`m4a`, `flac`, etc.) for long recordings instead of giant WAV files
- `GET /v1/info` reports whether `aligner` and `vad` are active

## Benchmarking

Use one composable benchmark CLI:

```bash
# Show all flags
uv run scripts/benchmark.py --help

# Compare Yuwp server vs mlx-audio on one long file
uv run scripts/benchmark.py \
  --audio /tmp/yuwp-subtitle-bench/video-32k.m4a \
  --tool yuwp \
  --tool mlx-audio \
  --compare-text

# Compare Yuwp VAD vs low-energy fallback on the same audio
uv run scripts/benchmark.py \
  --audio /tmp/yuwp-subtitle-bench/video-32k.m4a \
  --tool yuwp \
  --yuwp-chunking vad \
  --yuwp-chunking energy \
  --compare-text

# Compare several files and tools explicitly
uv run scripts/benchmark.py \
  --audio ~/workspace/qwen-asr/samples/jfk.wav \
  --audio /tmp/yuwp-subtitle-bench/video-4m.m4a \
  --audio /tmp/yuwp-subtitle-bench/video-32k.m4a \
  --tool yuwp \
  --tool mlx-audio \
  --tool qwen-asr \
  --qwen-args '-S 30 -W 3' \
  --compare-text \
  --json /tmp/asr-benchmark.json
```

The benchmark script is batch-transcription focused. It expands the explicit cross-product you ask for: audio files × tools × model variants × chunking modes × repeats.

## Silence Handling

Three layers prevent hallucinated text during silence:

1. **Server**: RMS energy below threshold → skip inference entirely
2. **Server**: Token guard — silence cannot erase confirmed text via rollback
3. **Swift**: Empty partials filtered before reaching the typewriter animator

## Project Structure

```
Sources/
  App.swift                 # NSApplication entry, menu bar, orchestration
  DictationSession.swift    # Session state machine and protocol abstractions
  NativeASRProvider.swift   # asr-server process management, HTTP STT sessions
  ModelManager.swift        # HF model resolution from cache or local paths
  HotkeyManager.swift       # Global hotkey via CGEvent tap
  AudioCapture.swift        # AVAudioEngine → 16kHz mono PCM
  AXTextInjector.swift      # Accessibility API text injection
  CGEventInjector.swift     # Keyboard event injection for terminals
  ClipboardInjector.swift   # Clipboard fallback
  TextInjectorFactory.swift # Injection strategy selection
  MicPanel.swift            # Floating mic indicator
  TypewriterAnimator.swift  # Character-by-character text reveal
  Config.swift              # UserDefaults preferences
  NativeASR/                # MLX model loading, inference, streaming session
    ForcedAligner.swift     # Subtitle alignment model wrapper
    SileroVAD.swift         # CoreML VAD chunking for long batch jobs
    Resources/              # Bundled Silero VAD CoreML model
  asr-server/
    main.swift              # Legacy compatibility server entrypoint
  yuwp-asr/
    main.swift              # Canonical CLI: `serve` + `transcribe`
  yuwp-transcribe/
    main.swift              # Legacy compatibility wrapper for `yuwp-asr transcribe`
  asr-stream-test/
    main.swift              # Replay WAVs through streaming + batch, emit quality metrics
  align-test/
    main.swift              # Local forced-alignment CLI
Tests/
  DictationSessionTests.swift
  TypewriterAnimatorTests.swift
  CGEventInjectorTests.swift
  EnterInterceptionTests.swift
  ASRServerTests.swift
  ForcedAlignerTests.swift
  SileroVADTests.swift
scripts/
  build.sh                  # Build + stable codesign
  run.sh                    # Build + launch as .app bundle
  benchmark.py              # Composable batch transcription benchmark CLI
  validate-native-asr.py    # Golden transcript validation
```

## Development

```bash
swift build       # build
swift test        # run tests
swift run asr-stream-test ~/path/to/sample.wav   # replay + compare against batch baseline
scripts/build.sh  # build + codesign with stable identity
scripts/run.sh    # build + launch as .app bundle
```

See [AGENTS.md](AGENTS.md) for architecture details, protocol specs, and coding conventions.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for machine learning on Apple Silicon
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Swift bindings for MLX
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — speech recognition model by the Qwen team at Alibaba
- [qwen-asr](https://github.com/antirez/qwen-asr) — streaming ASR reference with sliding window KV cache optimizations
- [Silero VAD](https://github.com/snakers4/silero-vad) — original VAD model by the Silero Team; the bundled CoreML model in this repo is derived from [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml)

## License

[MIT](LICENSE)

Third-party model/resource acknowledgments:
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
