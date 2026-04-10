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

## Standalone ASR Server

The ASR server can run independently for streaming dictation, OpenAI-style batch transcription, and subtitle generation:

```bash
swift build -c release --product asr-server
.build/arm64-apple-macosx/release/asr-server <streaming-model-dir> \
  [--batch-model <dir>] \
  [--aligner-model <dir>] \
  [--disable-batch-retranscribe] \
  [--port 9748] \
  [--host 127.0.0.1] \
  [--warmup]
```

Use `--host 0.0.0.0` only when you explicitly want LAN clients to connect.

Pass `--aligner-model` to enable `/v1/audio/subtitles`. In the menu bar app, Yuwp will also auto-load the default aligner model from the local Hugging Face cache when it is already present.

### HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/info` | Model info and server status (`aligner` / `vad` included) |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible batch transcription (multipart upload) |
| `POST` | `/audio/transcriptions` | Alias for `/v1/audio/transcriptions` |
| `POST` | `/v1/audio/subtitles` | Subtitle generation / forced alignment |
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
  -F model=qwen3-asr \
  -F response_format=verbose_json
```

Supported `response_format` values:
- `json`
- `text`
- `verbose_json`

#### Subtitles / forced alignment

```bash
curl http://localhost:9748/v1/audio/subtitles \
  -F file=@audio.m4a \
  -F response_format=srt
```

If you already have a transcript, pass it as `text` and the server will align that text instead of retranscribing.

```bash
curl http://localhost:9748/v1/audio/subtitles \
  -F file=@audio.m4a \
  -F text="existing transcript goes here" \
  -F response_format=json
```

Supported subtitle `response_format` values:
- `srt`
- `vtt`
- `json`
- `text` (transcript only)

Tune subtitle grouping with:
- `max_words_per_line` (default `8`)
- `max_duration` (default `5.0` seconds)
- `pause_threshold` (default `0.5` seconds)

#### Long-audio behavior

The server keeps the streaming path unchanged. Long-file logic only applies to batch endpoints.

- `/v1/audio/transcriptions`
  - short files: single pass
  - audio over `10 min`: chunked on the server
  - when built-in Silero VAD is available, long audio is split on silence
  - if VAD is unavailable, the server falls back to fixed `120s` chunks
- `/v1/audio/subtitles`
  - requires `--aligner-model`
  - short files: single pass align/transcribe + align
  - audio over `4 min`: chunked with built-in Silero VAD when available
  - if VAD is unavailable, the simple non-chunked subtitle path is limited to `10 min`

Other batch limits and notes:
- request body limit: `100 MB`
- oversized uploads return JSON `413` instead of a dropped connection
- use compressed uploads (`m4a`, `flac`, etc.) for long recordings instead of giant WAV files
- `GET /v1/info` reports whether `aligner` and `vad` are active

## Benchmarking

```bash
# Native server concurrency benchmark (requires asr-server to be built)
uv run scripts/bench-native-asr.py --help

# Model comparison benchmark
uv run scripts/bench-models.py --help

# Subtitle stress harness for a sample set
uv run scripts/bench-subtitles.py --help
```

`scripts/bench-subtitles.py` runs `/v1/audio/subtitles` across a manifest of local sample files, saves raw subtitle JSON per sample, and writes a summary JSON with timing, subtitle counts, gap/overlap checks, and approximate realtime factor. Start from `scripts/bench-subtitles.example.json` and swap in your own files.

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
    main.swift              # Native HTTP streaming + batch + subtitle server
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
  bench-models.py           # ASR model benchmarking
  bench-native-asr.py       # Native server concurrency benchmarking
  bench-subtitles.py        # Subtitle stress harness for local samples
  validate-native-asr.py    # Golden transcript validation
```

## Development

```bash
swift build       # build
swift test        # run tests
scripts/build.sh  # build + codesign with stable identity
scripts/run.sh    # build + launch as .app bundle
```

See [AGENTS.md](AGENTS.md) for architecture details, protocol specs, and coding conventions.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for machine learning on Apple Silicon
- [mlx-swift](https://github.com/ml-explore/mlx-swift) — Swift bindings for MLX
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — speech recognition model by the Qwen team at Alibaba
- [qwen-asr](https://github.com/antirez/qwen-asr) — streaming ASR reference with sliding window KV cache optimizations

## License

[MIT](LICENSE)
