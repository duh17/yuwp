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
- **Dictation Mode**
  - **Toggle** — press once to start, again to stop
  - **Push to Talk** — hold to dictate, release to stop
- **Shortcut**
  - recorded directly in the settings window
  - current presets: Ctrl+`, ⌥+Space, ⌘+⇧+D
- **Server Mode**
  - **Off** — don't run the bundled ASR server
  - **Localhost** — run the server on `127.0.0.1:<port>`
  - **0.0.0.0** — expose the server on all interfaces for LAN clients
- **Server Port**
  - configurable in Settings
  - defaults to `9748`

### Text Injection Strategies

Yuwp picks the best injection method for the focused app:

| Strategy | How | Works in |
|----------|-----|----------|
| **AX API** (default) | Reads/writes the focused text element via Accessibility | Most apps (Safari, Notes, TextEdit, etc.) |
| **CGEvent** | Simulates keyboard events | Terminal, iTerm2, other apps that block AX writes |
| **Clipboard** | Saves clipboard → paste → restore | Fallback when both AX and CGEvent fail |

Live streaming (seeing words appear as you speak) works with AX API and CGEvent. Clipboard fallback commits the final text only.

### Model Presets

The menu bar offers model presets:

| Preset | Model | Use Case |
|--------|-------|----------|
| **Small** | Qwen3-ASR-0.6B-4bit | Low latency, lower memory (~900MB) |
| **Large** | Qwen3-ASR-1.7B-bf16 | Best accuracy, higher latency (~4GB) |

Both presets use the same model for streaming and batch retranscription (final correction on pause/stop).

Models are only downloaded after explicit user action from the menu. Yuwp never starts a model download on launch by itself.

## Standalone ASR Server

The ASR server can run independently:

```bash
swift build -c release --product asr-server
.build/arm64-apple-macosx/release/asr-server <model-dir> [--port 9748] [--host 127.0.0.1]
```

Use `--host 0.0.0.0` only when you explicitly want LAN clients to connect.

### HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/info` | Model info and server status |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible batch transcription (multipart upload) |
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

#### Batch (OpenAI-compatible)

```bash
curl http://localhost:9748/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen3-asr \
  -F response_format=json
```

## Benchmarking

```bash
# Native server benchmark (requires asr-server to be built)
scripts/bench-native-asr.py --help

# Model comparison benchmark
scripts/bench-models.py --help
```

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
  asr-server/
    main.swift              # Native HTTP streaming ASR server
Tests/
  DictationSessionTests.swift
  TypewriterAnimatorTests.swift
  CGEventInjectorTests.swift
  EnterInterceptionTests.swift
  ASRServerTests.swift
scripts/
  build.sh                  # Build + stable codesign
  run.sh                    # Build + launch as .app bundle
  bench-models.py           # ASR model benchmarking
  bench-native-asr.py       # Native server concurrency benchmarking
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
