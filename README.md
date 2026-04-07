# Yuwp

System-wide voice dictation for macOS. Press a hotkey, talk, see your words appear in any text field.

Runs entirely on-device — no cloud APIs, no network requests. Audio never leaves your machine.

## How It Works

A Swift menu bar app captures microphone audio and streams it to a Python sidecar process running [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) via [mlx-audio](https://github.com/Blaizzy/mlx-audio). Transcription results are injected into the focused text field in real time using the macOS Accessibility API.

```
Hotkey → Mic Capture → ASR Sidecar → Text Injection
         (16kHz PCM)   (mlx-audio)    (AX API)
```

The sidecar implements streaming optimizations from [qwen_asr](https://github.com/nickovchinnikov/qwen_asr) — encoder window caching, decoder KV cache reuse, and prefix rollback — to achieve O(1) latency per audio chunk instead of reprocessing the entire recording each time.

## Requirements

- macOS 14+
- Apple Silicon (M1 or later)
- [uv](https://docs.astral.sh/uv/) (Python package manager)

Model downloads automatically on first run (~1GB for the default 0.6B-4bit model).

## Install & Run

```bash
git clone https://github.com/duh17/yuwp.git
cd yuwp
scripts/run.sh
```

This builds the app, creates a proper .app bundle with stable code signing, and launches it. Stable signing means Accessibility and Microphone permissions persist across rebuilds.

Grant **Accessibility** and **Microphone** permissions when prompted.

### Alternative: direct build

```bash
swift build
swift run Yuwp
```

Note: without `scripts/run.sh`, macOS ties permissions to the binary hash — every rebuild requires re-granting Accessibility and Microphone access.

## Usage

**Double-tap Right Option** to toggle dictation. Talk normally — text appears in whatever field has focus. Double-tap again to stop.

Hotkey is configurable via the menu bar icon:
- Double-tap Right Option (default)
- Double-tap Fn
- Ctrl+`

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

| Preset | Streaming Model | Batch Correction | Use Case |
|--------|----------------|-------------------|----------|
| **Fast** | 0.6B-4bit | Off | Low latency, lower memory |
| **Balanced** | 0.6B-4bit | 1.7B-bf16 | Fast streaming + high-quality final correction |
| **Quality** | 1.7B-bf16 | 1.7B-bf16 | Best accuracy, higher latency |

The "Balanced" preset uses a small model for real-time streaming and a larger model to retranscribe the full audio on pause, correcting errors from the streaming pass.

## Standalone ASR Server

The sidecar can run independently as an HTTP server:

```bash
uv run --script Sources/sidecar/transcribe.py --serve-only
# Listening on http://127.0.0.1:9748
```

### HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions/stream` | Create a new transcription session |
| `POST` | `/v1/audio/transcriptions/stream/:id` | Feed audio chunk (raw 16kHz mono s16le PCM) |
| `DELETE` | `/v1/audio/transcriptions/stream/:id` | Stop session, returns final transcription |
| `GET` | `/v1/info` | Model info and server status |

```bash
# Create session
ID=$(curl -s -X POST http://localhost:9748/v1/audio/transcriptions/stream | jq -r .id)

# Feed audio (raw 16kHz mono s16le PCM)
curl -X POST --data-binary @audio.pcm http://localhost:9748/v1/audio/transcriptions/stream/$ID

# Stop and get final text
curl -s -X DELETE http://localhost:9748/v1/audio/transcriptions/stream/$ID
```

## Benchmarking

The included benchmark script tests different ASR models:

```bash
# Requires the standalone server to be running
scripts/bench-models.py --help
```

## Silence Handling

Three layers prevent hallucinated text during silence:

1. **Sidecar**: RMS energy below threshold → skip inference entirely
2. **Sidecar**: Token guard — silence cannot erase confirmed text via rollback
3. **Swift**: Empty partials filtered before reaching the typewriter animator

## Project Structure

```
Sources/
  App.swift                 # NSApplication entry, menu bar, orchestration
  DictationSession.swift    # Session state machine and protocol abstractions
  HotkeyManager.swift       # Global hotkey via CGEvent tap
  AudioCapture.swift        # AVAudioEngine → 16kHz mono PCM
  ASRSidecar.swift          # Python sidecar process management
  AXTextInjector.swift      # Accessibility API text injection
  CGEventInjector.swift     # Keyboard event injection for terminals
  ClipboardInjector.swift   # Clipboard fallback
  TextInjectorFactory.swift # Injection strategy selection
  MicPanel.swift            # Floating mic indicator
  TypewriterAnimator.swift  # Character-by-character text reveal
  Config.swift              # UserDefaults preferences
  sidecar/
    transcribe.py           # Streaming ASR engine (mlx-audio)
Tests/
  DictationSessionTests.swift
  TypewriterAnimatorTests.swift
  CGEventInjectorTests.swift
  EnterInterceptionTests.swift
  SidecarStreamingTests.swift
scripts/
  build.sh                  # Build + stable codesign
  run.sh                    # Build + launch as .app bundle
  bench-models.py           # ASR model benchmarking
```

## Development

```bash
swift build       # build
swift test        # run tests (66 tests)
scripts/build.sh  # build + codesign with stable identity
scripts/run.sh    # build + launch as .app bundle
```

See [AGENTS.md](AGENTS.md) for architecture details, protocol specs, and coding conventions.

## Acknowledgments

- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX inference backend for audio models on Apple Silicon
- [qwen_asr](https://github.com/nickovchinnikov/qwen_asr) — streaming ASR implementation with sliding window KV cache optimizations
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — speech recognition model by the Qwen team at Alibaba
- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for machine learning on Apple Silicon

## License

[MIT](LICENSE)
