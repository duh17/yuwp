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

Model downloads automatically on first run (~3.4GB).

## Install & Run

```bash
git clone https://github.com/duh17/yuwp.git
cd yuwp
swift build
swift run Yuwp
```

Grant **Accessibility** and **Microphone** permissions when prompted.

## Usage

Double-tap **Right Option** to toggle dictation. Talk normally — text appears in whatever field has focus. Double-tap again to stop.

Hotkey is configurable via the menu bar icon (double-tap Right Option, double-tap Fn, or Ctrl+`).

## Standalone ASR Server

The sidecar can run independently as an HTTP server for other apps:

```bash
uv run --script Sources/sidecar/transcribe.py --serve-only
# Listening on http://127.0.0.1:9748
```

See [AGENTS.md](AGENTS.md) for the HTTP API.

## Acknowledgments

- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX inference backend for audio models on Apple Silicon
- [qwen_asr](https://github.com/nickovchinnikov/qwen_asr) — streaming ASR implementation with sliding window KV cache optimizations
- [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — speech recognition model by the Qwen team at Alibaba

## License

MIT
