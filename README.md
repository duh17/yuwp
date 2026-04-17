# Yuwp

Yuwp is a small on-device dictation app for macOS.

- Runs locally with Swift + MLX (Qwen3-ASR)
- Uses a global hotkey to start/stop dictation
- Injects text into most apps (AX API, terminal key events, clipboard fallback)
- No cloud API required after model download

## Requirements

- macOS 14+
- Apple Silicon

## Install

Download the latest notarized release assets:

- https://github.com/duh17/yuwp/releases/latest

Build from source:

```bash
git clone https://github.com/duh17/yuwp.git
cd yuwp
xcodebuild -downloadComponent MetalToolchain   # one-time on fresh machines
scripts/run.sh
```

`scripts/run.sh` builds a signed app bundle and launches it from `/Applications/Yuwp.app`.

## First launch

1. Open Yuwp from the menu bar.
2. Click **Grant Accessibility Permission** and allow Yuwp in Privacy & Security.
3. Press the hotkey once to trigger the **Microphone** permission prompt.
4. Open **Settings… → Transcription** and download/select a model.

## Usage

- Default shortcut: **Ctrl+`**
- Press once to start dictation, press again to stop
- Configure shortcut, model, server mode, recording, mic panel, and chimes in **Settings…**

## CLI / Server

Build CLIs:

```bash
swift build -c release --product yuwp-asr
swift build -c release --product swift-mlx-asr-server
bash scripts/build_mlx_metallib.sh release
```

Transcribe a file:

```bash
.build/arm64-apple-macosx/release/yuwp-asr transcribe Tests/fixtures/jfk.wav
```

Run standalone HTTP server (default transport is stdio, so pass `--transport http`):

```bash
.build/arm64-apple-macosx/release/swift-mlx-asr-server <model-dir> --transport http --host 127.0.0.1 --port 7936
curl -sf http://127.0.0.1:7936/v1/info | jq .
```

## Development

```bash
swift build
swift test
scripts/build.sh
scripts/run.sh
```

Fresh clone note: `swift test` works on a clean clone. `scripts/build.sh` / `scripts/run.sh` require the Metal toolchain to produce `mlx.metallib`.

## Benchmarks

Benchmark tooling lives under [`benchmarks/`](benchmarks/README.md).

```bash
uv run benchmarks/cli.py --help
```

## Privacy

- Private by default: audio processing and transcription run locally on your Mac.
- Recording is off by default. Audio is only saved if you explicitly enable **Save Recordings**.
- Diagnostic logging is off by default and can be enabled manually in Settings when troubleshooting.

## Acknowledgments

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [MLX](https://github.com/ml-explore/mlx) and [mlx-swift](https://github.com/ml-explore/mlx-swift)
- [qwen-asr](https://github.com/antirez/qwen-asr) (streaming reference ideas)
- [Silero VAD](https://github.com/snakers4/silero-vad)

## License

- [MIT](LICENSE)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- Binary app bundles and release DMGs include these notices plus vendored upstream license texts under `OpenSource/`
