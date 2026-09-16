# Yuwp

Yuwp is local dictation for macOS. Press a hotkey, speak, and Yuwp inserts the text into the app you were already using.

- Runs locally on Apple Silicon with Swift + MLX (Qwen3-ASR)
- Uses a global hotkey to start/stop dictation
- Injects text into most apps (AX API, terminal key events, clipboard fallback)
- No cloud API required after model download

## Requirements

- macOS 14+
- Apple Silicon
- Swift 6.4 / Xcode 27+

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
swift build --build-system native -c release --product yuwp-asr
swift build --build-system native -c release --product yuwp-tts
bash scripts/build_mlx_metallib.sh release
```

Fresh app-bundle / DMG installs also include:

```bash
/Applications/Yuwp.app/Contents/MacOS/yuwp-asr
/Applications/Yuwp.app/Contents/MacOS/yuwp-tts
```

Transcribe a file:

```bash
.build/arm64-apple-macosx/release/yuwp-asr transcribe Tests/fixtures/jfk.wav
# or from an installed app bundle / DMG
/Applications/Yuwp.app/Contents/MacOS/yuwp-asr transcribe Tests/fixtures/jfk.wav
```

Run standalone ASR HTTP server (default transport is stdio, so pass `--transport http`):

```bash
.build/arm64-apple-macosx/release/yuwp-asr serve --model <asr-model-dir> --transport http --host 127.0.0.1 --port 7936
curl -sf http://127.0.0.1:7936/v1/info | jq .
```

`POST /v1/audio/transcriptions/stream` creates a session. Existing clients may send an empty body.
Compatible clients may send `{model, stream_config:{contextual_strings:[...]}}` and must omit
`stream_config` when there are no hints. The response includes `session_id` and `context_applied`
(true only when nonempty hints were consumed). Vocabulary bounds: 100 phrases, 256 UTF-8 bytes
each, 8192 aggregate UTF-8 bytes; empty/whitespace-only strings and control characters are rejected.
Yuwp does not accept a client `system_prompt`; it derives a short internal vocabulary header.

Batch transcription uses automatic chunking: short inputs retain VAD, while inputs
longer than 120 seconds use energy boundaries. Override it without changing live
streaming with `--batch-chunking automatic|vad|energy`; an explicit `vad` request safely falls
back to energy if batch VAD is unavailable. `--disable-vad` remains the legacy flag
that also disables streaming VAD. HTTP `/v1/info` reports the requested batch mode,
its duration-dependent/fallback resolution, and batch-VAD availability separately.

Run standalone TTS HTTP server:

```bash
.build/arm64-apple-macosx/release/yuwp-tts serve --transport http --model <qwen3-tts-model-dir> --host 127.0.0.1 --port 7937
curl -sf http://127.0.0.1:7937/v1/info | jq .
```

Generate speech directly from the CLI:

```bash
.build/arm64-apple-macosx/release/yuwp-tts --model <qwen3-tts-model-dir> --text "Hello from Yuwp" --out /tmp/hello.wav
# or from an installed app bundle / DMG
/Applications/Yuwp.app/Contents/MacOS/yuwp-tts --model <qwen3-tts-model-dir> --text "Hello from Yuwp" --out /tmp/hello.wav
```

TTS exposes `POST /v1/audio/speech` for full WAV responses and `POST /v1/audio/speech/stream` for chunked NDJSON audio events (`metadata`, `audio`, `done`, `error`) with base64 `pcm_s16le` chunks.

## Development

```bash
swift build --build-system native
swift test --build-system native
scripts/build.sh
scripts/run.sh
```

Swift 6.4's SwiftPM defaults to Swift Build, which tries to compile mlx-swift Metal sources and fails unless the standalone Metal toolchain is installed. Pass `--build-system native`. Fresh clone note: `swift test --build-system native` works on a clean clone. `scripts/build.sh` / `scripts/run.sh` still need the Metal toolchain to produce `mlx.metallib`.

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
