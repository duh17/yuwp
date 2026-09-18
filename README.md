# Yuwp

[中文](README.zh.md)

Yuwp is a local dictation app for macOS. Press a hotkey and speak to enter text in the app you're using.

It runs Qwen3-ASR on Apple Silicon using Swift and MLX. Once the model is on disk, audio stays on the Mac. Yuwp inserts text through the Accessibility API, uses key events in terminals, and falls back to the clipboard if those methods fail.

## Requirements

macOS 14+, Apple Silicon, Swift 6.4 / Xcode 27+.

## Install

Download a notarized build: https://github.com/duh17/yuwp/releases/latest

From source:

```bash
git clone https://github.com/duh17/yuwp.git
cd yuwp
xcodebuild -downloadComponent MetalToolchain   # once, on a fresh machine
scripts/run.sh
```

`scripts/run.sh` signs the app and launches `/Applications/Yuwp.app`.

## First launch

1. Open Yuwp from the menu bar.
2. Click **Grant Accessibility Permission** and allow Yuwp in Privacy & Security.
3. Press the hotkey once to trigger the macOS **Microphone** permission prompt.
4. In **Settings… → Transcription**, download or select a model.

## Usage

The default shortcut is **Ctrl+`**. Press it to start dictation; press it again to stop.

Use Settings to change the shortcut, model, server mode, recording, mic panel, and chimes.

## CLI / Server

Build the ASR and TTS tools:

```bash
swift build -c release --product yuwp-asr
swift build -c release --product yuwp-tts
```

DMGs and app bundles include both tools:

```
/Applications/Yuwp.app/Contents/MacOS/yuwp-asr
/Applications/Yuwp.app/Contents/MacOS/yuwp-tts
```

Transcribe a file:

```bash
.build/out/Products/Release/yuwp-asr transcribe Tests/fixtures/jfk.wav
/Applications/Yuwp.app/Contents/MacOS/yuwp-asr transcribe Tests/fixtures/jfk.wav
```

ASR HTTP server (stdio is the default transport; pass `--transport http`):

```bash
.build/out/Products/Release/yuwp-asr serve --model <asr-model-dir> --transport http --host 127.0.0.1 --port 7936
curl -sf http://127.0.0.1:7936/v1/info | jq .
```

`POST /v1/audio/transcriptions/stream` opens a session and accepts an empty body. Compatible clients can send `{model, stream_config:{contextual_strings:[...]}}`. They must omit `stream_config` when there are no hints.

- The response is `{session_id, context_applied}`. `context_applied` is true only when nonempty hints were used.
- Limits: 100 phrases, 256 UTF-8 bytes per phrase, 8192 bytes total.
- Empty strings, whitespace-only strings, and strings containing control characters are rejected.
- Clients cannot set `system_prompt`; Yuwp writes a short vocabulary header itself.

Batch transcription splits audio automatically, using VAD for short files and energy boundaries for files longer than 120 seconds. `--batch-chunking automatic|vad|energy` affects batch transcription only, not live streaming. Explicit `vad` falls back to energy if batch VAD cannot load. The legacy `--disable-vad` flag also turns off streaming VAD. `GET /v1/info` reports the requested batch mode, the resolved mode, and whether batch VAD is available.

TTS HTTP server:

```bash
.build/out/Products/Release/yuwp-tts serve --transport http --model <qwen3-tts-model-dir> --host 127.0.0.1 --port 7937
curl -sf http://127.0.0.1:7937/v1/info | jq .
```

```bash
.build/out/Products/Release/yuwp-tts --model <qwen3-tts-model-dir> --text "Hello from Yuwp" --out /tmp/hello.wav
/Applications/Yuwp.app/Contents/MacOS/yuwp-tts --model <qwen3-tts-model-dir> --text "Hello from Yuwp" --out /tmp/hello.wav
```

`POST /v1/audio/speech` returns a WAV. `POST /v1/audio/speech/stream` returns NDJSON events (`metadata`, `audio`, `done`, `error`) with base64-encoded `pcm_s16le` chunks.

### AuK-Flash (instruction-driven TTS / audio editing)

`yuwp-tts` can also load a native Swift/MLX port of [AuK-Flash](https://github.com/Tencent-Hunyuan/AuK) (fixed 4 steps, CFG off). Runtime inference does not use Python. Convert official PyTorch weights once:

```bash
.build/out/Products/Release/yuwp-tts convert-auk \
  --src "$HOME/Library/Application Support/Yuwp/models/AuK-Flash" \
  --thinker-src "$HOME/Library/Application Support/Yuwp/models/Qwen2.5-Omni-3B" \
  --out "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx" \
  --bits 8
```

Then synthesize. Instruct TTS needs `--gen-seconds`. Pass `--ref-audio` for voice cloning / source-audio editing:

```bash
.build/out/Products/Release/yuwp-tts \
  --model "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx" \
  --instruction "Say the following with the same voice: 'Hello from Yuwp.'" \
  --ref-audio ref_24k.wav \
  --gen-seconds 4 \
  --out /tmp/auk.wav
```

## Development

```bash
swift build
swift test
scripts/build.sh
scripts/run.sh
```

Swift 6.4 SwiftPM uses Swift Build. Binaries land in `.build/out/Products/{Debug,Release}`, including `Sparkle.framework`. Packaging scripts prefer that copy (`swift build --show-bin-path`) and fall back to `.build/artifacts/sparkle/...`. Fresh machines need `xcodebuild -downloadComponent MetalToolchain` before `swift build` can compile mlx-swift Metal sources.

## Benchmarks

Same macOS 27.0 (26A428). Reproduce with `uv run benchmarks/cli.py asr headline`. Numbers: [`benchmarks/fixtures/public-jfk.json`](benchmarks/fixtures/public-jfk.json).

Dictation is 100 ms stdio packets on [`jfk.wav`](Tests/fixtures/jfk.wav) (11 s). First text is energy onset (0.30 s of leading quiet) to the first nonempty partial. Stop → final is the stop request to the final transcript. Transcribe is `yuwp-asr transcribe` wall time on the same clip.

Cloud streaming scores such as Meta Muse Voice Transcribe (~0.16 s to final on [Artificial Analysis](https://artificialanalysis.ai), 1 Sep 2026) start at **end of speech**. That is stop → final, not first text while you are still talking. An 80 ms ingest chunk is how often audio is sent, not time-to-first-text.

M3U is a Mac Studio (M3 Ultra, 512 GB). M4P is a Mac mini (M4 Pro, 64 GB).

| Model | First M3U | First M4P | Stop M3U | Stop M4P | Transcribe M3U | Transcribe M4P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B 4-bit | 620 ms | 1210 ms | 94 ms | 176 ms | 1.18 s | 0.71 s |
| 0.6B bf16 | 618 ms | 1247 ms | 132 ms | 247 ms | 1.25 s | 0.84 s |
| 1.7B 4-bit | 625 ms | 1274 ms | 150 ms | 323 ms | 1.23 s | 0.90 s |
| 1.7B bf16 | 644 ms | 1375 ms | 181 ms | 383 ms | 1.55 s | 1.33 s |

On a mixed-speech Silero-onset canary, M3U 1.7B bf16 first text is 505 ms.

**Speak**, Qwen3-TTS 1.7B CustomVoice on M3U: 2.05 s to generate 4.16 s of audio.

More scenarios: [`benchmarks/`](benchmarks/README.md).

## Privacy

Audio is processed on the Mac. Recording is off unless you enable **Save Recordings**. Diagnostic logs are off unless you turn them on in Settings.

## Acknowledgments

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [AuK](https://github.com/Tencent-Hunyuan/AuK) (AuK-Flash MLX architecture reference)
- [MLX](https://github.com/ml-explore/mlx) and [mlx-swift](https://github.com/ml-explore/mlx-swift)
- [qwen-asr](https://github.com/antirez/qwen-asr) (streaming reference ideas)
- [Silero VAD](https://github.com/snakers4/silero-vad)

## License

- [MIT](LICENSE)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- App bundles and release DMGs include those notices plus vendored upstream licenses under `OpenSource/`
