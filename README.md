# Yuwp

On-device dictation for macOS.

Press a hotkey, talk, and Yuwp types into the focused field. Once a model is installed, dictation runs locally on your Mac with no cloud API calls.

## Requirements

- macOS 14+
- Apple Silicon

## Quick start

```bash
git clone https://github.com/duh17/yuwp.git
cd yuwp
scripts/run.sh
```

This builds and launches a signed `Yuwp.app` bundle so Accessibility and Microphone permissions survive rebuilds.

On first launch:
1. grant **Accessibility** and **Microphone** access
2. open **Settings…**
3. choose or download a model

If you run the app directly with `swift run Yuwp`, macOS may ask for permissions again after each rebuild.

## Usage

- Default shortcut: **Ctrl+`**
- Press once to start dictation
- Press again to stop
- Configure shortcut, models, recordings, server mode, mic panel, and chimes in **Settings…**

Yuwp chooses the best text injection method for the focused app:
- **AX API** for normal text fields
- **CGEvent** for terminals and AX-hostile editors
- **Clipboard fallback** when needed

## CLI

Build the standalone CLI:

```bash
swift build -c release --product yuwp-asr
bash scripts/build_mlx_metallib.sh release
```

Transcribe a file:

```bash
.build/arm64-apple-macosx/release/yuwp-asr transcribe Tests/fixtures/jfk.wav
```

JSON output:

```bash
.build/arm64-apple-macosx/release/yuwp-asr transcribe Tests/fixtures/jfk.wav --format json
```

SRT output:

```bash
.build/arm64-apple-macosx/release/yuwp-asr transcribe Tests/fixtures/jfk.wav --format srt --output /tmp/jfk.srt
```

## HTTP server

Start the local transcription server:

```bash
.build/arm64-apple-macosx/release/yuwp-asr serve
```

Health check:

```bash
curl -sf http://127.0.0.1:9748/v1/info | jq .
```

Batch transcription:

```bash
curl -sf http://127.0.0.1:9748/v1/audio/transcriptions \
  -F file=@Tests/fixtures/jfk.wav \
  -F response_format=text
```

## Development

```bash
swift build
swift test
scripts/build.sh
scripts/run.sh
```

## License

- [MIT](LICENSE)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
