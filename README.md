# Yuwp

[中文](README.zh.md)

Yuwp is a local dictation app for macOS. Press a hotkey and speak to enter text in the app you're using.

It runs Qwen3-ASR or Confucius4-R2T2 on Apple Silicon using Swift and MLX. Once the model is on disk, audio stays on the Mac. Yuwp inserts text through the Accessibility API, uses key events in terminals, and falls back to the clipboard if those methods fail.

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

Transcribe a file or make subtitles (subtitles require a local forced aligner):

```bash
.build/out/Products/Release/yuwp-asr transcribe Tests/fixtures/jfk.wav
/Applications/Yuwp.app/Contents/MacOS/yuwp-asr transcribe Tests/fixtures/jfk.wav
.build/out/Products/Release/yuwp-asr transcribe recording.wav --format srt --output recording.srt
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

Plain batch transcription uses VAD for short files and energy boundaries for files longer than 120 seconds. Subtitles use VAD boundaries at any length when available, then cut audio to at most 30 seconds **before** ASR. Each clip is transcribed and aligned with its own text. If VAD cannot load, subtitles use capped energy cuts. Supplying `text` with an HTTP subtitle request instead distributes that supplied text across clips approximately; it does not run ASR or guarantee word-level placement. Pass `language` for mixed-script text or all-Kanji Japanese, where the script alone does not identify the language.

`--batch-chunking automatic|vad|energy` affects batch jobs, not live streaming. Explicit `vad` falls back to energy if batch VAD cannot load. The legacy `--disable-vad` flag also turns off streaming VAD. `GET /v1/info` reports the requested batch mode, the plain-transcription resolution, and whether batch VAD is available. Subtitle JSON with `debug=true` shows the actual subtitle chunk boundaries.

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

### AuK-Flash and AuK Base (expressive create / clone / edit)

Qwen3-TTS stays the ordinary low-latency default. `yuwp-tts` can also load a native Swift/MLX port of [AuK](https://github.com/Tencent-Hunyuan/AuK): **AuK-Flash** (fixed 4 steps, CFG off, sway off) or **AuK Base** (default `nfe=32`, `cfg=2.0`, `sway=-1.0`). One process loads one model directory; the backend is detected from that directory (`GET /v1/info` reports `backend` `auk-flash` or `auk-base` plus `variant` and the active sampling defaults). Runtime inference does not use Python.

Convert official PyTorch weights once. `--thinker-src` may be Qwen2.5-Omni-3B or an already-converted mlx directory (Base can reuse Flash's thinker):

```bash
.build/out/Products/Release/yuwp-tts convert-auk \
  --src "$HOME/Library/Application Support/Yuwp/models/AuK-Flash" \
  --thinker-src "$HOME/Library/Application Support/Yuwp/models/Qwen2.5-Omni-3B" \
  --out "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx"

.build/out/Products/Release/yuwp-tts convert-auk \
  --src "$HOME/Library/Application Support/Yuwp/models/AuK" \
  --thinker-src "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx" \
  --out "$HOME/Library/Application Support/Yuwp/models/auk-base-mlx"
```

Instruct TTS needs `--gen-seconds`. Pass `--ref-audio` for voice cloning or source-audio editing. Base accepts `--nfe --cfg --sway` (ignored on Flash):

```bash
.build/out/Products/Release/yuwp-tts \
  --model "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx" \
  --instruction "Say the following with the same voice: 'Hello from Yuwp.'" \
  --ref-audio ref_24k.wav \
  --gen-seconds 4 \
  --out /tmp/auk.wav
```

`POST /v1/audio/speech/stream` is honest about latency: for multi-chunk TTS it emits the first playable PCM `audio` event before later chunks generate. A short single utterance may still be one full latent/VAE decode — that is not incremental streaming. NDJSON events are `metadata` (format `pcm_s16le`, `encoding` `base64`, `backend`, `variant`), then `audio` (`chunk`, `samples`, `seconds`, `elapsed_seconds`, `audio`), then `done` (`first_audio_seconds`, `audio_duration_seconds`, `wall_seconds`, `chunks`). Errors are `event=error`.

Long-form **auto-chunk is TTS only**. It preserves sentence boundaries, keeps the same voice/style wrapper on each chunk, and inserts 200 ms of silence between chunks (`--chunk-pause-ms`, `chunk_pause_ms`, or `inter_chunk_pause_seconds`). It runs when `task`/`mode` is `tts`, when `input` is speak-text and `instruction` is voice/style, or when the instruction matches official zero-shot/instruct templates so only the target text is split. Existing-audio edit / enhancement / separation stays a single instruction plus source audio. Requesting auto-chunk on those tasks returns a structured HTTP 400 (`code=auto_chunk_unsupported`) with explicit options rather than splitting the source.

Cookbook-style edits (content replace, pitch/volume/emotion, enhancement, separation) are representable on the same path. **Verified** vs **representable**: content replace and a non-content transform are checked with ASR / pitch-cents / dB in `results/` when weights are present; lyric editing is representable but not claimed verified (upstream Base/Flash/torch also fail the cookbook "rear view" → "like you" case).

### Fish Audio Drama 3 (expressive multi-speaker TTS, CLI only)

`yuwp-tts` runs a native Swift/MLX port of [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3): a 35B-A3B Qwen3.5-MoE backbone, a 13-layer codebook tower, and the 44.1 kHz Fish codec. Convert the official checkpoint once. The converter dequantizes the block-FP8 experts, reads `codec.pth` without Python, and can quantize to 8 or 4 bits:

```bash
hf download fishaudio/drama-3
.build/out/Products/Release/yuwp-tts convert-drama3 \
  --src ~/.cache/huggingface/hub/models--fishaudio--drama-3/snapshots/<rev> \
  --out "$HOME/Library/Application Support/Yuwp/models/drama3-mlx-8bit" --bits 8

.build/out/Products/Release/yuwp-tts \
  --model "$HOME/Library/Application Support/Yuwp/models/drama3-mlx-8bit" \
  --text "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair." \
  --ref-audio ref.wav --ref-text "Transcript of ref.wav." \
  --out /tmp/drama3.wav
```

`--ref-audio`/`--ref-text` are optional (voice clone). Repeat the pair to cast a scene: pair *i* voices `<|speaker:i|>`, so a narrator plus four characters takes five pairs. Without references the model picks its own voices. Inline tags such as `[laughs]` and `<|speaker:N|>` (0–15) go in `--text`. Sampling defaults match upstream: `--temperature 1.0 --top-p 0.9 --top-k 30`; `--max-tokens` caps frames (21.5 frames per second of audio). `--stream` decodes as frames arrive (first audio after about four frames, then `--streaming-interval`, default 0.2 s) using the same `StreamingWAVWriter` path as Qwen3-TTS. `serve` does not load Drama 3 yet.

Mac Studio M3 Ultra, one sentence (ASR transcript matched the input on every run):

| Weights | Disk | Frames/s | Real-time factor |
| --- | ---: | ---: | ---: |
| bf16 | 68 GB | 21–29 | ~1.0–1.3× |
| 8-bit | 38 GB | 35–37 | ~1.7× |
| 4-bit | 22 GB | ~54 | ~2.5× |

The weights use the Fish Audio Research License; check it before anything beyond personal use.

### YuE2 music (CLI only)

`yuwp-music` runs a native Swift/MLX port of [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B): style tags and lyrics in, a 48 kHz stereo song out. YuE2 first writes an ABC score (melody, plus chords in `full` mode), generates 25 Hz semantic codec ids from it, solves acoustic latents with midpoint flow matching, and decodes them with the Oobleck VAE. It loads the converted [`vanch007/mlx-Yue2-3B`](https://huggingface.co/vanch007/mlx-Yue2-3B) layout (`ar-bf16`, `ar-8bit`, `nar-bf16`, `qwen.tiktoken`, `vae/`). `--precision 8bit|4bit|bf16` picks the token-generation weights (default 8-bit); without an `ar-4bit` file, 4-bit quantizes the bf16 weights at load. Flow matching always uses bf16. Runtime inference does not use Python.

```bash
hf download vanch007/mlx-Yue2-3B --local-dir ~/models/mlx-Yue2-3B
swift build -c release --product yuwp-music

.build/out/Products/Release/yuwp-music generate --model ~/models/mlx-Yue2-3B \
  --style "Mandarin, 1990s Hong Kong TVB wuxia theme, male tenor lead vocal, erhu, guzheng, strings" \
  --lyrics-file lyrics.txt --out /tmp/song
```

Lyrics use section tags on their own lines (`[Verse]`, `[Chorus]`, …). `--mode full|melody|off` picks the score plan; `--abc-file` renders your own score instead. `--steps` sets flow-matching steps (default 32; 8 is about 4× faster on that stage). The output directory holds `audio.wav`, `score.abc`, `prefix.npy`, `semantic.npy`, `noise.npy`, `latent.npy` and `result.json`. `yuwp-music render --from <dir>` re-solves a saved song with other steps, and `decode` runs the VAE alone. Song length is up to the model (at most 360 s). Seeds are stable within `yuwp-music` but do not reproduce mlx-Yue's audio: its MLX version draws different samples from the same key.

Mac Studio M3 Ultra, one 4.4-minute instrumental: the same style, score, seed and 32 steps in each engine, 8-bit token weights. Each engine samples its own tokens, so lengths differ slightly (frames in brackets).

| Stage | mlx-Yue (Python MLX) | mlx-serve 26.10.2 (Zig) | `yuwp-music` |
| --- | ---: | ---: | ---: |
| Semantic tokens | 98.8 tok/s | 136.5 tok/s | **180.9 tok/s** |
| Flow matching, 32 steps | 139.6 s (6683) | 123.4 s (6607) | **121.1 s** (6497) |
| VAE decode | 5.4 s | 7.8 s | **2.1 s** |
| End to end | 212.8 s | 181.6 s | **160.0 s** |

mlx-serve and `yuwp-music` are even on flow matching once length is accounted for (both run bf16 MLX attention over the whole chunk; mlx-Yue is about 10% slower). The gap is token generation, where `yuwp-music` keeps sampling on the GPU and queues the next step before reading the current token back. mlx-serve's end-to-end excludes model load (already resident); the other two include it. Method and raw numbers: [`benchmarks/`](benchmarks/README.md#yue2-music), [`yue2-tvb-m3u.json`](benchmarks/fixtures/yue2-tvb-m3u.json). 4-bit token weights reach 261 tok/s.

On the same tokens and noise, latents match mlx-Yue at 37 dB SNR (cosine 0.9999) and VAE audio at 124 dB. YuE2 weights are CC BY-NC 4.0: personal and research use only.

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

### R2T2

Optional Settings model. The 8-bit weights are `mlx-community/Confucius4-R2T2-8bit`: 8-bit affine (group 64) on the language model, bf16 on the audio encoder, not FP8. Settings also offers unquantized bf16 (4.08 GB).

Same headline clocks as the Qwen table, Mac Studio M3 Ultra, macOS 27.2 (26B5091g). Mini R2T2 numbers are not measured. Reproduce with `uv run benchmarks/cli.py asr headline --skip-batch --skip-tts --asr-model LABEL=PATH`. Numbers: [`benchmarks/fixtures/public-jfk-r2t2.json`](benchmarks/fixtures/public-jfk-r2t2.json).

| Model | First M3U | Stop M3U | Transcribe M3U |
| --- | ---: | ---: | ---: |
| R2T2 8-bit | 990 ms | 171 ms | 0.86 s |
| R2T2 bf16 | 939 ms | 206 ms | 1.03 s |

R2T2 commits append-only text every 160 ms and holds one trailing token, so this first-text clock is later than Qwen's rollback/preview path. After the first commit it stays inside the chunk: on `jfk.wav`, Silero speech onset to first text is 800 ms; speech-active 160 ms chunks take 47 ms median and 60 ms p99. Stop → final on that 11 s clip is 171 ms and includes the short-session stop batch. Stream vs batch WER is 0 on `jfk.wav` and `asr_zh.wav`.

Official [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) publishes dataset WER/CER and 200–600 ms average streaming latency (retrospective chunk-wise mean fuzzy latency) for CUDA vLLM / llama.cpp. They do not publish Apple Silicon numbers. That Python path needs NVIDIA CUDA and does not run on this Mac. The table above is Yuwp's MLX Swift port, not a same-hardware comparison with vLLM.

More scenarios: [`benchmarks/`](benchmarks/README.md).

## Privacy

Audio is processed on the Mac. Recording is off unless you enable **Save Recordings**. Diagnostic logs are off unless you turn them on in Settings.

## Acknowledgments

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2)
- [AuK](https://github.com/Tencent-Hunyuan/AuK) (AuK-Flash and AuK Base MLX architecture reference)
- [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) and [SGLang-Omni](https://github.com/sgl-project/sglang-omni) (Drama 3 inference reference)
- [YuE2](https://github.com/multimodal-art-projection/YuE), [mlx-Yue](https://github.com/vanch007/mlx-Yue) and [mlx-serve](https://github.com/ddalcu/mlx-serve) (YuE2 inference references)
- [MLX](https://github.com/ml-explore/mlx) and [mlx-swift](https://github.com/ml-explore/mlx-swift)
- [qwen-asr](https://github.com/antirez/qwen-asr) (streaming reference ideas)
- [Silero VAD](https://github.com/snakers4/silero-vad)

## License

- [MIT](LICENSE)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- App bundles and release DMGs include those notices plus vendored upstream licenses under `OpenSource/`
