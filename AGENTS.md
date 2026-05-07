# Yuwp — Development Guide

Instructions for AI coding agents working on this codebase.

## Architecture

```
┌───────────────────────────────────────────┐
│            Yuwp.app (macOS)               │
│  Hotkey → AudioCapture → NativeASRProvider│
│                      ↕ stdio / HTTP :7936 │
│               TextInjector (AX API)       │
└───────────────────────┬───────────────────┘
                        │
           ┌────────────▼────────────┐
           │  yuwp-asr serve         │
           │  ┌──────────────────┐   │
           │  │ StreamingSession │   │
           │  │ (encoder cache,  │   │
           │  │  KV reuse,       │   │
           │  │  prefix rollback)│   │
           │  └──────────────────┘   │
           │   stdio / HTTP :7936    │
           └─────────────────────────┘
```

The native `yuwp-asr serve` process loads the MLX model once and serves either
stdio IPC (default) or HTTP.
Yuwp.app launches it as a child process and usually talks via stdio.
External clients can use the same HTTP API when the server runs with
`--transport http`.

## Build & Run

```bash
# Build everything (app + ASR/TTS CLIs)
swift build
swift build -c release --product yuwp-asr
bash scripts/build_mlx_metallib.sh release  # compile Metal shaders

swift run Yuwp
```

Use `scripts/build.sh` to codesign with a stable identifier (preserves
Accessibility/Microphone permissions across rebuilds). Use `scripts/run.sh`
to build and launch as a proper .app bundle with TCC-compatible Info.plist.

### Release (notarized DMG)

```bash
scripts/release.sh <version>
# Requires: YUWP_SIGN_IDENTITY and either:
#   - YUWP_NOTARY_PROFILE, or
#   - YUWP_TEAM_ID + YUWP_APPLE_ID + YUWP_APP_PASSWORD
```

Sparkle is wired through the generated Info.plist by default. Override the
appcast URL or public key with `YUWP_SPARKLE_FEED_URL` or
`YUWP_SPARKLE_PUBLIC_ED_KEY` when needed.

### Standalone ASR server (no GUI)

```bash
.build/arm64-apple-macosx/release/yuwp-asr serve --model <model-dir> --transport http [--port 7936] [--host 127.0.0.1]
```

## Key Components

| File | Purpose |
|------|---------|
| App.swift | NSApplication entry, menu bar, orchestration |
| DictationSession.swift | Session state machine, protocol abstractions |
| NativeASRProvider.swift | Manages yuwp-asr serve process, stdio/HTTP STT sessions |
| ModelManager.swift | HF model resolution from cache or local paths |
| HotkeyManager.swift | Carbon global hotkey + Enter interception tap |
| AudioCapture.swift | AVAudioEngine → 16kHz mono PCM |
| AXTextInjector.swift | AX API text injection (preferred) |
| CGEventInjector.swift | CGEvent keyboard injection for terminals |
| ClipboardInjector.swift | Clipboard fallback for unsupported apps |
| TextInjectorFactory.swift | Selects injection strategy per app |
| MicPanel.swift | Floating mic indicator (NSPanel) |
| TypewriterAnimator.swift | Character-by-character text reveal |
| Config.swift | UserDefaults-based preferences |
| yuwp-asr/main.swift | ASR CLI entrypoint: transcribe and serve |
| ASRServerSupport/ | ASR server runtime, CLI parsing, HTTP/stdin routing |
| NativeASR/ | MLX model loading, inference, streaming session |

## HTTP API

When transport is HTTP, communication uses this API on `127.0.0.1:7936` by default:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/info` | Server status and model info |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible batch transcription |
| `POST` | `/v1/audio/transcriptions/stream` | Create session |
| `POST` | `/v1/audio/transcriptions/stream/:id` | Feed audio chunk (raw s16le PCM) |
| `DELETE` | `/v1/audio/transcriptions/stream/:id` | Stop session, get final text |

Override host/port with `--host` / `--port` flags.

## Code Quality

### Swift
- Swift 6 strict concurrency (swift-tools-version: 6.0)
- No SwiftUI — pure AppKit for minimal overhead
- No Xcode project — Swift Package only (`swift build` / `swift run`)
- Target macOS 14+
- No force unwraps in production code
- All async work on dedicated actors or `Task.detached`
- `@MainActor` for all UI, `@unchecked Sendable` for audio thread types
- CGEvent tap callback uses `nonisolated(unsafe)` static state
- MLX is NOT thread-safe — all inference MUST go through inferenceLock

## Complexity Guardrails

Yuwp is still intentionally small, even after the native ASR split. Resist adding new top-level app files unless a file exceeds ~400 lines or the boundary is clearly reusable.
Check the component table above before adding new files.

```bash
rg 'class |struct |enum |protocol ' Sources/*.swift
```

## Gotchas

- **Accessibility permission is tied to the code signing hash**, not the bundle ID.
  Every `swift build` produces a new binary hash. Use `scripts/build.sh` to
  codesign with a stable identifier.
- **Microphone permission is tied to process identity** (bundle ID or hash).
  Same re-granting issue. Use `scripts/run.sh` for a proper .app bundle.
- **Bluetooth audio devices** (e.g., AirPods) often deliver silent audio buffers
  when there's a codec conflict. Detect silent buffers and warn.
- **Bypass `AXIsProcessTrusted()` check** — attempt to create the CGEventTap
  directly. The check itself can return stale results.
- **Global dictation shortcut** uses Carbon hotkeys now. The CGEvent tap is only
  for swallowing Return during active dictation.
- **yuwp-asr must be built before running Yuwp** — the app launches `yuwp-asr serve`
  from the app bundle or `.build/arm64-apple-macosx/release/yuwp-asr`.

## Style

- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`
- Technical prose, direct
- Subject line under 72 chars

## Definition of Done

1. `swift build` succeeds with no warnings
2. `swift test` passes
3. `yuwp-asr serve --transport http` starts and responds to `/v1/info` with `"status": "ready"`
4. Integration tests pass: `swift test --filter "ASR Server"`
5. Tested: hotkey → record → transcribe → inject text (manual, end-to-end)
