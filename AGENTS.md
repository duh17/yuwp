# Yuwp — Development Guide

Instructions for AI coding agents working on this codebase.

## Architecture

```
┌───────────────────────────────────────────┐
│            Yuwp.app (macOS)               │
│  Hotkey → AudioCapture → NativeASRProvider│
│                            ↕ HTTP :9748   │
│               TextInjector (AX API)       │
└───────────────────────┬───────────────────┘
                        │
           ┌────────────▼────────────┐
           │  asr-server (native)    │
           │  ┌──────────────────┐   │
           │  │ StreamingSession │   │
           │  │ (encoder cache,  │   │
           │  │  KV reuse,       │   │
           │  │  prefix rollback)│   │
           │  └──────────────────┘   │
           │     HTTP :9748          │
           │  (Yuwp + Oppi clients)  │
           └─────────────────────────┘
```

The native `asr-server` loads the MLX model once and serves HTTP.
Yuwp.app launches it as a child process, communicates via localhost HTTP.
External clients (Oppi) use the same HTTP API.

## Build & Run

```bash
# Build everything (app + server)
swift build
swift build -c release --product asr-server
bash scripts/build_mlx_metallib.sh release  # compile Metal shaders

swift run Yuwp
```

Use `scripts/build.sh` to codesign with a stable identifier (preserves
Accessibility/Microphone permissions across rebuilds). Use `scripts/run.sh`
to build and launch as a proper .app bundle with TCC-compatible Info.plist.

### Standalone ASR server (no GUI)

```bash
.build/arm64-apple-macosx/release/asr-server <model-dir> [--port 9748] [--host 127.0.0.1]
```

## Key Components

| File | Purpose |
|------|---------|
| App.swift | NSApplication entry, menu bar, orchestration |
| DictationSession.swift | Session state machine, protocol abstractions |
| NativeASRProvider.swift | Manages asr-server process, HTTP STT sessions |
| HotkeyManager.swift | Global hotkey via CGEvent tap |
| AudioCapture.swift | AVAudioEngine → 16kHz mono PCM |
| AXTextInjector.swift | AX API text injection (preferred) |
| CGEventInjector.swift | CGEvent keyboard injection for terminals |
| ClipboardInjector.swift | Clipboard fallback for unsupported apps |
| TextInjectorFactory.swift | Selects injection strategy per app |
| MicPanel.swift | Floating mic indicator (NSPanel) |
| TypewriterAnimator.swift | Character-by-character text reveal |
| Config.swift | UserDefaults-based preferences |
| asr-server/main.swift | Native HTTP streaming ASR server |
| NativeASR/ | MLX model loading, inference, streaming session |

## HTTP API

All communication uses a single HTTP protocol on `127.0.0.1:9748`:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/info` | Server status and model info |
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

Yuwp is small (~13 source files). Resist splitting unless a file exceeds ~400 lines.
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
- **Double-tap hotkey timing** — track tap timestamps manually. CGEvent key-down
  events are the source of truth; don't rely on NSEvent for global hotkeys.
- **asr-server must be built before running Yuwp** — the app locates the binary
  in `.build/arm64-apple-macosx/release/asr-server`.

## Style

- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`
- Technical prose, direct
- Subject line under 72 chars

## Definition of Done

1. `swift build` succeeds with no warnings
2. `swift test` passes
3. asr-server starts and responds to `/v1/info` with `"status": "ready"`
4. Integration tests pass: `swift test --filter "ASR Server"`
5. Tested: hotkey → record → transcribe → inject text (manual, end-to-end)
