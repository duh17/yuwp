# Yuwp — Development Guide

Instructions for AI coding agents working on this codebase.

## Architecture

```
┌─────────────────────────────────────┐
│           Yuwp.app (macOS)        │
│  Hotkey → AudioCapture → ASRSidecar │
│                         ↕ stdio     │
│              TextInjector (AX API)  │
└─────────────────────┬───────────────┘
                      │
         ┌────────────▼────────────┐
         │  transcribe.py sidecar  │
         │  ┌──────────────────┐   │
         │  │ StreamSession    │   │
         │  │ (encoder cache,  │   │
         │  │  KV reuse,       │   │
         │  │  prefix rollback)│   │
         │  └────────┬─────────┘   │
         │     ┌─────┴──────┐      │
         │  stdio JSON   HTTP :9748│
         │  (Yuwp)    (clients)  │
         └─────────────────────────┘
```

The Python sidecar loads the model once and runs in dual mode:
- **stdio** — JSON lines for Yuwp.app's local dictation
- **HTTP** — streaming session API on localhost:9748 for external clients

No server dependency. The sidecar runs mlx-audio directly with optimized
streaming: encoder window caching, decoder KV reuse, prefix rollback.

## Build & Run

```bash
swift build
swift run Yuwp
```

Use `scripts/build.sh` to codesign with a stable identifier (preserves
Accessibility/Microphone permissions across rebuilds). Use `scripts/run.sh`
to build and launch as a proper .app bundle with TCC-compatible Info.plist.

### Standalone ASR server (no GUI)

```bash
uv run --script Sources/sidecar/transcribe.py --serve-only
# Listening on http://127.0.0.1:9748
```

## Key Components

| File | Purpose |
|------|---------|
| App.swift | NSApplication entry, menu bar, orchestration |
| DictationSession.swift | Session state machine, protocol abstractions |
| HotkeyManager.swift | Global hotkey via CGEvent tap |
| AudioCapture.swift | AVAudioEngine → 16kHz mono PCM |
| ASRSidecar.swift | Manages Python sidecar process (stdio + HTTP) |
| AXTextInjector.swift | AX API text injection (preferred) |
| CGEventInjector.swift | CGEvent keyboard injection for terminals |
| ClipboardInjector.swift | Clipboard fallback for unsupported apps |
| TextInjectorFactory.swift | Selects injection strategy per app |
| MicPanel.swift | Floating mic indicator (NSPanel) |
| TypewriterAnimator.swift | Character-by-character text reveal |
| Config.swift | UserDefaults-based preferences |
| sidecar/transcribe.py | Streaming ASR engine (mlx-audio, dual-mode) |

## Sidecar Protocol

### Stdio (Yuwp.app)

JSON lines over stdin/stdout:
- **Swift → Python**: `start`, `audio` (base64 PCM), `stop`, `quit`
- **Python → Swift**: `ready`, `partial`, `final`, `error`

### HTTP (external clients)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions/stream` | Create session |
| `POST` | `/v1/audio/transcriptions/stream/:id` | Feed audio chunk (raw PCM) |
| `DELETE` | `/v1/audio/transcriptions/stream/:id` | Stop session, get final text |
| `GET` | `/v1/info` | Model info and server status |

Default: `127.0.0.1:9748`. Override with `--host` / `--port`.

## Protocol Discipline

Yuwp has two protocol surfaces — stdio JSON (Swift ↔ Python) and HTTP REST
(Python ↔ external clients). When changing message contracts:

1. Update Python message handling in `Sources/sidecar/transcribe.py`
2. Update Swift types in `Sources/ASRSidecar.swift`
3. Update HTTP endpoints if the change affects external clients
4. Update the protocol tables in this file

No partial protocol updates — both sides must stay in sync.

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

### Python (sidecar)
- Type hints on all public functions
- `uv run --script` for execution — no manual venv
- `ruff` for linting/formatting

## Complexity Guardrails

Yuwp is small (~13 files). Resist splitting unless a file exceeds ~400 lines.
Check the component table above before adding new files.

```bash
rg 'class |struct |enum |protocol ' Sources/*.swift
rg '"type"' Sources/ASRSidecar.swift Sources/sidecar/transcribe.py
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

## Style

- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`
- Technical prose, direct
- Subject line under 72 chars

## Definition of Done

1. `swift build` succeeds with no warnings
2. `swift test` passes
3. Sidecar starts and responds to `ready` handshake
4. Protocol changes mirrored in both Swift and Python
5. Tested: hotkey → record → transcribe → inject text (manual, end-to-end)
