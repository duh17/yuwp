# Yuwp — System-Wide Voice Dictation for macOS

Menu bar app that streams microphone audio to a local ASR engine and live-types
transcription results into any focused text field.

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
         │  (Yuwp)     (clients)    │
         └─────────────────────────┘
```

The Python sidecar loads the model once and runs in dual mode:
- **stdio** — JSON lines for Yuwp.app's local dictation
- **HTTP** — streaming session API on localhost:9748 for external clients

No server dependency. The sidecar runs mlx-audio directly with optimized
streaming: encoder window caching, decoder KV reuse, prefix rollback.

## Build & Run

```bash
cd ~/workspace/yuwp
swift build
swift run Yuwp           # menu bar app (stdio + HTTP server)
```

### Standalone ASR server (no GUI)

```bash
uv run --script Sources/sidecar/transcribe.py --serve-only
# Listens on http://127.0.0.1:9748
```

## Key Components

| File | Purpose |
|------|---------|
| App.swift | NSApplication entry, menu bar, orchestration |
| HotkeyManager.swift | Global hotkey via CGEvent tap |
| AudioCapture.swift | AVAudioEngine → 16kHz mono PCM |
| ASRSidecar.swift | Manages Python sidecar process (stdio + HTTP) |
| TextInjector.swift | AX API text injection + clipboard fallback |
| MicPanel.swift | Floating mic indicator (NSPanel) |
| TypewriterAnimator.swift | Character-by-character text reveal |
| Config.swift | UserDefaults-based preferences |
| sidecar/transcribe.py | Streaming ASR engine (mlx-audio, dual-mode) |

## Permissions

- **Accessibility** — required for hotkey (CGEvent tap) and text injection (AXUIElement)
- **Microphone** — required for audio capture

On first launch, the app prompts for Accessibility permission and polls every 2s
until granted. The menu bar shows status.

## Hotkey Modes

Two modes, configurable via menu bar:

| Mode | Default | How it works |
|------|---------|--------------|
| **Double-tap** | Right ⌥ | Tap a modifier key twice within 400ms. Clean taps only — using the key as an actual modifier doesn't trigger. |
| **Combo** | Ctrl+` | Standard modifier + key combination. Event is swallowed. |

Double-tap uses device-specific flag masks (NX_DEVICE*KEYMASK) to distinguish
left vs right modifier keys. The state machine tracks "dirty" taps where another
key was pressed during the modifier hold.

## Sidecar Protocol

### Stdio (Yuwp.app)

JSON lines over stdin/stdout:
- **Swift → Python**: `start`, `audio` (base64 PCM), `stop`, `quit`
- **Python → Swift**: `ready`, `partial`, `final`, `error`

### HTTP (external clients, external clients)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions/stream` | Create session |
| `POST` | `/v1/audio/transcriptions/stream/:id` | Feed audio chunk (raw PCM) |
| `DELETE` | `/v1/audio/transcriptions/stream/:id` | Stop session, get final text |

Default: `127.0.0.1:9748`. Override with `--host` / `--port`.

## HTTP Integration

Yuwp's HTTP server is the ASR backend for external clients. In external's config:

```json
{
  "asr": {
    "sttEndpoint": "http://localhost:9748"
  }
}
```

When running, connected apps receive transcription automatically.
When not running, connected apps can fall back to other STT backends.

## Text Injection

Two strategies:
1. **AX API** (preferred) — reads/writes focused element via Accessibility
2. **Clipboard fallback** — saves clipboard, pastes, restores

Live streaming only works with AX API. Clipboard fallback commits final text only.
The injector probes the target element at capture time and degrades gracefully
mid-session if AX writes fail.

## Silence Handling

Three layers prevent hallucinated text during silence:
1. **Sidecar**: RMS energy < 0.003 → skip inference entirely (0ms)
2. **Sidecar**: `if new_tokens:` guard — silence can't eat confirmed text via rollback
3. **Swift**: Empty or "None" partials filtered before reaching typewriter

## Code Quality

### Swift
- Swift 6 strict concurrency (swift-tools-version: 6.0)
- No SwiftUI — pure AppKit for minimal overhead
- No Xcode project — Swift Package only (`swift build` / `swift run`)
- Target macOS 14+
- No force unwraps in production code
- All async work on dedicated actors or `Task.detached` — AppKit main thread stays responsive
- `@MainActor` for all UI, `@unchecked Sendable` for audio thread types
- CGEvent tap callback uses `nonisolated(unsafe)` static state (no captures in C function pointers)
- Prefer `if let x` over `if let x = x`

### Python (sidecar)
- Type hints on all public functions
- `uv run --script` for execution — no manual venv
- Prefer `ruff` for linting/formatting

## Complexity Guardrails

Before writing new code, search for existing implementations:
```bash
# Swift components
rg 'class |struct |enum |protocol ' Sources/*.swift
# Sidecar protocol messages
rg '"type"' Sources/ASRSidecar.swift Sources/sidecar/transcribe.py
# Config keys
rg 'UserDefaults\|@AppStorage\|Config\.' Sources/Config.swift
```

Yuwp is small (~9 files). Resist splitting unless a file exceeds ~400 lines. Check the component table above before adding new files.

## Protocol Discipline

Yuwp has two protocol surfaces — stdio JSON (Swift ↔ Python) and HTTP REST (Python ↔ external clients). When changing message contracts:

1. Update Python message handling in `Sources/sidecar/transcribe.py`
2. Update Swift types in `Sources/ASRSidecar.swift`
3. Update HTTP endpoints if the change affects external clients
4. Update the protocol tables in this file

No partial protocol updates — both sides must stay in sync.

## Gotchas

- **Accessibility permission is tied to the code signing hash**, not the bundle ID. Every `swift build` produces a new binary hash, requiring re-granting Accessibility permission. Use `codesign --force --sign -` to stabilize the hash, or run from a fixed build path.
- **Microphone permission is tied to process identity** (bundle ID or hash). Same re-granting issue as Accessibility on every rebuild.
- **Bluetooth audio devices** (e.g., AirPods) often deliver silent audio buffers when there's a codec conflict between output (AAC) and input (SCO). Detect silent buffers and warn the user rather than transcribing silence.
- **Bypass `AXIsProcessTrusted()` check** — attempt to create the `CGEventTap` directly. The check itself can return stale results; the tap creation is the ground truth.
- **Double-tap hotkey timing** — track tap timestamps and key codes manually. `CGEventMonitor` key-down events are the source of truth; don't rely on `NSEvent` for global hotkeys.
- **`os.log .info` not persisted** in device archives — use `.error` level for diagnostics you need to see in Console.app after the fact.

## Style

- No emojis in commits or code
- Technical prose, direct
- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`

## Definition of Done

1. `swift build` succeeds with no warnings
2. Sidecar starts and responds to `ready` handshake
3. Protocol changes are mirrored in both Swift and Python with updated docs
4. Tested: hotkey → record → transcribe → inject text (manual, end-to-end)
5. HTTP endpoints tested if changed (`curl` smoke test minimum)
