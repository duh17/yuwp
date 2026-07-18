# Yuwp Crash Signatures

Known failure patterns seen during development.

## Contents

- [Sparkle framework missing at launch](#1-sparkle-framework-missing-at-launch)
- [Accessibility prompt loop despite prior approval](#2-accessibility-prompt-loop-despite-prior-approval)
- [Crash on key press while dictation pill is visible](#3-crash-on-key-press-while-dictation-pill-is-visible)
- [AVAudio installTap format mismatch](#4-avaudio-installtap-format-mismatch)
- [Dead mic or silent buffers](#5-dead-mic-or-silent-buffers)
- [Server healthy but app missing model](#6-server-healthy-but-app-missing-model)

## 1. Sparkle framework missing at launch

### Symptom

App dies at launch with a dyld error like:

```text
Library not loaded: @rpath/Sparkle.framework/Versions/B/Sparkle
```

### Where to look

- `/Applications/Yuwp.app/Contents/Frameworks/Sparkle.framework`
- `otool -l /Applications/Yuwp.app/Contents/MacOS/Yuwp | rg LC_RPATH -A2 -B1`
- latest `Yuwp-*.ips`

### Likely cause

- missing `@executable_path/../Frameworks` rpath
- bad bundle assembly
- nested or corrupted `Sparkle.framework`

## 2. Accessibility prompt loop despite prior approval

### Symptom

Yuwp keeps asking for Accessibility again, or `CGEvent.tapCreate(...)` fails repeatedly.

### Where to look

- `codesign -dv --verbose=4 /Applications/Yuwp.app 2>&1`
- TCC rows in `~/Library/Application Support/com.apple.TCC/TCC.db`
- `/tmp/yuwp.log`

### Likely cause

- app was ad-hoc signed or signed with a different identity than the previously granted build
- app was launched outside `scripts/run.sh`

### Rule

Accessibility grants are tied to code identity. Stable bundle path alone is not enough.

## 3. Crash on key press while dictation pill is visible

### Symptom

Pressing a key during dictation crashes on the main thread.

### Where to look

- latest `Yuwp-*.ips`
- `Sources/MicPanel.swift`
- `/tmp/yuwp.log`

### Signature seen

Crash frame in or near:
- `closure #1 in MicPanel.startEscapeMonitor()`
- `swift_task_isMainExecutorImpl`
- `swift_getObjectType`

### Root cause seen

A global `NSEvent` monitor callback touched `@MainActor`-isolated state indirectly. The current fix captures the dismiss closure directly instead of referencing `MicPanel` inside the monitor callback.

## 4. AVAudio installTap format mismatch

### Symptom

Dictation start crashes or throws an AppKit or AVFAudio exception with text like:

```text
Format mismatch: input hw <... 24000 Hz ...>, client format <... 48000 Hz ...>
Failed to create tap due to format mismatch
```

### Where to look

- unified log (`log show ...`) rather than only `/tmp/yuwp.log`
- `Sources/AudioCapture.swift`
- latest `Yuwp-*.ips`

### Why it matters

The thrown exception can happen before the app logs a friendly failure path. This is usually an audio-engine, route, or hardware-format issue.

## 5. Dead mic or silent buffers

### Symptom

Yuwp starts dictation, records about 2 seconds, then auto-stops with zero text.

### Where to look

- `/tmp/yuwp.log`
- recorded WAV in `~/Library/Application Support/Yuwp/recordings/`
- unified logs for CoreAudio route changes

### Signature

```text
Warning: 2.0s of dead silence — mic may not be working
Dead mic detected (2.0s silence) — stopping session
```

### Likely cause

Bluetooth codec conflict or bad input route. Not necessarily an ASR bug.

## 6. Server healthy but app missing model

### Symptom

App says model missing, or launch never reaches ready state on a fresh Mac.

### Where to look

- `Sources/ModelManager.swift`
- `~/Library/Application Support/Yuwp/models/`
- `~/.cache/huggingface/hub/`
- `/tmp/yuwp.log`

### Rule

Model downloads are manual only. Missing-model state is expected until the user explicitly downloads one from the menu.
