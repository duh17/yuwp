# Yuwp Debug Paths

Absolute locations and why they matter.

## Contents

- [Runtime + Bundle](#runtime--bundle)
- [Logs](#logs)
- [Permissions + Identity](#permissions--identity)
- [Models + User Data](#models--user-data)
- [HTTP Endpoints](#http-endpoints)
- [Build Outputs](#build-outputs)
- [Useful Commands](#useful-commands)

## Runtime + Bundle

| What | Path | Notes |
|------|------|-------|
| Repo root | `~/workspace/yuwp` | Main source tree |
| Installed app bundle | `/Applications/Yuwp.app` | Built by `scripts/run.sh` |
| Main executable | `/Applications/Yuwp.app/Contents/MacOS/Yuwp` | Check codesign and rpaths |
| Bundled server | `/Applications/Yuwp.app/Contents/MacOS/asr-server` | Child subprocess launched by app |
| Bundled Metal lib | `/Applications/Yuwp.app/Contents/MacOS/mlx.metallib` | Required for MLX runtime |
| Bundled framework | `/Applications/Yuwp.app/Contents/Frameworks/Sparkle.framework` | Sparkle runtime |
| Release app bundle | `~/workspace/yuwp/release/Yuwp.app` | Created by `scripts/release.sh` |
| Release DMG | `~/workspace/yuwp/release/Yuwp-<version>.dmg` | Notarized release artifact |
| Appcast | `~/workspace/yuwp/release/appcast.xml` | Sparkle feed metadata |

## Logs

| What | Path / Command | Notes |
|------|----------------|-------|
| App log | `/tmp/yuwp.log` | Captured by `scripts/run.sh`; contains `yuwpLog(...)` and forwarded server stderr |
| Unified logs | `log show --style compact --last 10m --predicate 'process == "Yuwp" OR process == "asr-server"'` | Best source for AVFAudio, AppKit, dyld, CoreAudio, and CFNetwork issues |
| Crash reports dir | `~/Library/Logs/DiagnosticReports/` | Contains `Yuwp-*.ips` and `asr-server-*.ips` |
| Latest Yuwp crash | `ls -1t ~/Library/Logs/DiagnosticReports/Yuwp-*.ips | head -n 1` | Read the exact file path |
| Latest server crash | `ls -1t ~/Library/Logs/DiagnosticReports/asr-server-*.ips | head -n 1` | Same workflow |

## Permissions + Identity

| What | Path / Command | Notes |
|------|----------------|-------|
| TCC database | `~/Library/Application Support/com.apple.TCC/TCC.db` | Query with `sqlite3`; do not edit directly |
| App defaults domain | `com.yuwp.app` | Backed by `UserDefaults.standard` |
| Dump defaults | `defaults export com.yuwp.app - | plutil -convert json -o - -` | Preferred machine-readable view |
| Codesign summary | `codesign -dv --verbose=4 /Applications/Yuwp.app 2>&1` | Inspect authority, Team ID, runtime |
| Gatekeeper assessment | `spctl --assess --type execute -vv /Applications/Yuwp.app` | Check whether the release bundle is accepted |

Useful TCC query:

```bash
sqlite3 "$HOME/Library/Application Support/com.apple.TCC/TCC.db" \
  "select service,client,client_type,auth_value,auth_reason,flags,last_modified from access where client='com.yuwp.app';"
```

Look for:
- `kTCCServiceAccessibility`
- `kTCCServiceMicrophone`

`auth_value = 2` means granted.

## Models + User Data

| What | Path | Notes |
|------|------|-------|
| Managed model root | `~/Library/Application Support/Yuwp/models/` | Explicitly downloaded models live here |
| Recording archive | `~/Library/Application Support/Yuwp/recordings/` | WAV files saved by Yuwp |
| Latest saved dictation | `ls -1t ~/Library/Application\ Support/Yuwp/recordings/yuwp-*.wav | head -n 1` | Fast path to the newest local recording |
| Hugging Face cache | `~/.cache/huggingface/hub/` | Fallback model resolution path |
| Typical HF model dir | `~/.cache/huggingface/hub/models--mlx-community--Qwen3-ASR-*/snapshots/<sha>/` | Resolved by `ModelLocator` |

## HTTP Endpoints

All local server traffic uses `127.0.0.1:9748` by default.

| Method | URL | Purpose |
|--------|-----|---------|
| `GET` | `http://127.0.0.1:9748/v1/info` | Health and model info |
| `POST` | `http://127.0.0.1:9748/v1/audio/transcriptions/stream` | Create streaming session |
| `POST` | `http://127.0.0.1:9748/v1/audio/transcriptions/stream/:id` | Feed PCM chunk |
| `DELETE` | `http://127.0.0.1:9748/v1/audio/transcriptions/stream/:id` | Finalize session |
| `POST` | `http://127.0.0.1:9748/v1/audio/transcriptions` | Batch transcription |

## Build Outputs

| What | Path |
|------|------|
| Release binary dir | `~/workspace/yuwp/.build/arm64-apple-macosx/release/` |
| Debug binary dir | `~/workspace/yuwp/.build/arm64-apple-macosx/debug/` |
| Release app binary | `~/workspace/yuwp/.build/arm64-apple-macosx/release/Yuwp` |
| Release server binary | `~/workspace/yuwp/.build/arm64-apple-macosx/release/asr-server` |
| Release Metal lib | `~/workspace/yuwp/.build/arm64-apple-macosx/release/mlx.metallib` |

## Useful Commands

### Catch-up overview

```bash
{baseDir}/scripts/yuwp-workflow.py
```

### Health

```bash
curl -sf http://127.0.0.1:9748/v1/info
```

### Process tree

```bash
pgrep -fl '/Applications/Yuwp.app/Contents/MacOS/Yuwp|/Applications/Yuwp.app/Contents/MacOS/asr-server'
ps -p <server-pid> -o pid,ppid,comm,args
```

### Logs

```bash
tail -n 200 /tmp/yuwp.log
log show --style compact --last 15m --predicate 'process == "Yuwp" OR process == "asr-server"' | tail -n 200
```

### Crash reports

```bash
ls -1t "$HOME/Library/Logs/DiagnosticReports" | rg '^Yuwp|^asr-server' | head
```

### Defaults

```bash
defaults export com.yuwp.app - | plutil -convert json -o - -
```
