---
name: yuwp-dev
description: Triage Yuwp.app and its bundled ASR server on Chen's Macs. This skill should be used when checking app or server state, localhost health, crash reports, TCC/signing problems, packaging, or bundle contents.
---

# Yuwp Dev

Operate from `~/workspace/yuwp`.

## Default workflow

Run the command hub first:

```bash
{baseDir}/scripts/yuwp-workflow.py
# structured
{baseDir}/scripts/yuwp-workflow.py --json status
# CLI help
{baseDir}/scripts/yuwp-workflow.py --help
```

Command hub behavior:
- bare command prints a catch-up overview
- human-readable output is the default
- `--json` and `--compact` provide structured output
- stdout is data; actionable failures go to stderr

Subcommands:
- `status` — app, server, signing, health, TCC, latest crash
- `paths` — important paths with existence checks
- `logs` — `/tmp/yuwp.log`
- `unified` — recent macOS unified log lines for `Yuwp` and `asr-server`
- `crash` — latest crash report path(s)
- `tcc` — Accessibility and Microphone permission rows
- `defaults` — persisted `UserDefaults` for `com.yuwp.app`
- `health` — `GET /v1/info`

## Validation loop

Copy and track progress:
- [ ] Run `status` or `health` for a baseline
- [ ] Pick the narrowest lane below
- [ ] Validate with logs, crash data, or `/v1/info`
- [ ] Return the collaboration output contract

## Workflow lanes

### Lane 1 — Launch the signed app bundle

Run:

```bash
cd ~/workspace/yuwp
bash scripts/run.sh
```

Use for:
- Accessibility or Microphone behavior
- Sparkle embedding or bundle layout
- any issue that depends on a real `.app` bundle

### Lane 2 — Catch up on current state

Run:

```bash
{baseDir}/scripts/yuwp-workflow.py
{baseDir}/scripts/yuwp-workflow.py --json status
```

Use first for questions like:
- “what’s running?”
- “is the server healthy?”
- “why is Yuwp being weird?”

### Lane 3 — Check logs

Run:

```bash
# app stdout/stderr captured by scripts/run.sh
{baseDir}/scripts/yuwp-workflow.py logs
{baseDir}/scripts/yuwp-workflow.py logs 200

# system-side failures
{baseDir}/scripts/yuwp-workflow.py unified
{baseDir}/scripts/yuwp-workflow.py unified 30m 200
```

Interpretation:
- use `logs` for explicit `yuwpLog(...)` lines
- use `unified` for AVFAudio, AppKit, dyld, CoreAudio, and CFNetwork failures

### Lane 4 — Triage crashes

Run:

```bash
{baseDir}/scripts/yuwp-workflow.py crash
{baseDir}/scripts/yuwp-workflow.py --json crash
```

After locating the newest `.ips`, read that exact file path. Avoid broad searches under `~/Library`.

### Lane 5 — Debug permissions and signing

Run:

```bash
{baseDir}/scripts/yuwp-workflow.py tcc
{baseDir}/scripts/yuwp-workflow.py defaults
{baseDir}/scripts/yuwp-workflow.py status
codesign -dv --verbose=4 /Applications/Yuwp.app 2>&1 | sed -n '1,40p'
```

Use for:
- Accessibility prompt loops
- Microphone permission confusion
- “I already granted this” reports
- ad-hoc vs Developer ID signing issues

Critical rule: Accessibility grants are tied to code identity, not just bundle ID. Prefer `scripts/run.sh` over raw `swift run Yuwp` when testing permissions.

### Lane 6 — Debug server and models

Run:

```bash
{baseDir}/scripts/yuwp-workflow.py health
curl -sf http://127.0.0.1:9748/v1/info
```

Use for:
- model missing or wrong model loaded
- server not listening on `127.0.0.1:9748`
- subprocess adoption or crash recovery questions
- batch retranscribe state

Read for deeper context:
- `references/debug-paths.md`
- `references/crash-signatures.md`

### Lane 7 — Check packaging and release artifacts

Run:

```bash
cd ~/workspace/yuwp
bash scripts/run.sh
scripts/release.sh <version>
```

Use for:
- verifying `Yuwp`, `asr-server`, `mlx.metallib`, and `Sparkle.framework`
- checking signing identities and runtime paths
- release DMG assembly and notarization prep

## Fast ops checks

```bash
# App + server processes
pgrep -fl '/Applications/Yuwp.app/Contents/MacOS/Yuwp|/Applications/Yuwp.app/Contents/MacOS/asr-server'

# Port listener
lsof -iTCP:9748 -sTCP:LISTEN

# Health
curl -sf http://127.0.0.1:9748/v1/info

# App log
tail -n 120 /tmp/yuwp.log
```

## Key gotchas

- `scripts/run.sh` is the real dev launch path. Raw `swift run Yuwp` is fine for quick local hacking but bad for TCC stability.
- `scripts/run.sh` auto-detects a local `Developer ID Application` identity. If none is found, it falls back to ad-hoc signing and permissions may re-prompt.
- Yuwp does not auto-download models. Missing-model state on a fresh Mac is expected until the user explicitly downloads one from the menu.
- `log show` can reveal failures that never hit `/tmp/yuwp.log`, especially AVFAudio exceptions and dyld loader errors.
- `asr-server` is a child subprocess of `Yuwp.app`. If the app is gone but the server remains, inspect PPID and orphaning before assuming the server is healthy.
- Avoid `find ~/...` under TCC-protected folders. Use targeted `ls` or the workflow script.

## Collaboration output contract

After each lane run, return:

1. Lane used
2. Commands run exactly
3. Artifacts inspected or created with absolute paths
4. Result: `PASS` or `FAIL` with reason
5. Next action: one concrete step

## References

- `references/debug-paths.md`
- `references/crash-signatures.md`
