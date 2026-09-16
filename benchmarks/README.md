# Benchmark Tooling

Use one front door:

```bash
uv run benchmarks/cli.py asr <scenario> [options]
```

## ASR scenarios

```bash
# Headline numbers used in the README (hot batch + live stdio + CLI)
uv run benchmarks/cli.py asr headline
# Writes benchmarks/fixtures/public-jfk.json

# Local server load / latency / memory across models and concurrency
uv run benchmarks/cli.py asr load small --balanced 5 --concurrency 1 2 4 --json /tmp/yuwp-load.json

# Streaming quality canary against known fixtures
uv run benchmarks/cli.py asr quality --case jfk

# Compare yuwp-asr with local tools such as mlx-audio or qwen-asr
uv run benchmarks/cli.py asr compare --audio Tests/fixtures/jfk.wav --yuwp-model small

# Compare local yuwp-asr with an OpenAI-compatible remote ASR endpoint
uv run benchmarks/cli.py asr remote --remote-url http://127.0.0.1:8000

# Exercise timed subtitle output
uv run benchmarks/cli.py asr subtitles --file sample.wav

# Materialize pinned local-only long-form dev/heldout data
uv run benchmarks/cli.py asr prepare-long-form earnings21
```

README clocks (do not mix with cloud “time to final after end of speech”):

- Headline first text: `jfk.wav` energy onset (0.30 s) → first nonempty stdio partial
- Headline finalize: stop request → final transcript
- Transcribe: `yuwp-asr transcribe` wall time on `Tests/fixtures/jfk.wav`
- Speak: `yuwp-tts` wall time to write the WAV (Studio only)
- The 505 ms Studio 1.7B figure is a separate Silero speech-onset canary, not the jfk energy-onset clock

Long-form evaluator runs use explicit batch modes (`vad`, `energy`, or
`automatic`); production `automatic` keeps VAD for inputs up to 120 seconds and
uses energy boundaries for longer batch/subtitle inputs. Explicit `vad` falls
back to energy if batch VAD cannot be loaded. Use `--yuwp-batch-chunking energy`
for a batch-only experiment rather than `--disable-vad`, which also disables
live-stream VAD for legacy compatibility. Yuwp RTF uses the server-reported
decoded duration; other tools retain their existing duration sources. The
compact dev-only result record is
`fixtures/subtitle-external/earnings21-batch-chunking.json`.

Scenario-specific help:

```bash
uv run benchmarks/cli.py asr load --help
uv run benchmarks/cli.py asr compare --help
```

Plot load benchmark JSON:

```bash
uv run benchmarks/cli.py plot /tmp/yuwp-load.json --out-dir /tmp/yuwp-load-plots
```

## Dev-only English diagnostics

`english-parity` is **NON-ACCEPTANCE**: one Qwen/N1/P1 pass, at most 81
trials, no heldout inference or retries. Read
[the dev-only scope lock](fixtures/english-parity/dev-only-addendum.md) and
[the protocol](fixtures/english-parity/protocol.md) before execution.

Supply a JSON object with exactly `Qwen`, `Nemotron`, and `Parakeet` keys.
Each value is the complete executable-and-arguments array for its approved
isolated stdio process; no shell expansion or default model command is used.

```bash
# Plan only: does not spawn a process or read audio payloads.
uv run benchmarks/cli.py asr english-parity \
  --commands /tmp/english-commands.json --output /tmp/english-dev.jsonl

# Only after owner/reviewer sign-off for diagnostic-only work under recorded load:
uv run benchmarks/cli.py asr english-parity \
  --commands /tmp/english-commands.json --output /tmp/english-dev.jsonl \
  --execute --load-status diagnostic-under-load
```

The caller must record approved assets/binaries/settings and control machine
clearance, competing benchmarks, and continuous telemetry. The runner records
load snapshots, not load control; `no_load_control: true` remains explicit.
It uses fresh processes, an unscored first-dev-short warm-up, a clean measured
session, packet-end deadlines, and a four-hour total budget (`--max-seconds`
can shorten it). Output is a new JSONL file with raw framed events, failures,
transcripts, descriptive paired WER/C/S/D/I and latency summaries. Stderr goes
to adjacent per-trial files. Missing metrics stay null; no confidence or
promotion claim is made. Final-pass runtime is unavailable in common ASRIPC.

For approved fake-peer transport checks, add `--case-limit 1 --max-seconds 10
--load-status uncontrolled --execute`; any case limit labels the run as
transport smoke, never a completed dev pass. The default load status blocks
execution. Exit 1 means failed/missing trials; exit 2 means blocked execution
or invalid arguments. No model smoke is authorized by this documentation.

## Layout

```text
benchmarks/
  cli.py              # public entrypoint
  lib/asr.py          # scenario router
  lib/headline.py     # implementation for `asr headline`
  fixtures/public-jfk.json  # last headline numbers used in the README
  lib/native_server.py      # implementation for `asr load`
  lib/stream_quality.py     # implementation for `asr quality`
  lib/batch_compare.py      # implementation for `asr compare`
  lib/openai_asr_compare.py # implementation for `asr remote`
  lib/subtitles.py          # implementation for `asr subtitles`
  lib/external_benchmark_data.py # local-only long-form dataset materializer
  fixtures/subtitle-external/    # pinned plans and data-handling documentation
  lib/plot_native.py        # implementation for `plot`
```

Generated outputs should stay outside the repo by default, usually under `/tmp`.
