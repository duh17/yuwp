# Benchmark Tooling

Use one front door:

```bash
uv run benchmarks/cli.py asr <scenario> [options]
```

## ASR scenarios

```bash
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
```

Scenario-specific help:

```bash
uv run benchmarks/cli.py asr load --help
uv run benchmarks/cli.py asr compare --help
```

Plot load benchmark JSON:

```bash
uv run benchmarks/cli.py plot /tmp/yuwp-load.json --out-dir /tmp/yuwp-load-plots
```

## Layout

```text
benchmarks/
  cli.py              # public entrypoint
  lib/asr.py          # scenario router
  lib/native_server.py      # implementation for `asr load`
  lib/stream_quality.py     # implementation for `asr quality`
  lib/batch_compare.py      # implementation for `asr compare`
  lib/openai_asr_compare.py # implementation for `asr remote`
  lib/subtitles.py          # implementation for `asr subtitles`
  lib/plot_native.py        # implementation for `plot`
```

Generated outputs should stay outside the repo by default, usually under `/tmp`.
