# Benchmark Tooling

This directory is the single home for benchmark orchestration, benchmark fixtures,
and benchmark-specific helper modules.

Use the front door:

```bash
uv run benchmarks/cli.py <subcommand> [options]
```

Subcommands:

- `compare` — cross-tool batch comparisons (`yuwp`, `mlx-audio`, `qwen-asr`)
- `server-load` — concurrent HTTP server benchmark
- `stream-quality` — streaming canary / quality benchmark
- `subtitles` — timed subtitle-output benchmark
- `plot-native` — render plots from native server benchmark JSON

Layout:

```text
benchmarks/
  README.md
  cli.py
  fixtures/
    subtitles-manifest.example.json
  lib/
    batch_compare.py
    native_server.py
    stream_quality.py
    subtitles.py
    plot_native.py
```

Generated run outputs should stay outside the repo by default, typically under
`/tmp/...` paths, or in local-only benchmark work directories that remain ignored.
