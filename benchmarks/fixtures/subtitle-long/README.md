# Long-subtitle smoke fixture

This directory contains one deliberately small long-form regression fixture:

- **AMI EN2002b speaker D, 12:45–15:45**
- 3 minutes from one close-talking headset channel
- 10 human-adjusted transcript segments
- natural conversational disfluencies without competing speakers
- 1.1 MB AAC audio

The fixture is long enough to exercise repeated encoder windows without turning the repository into a corpus mirror. Use it for WER and output-structure smoke checks only. It is **not** timestamp or subtitle-segmentation gold: AMI word times are forced-alignment-derived, and the reference cues are human utterances rather than professional subtitle breaks.

The captured `spoken-v1` smoke baseline is **14.88% WER** (38 substitutions, 18 deletions, 23 insertions over 531 reference words per run). Subtitle output is structurally valid but violates the current readability policy heavily; see `baseline.json`. These numbers detect large local regressions only and must not decide whether a quality optimization ships.

## Run ASR quality and speed

```bash
uv run -m benchmarks.lib.asr_evaluate \
  --manifest benchmarks/fixtures/subtitle-long/asr-eval.jsonl \
  --tool yuwp \
  --yuwp-model "$HOME/Library/Application Support/Yuwp/models/mlx-community--Qwen3-ASR-0.6B-bf16"
```

## Run subtitle generation

Start the server with the forced aligner, then run the benchmark:

```bash
.build/arm64-apple-macosx/release/yuwp-asr serve \
  --model "$HOME/Library/Application Support/Yuwp/models/mlx-community--Qwen3-ASR-0.6B-bf16" \
  --aligner-model "$HOME/Library/Application Support/Yuwp/models/mlx-community--Qwen3-ForcedAligner-0.6B-8bit" \
  --transport http

uv run benchmarks/cli.py asr subtitles \
  --manifest benchmarks/fixtures/subtitle-long/subtitles-manifest.json
```

## Reference files

- `audio.m4a` — 16 kHz mono AAC excerpt.
- `human-segments.json` — human-adjusted AMI segment boundaries and text.
- `reference.srt` — the same utterance segments in SRT form, without speaker labels; diagnostic only, not subtitle-quality gold.
- `reference.txt` — spoken-word transcript used for WER.
- `fixture.json` — hashes, source offsets, license, alignment provenance, and explicit validity limits.
- `baseline.json` — captured ASR and subtitle-structure smoke metrics with binary/model provenance.

Speaker labels and bracketed non-speech events are excluded from the reference side of scoring. Hypothesis annotations remain scoreable, so a model cannot hide insertions by emitting bracketed text. The WER normalizer also treats AMI forms such as `X_M_L_` as `XML` and removes apostrophes inside words because they are not spoken boundaries.

## Attribution

Derived from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), meeting `EN2002b`. AMI signals and transcription are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
