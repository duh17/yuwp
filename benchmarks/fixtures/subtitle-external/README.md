# External long-form benchmark plan

`benchmark-plan.json` pins dataset revisions, cases, splits, hashes, and data-handling rules. `earnings21-baseline.json` records the compact development result without transcripts or audio. Corpus files and raw run output are downloaded under `benchmarks/data/`, which Git ignores.

## Earnings-21

Materialize or verify the four pinned calls:

```bash
# Download missing assets, verify hashes, and build local manifests
uv run benchmarks/cli.py asr prepare-long-form earnings21

# Verify without network access
uv run benchmarks/cli.py asr prepare-long-form earnings21 --check-only
```

Generated local manifests:

- `benchmarks/data/long-subtitle/earnings21/asr-dev.jsonl`
- `benchmarks/data/long-subtitle/earnings21/asr-heldout.jsonl`

The split is deliberately small and call-disjoint:

| Split | Calls | Duration | Purpose |
|---|---|---:|---|
| dev | 4346818, 4366522 | 96.5 min | tuning and repeated comparisons |
| heldout | 4363614, 4386541 | 111.4 min | confirmation only |

The three-repeat dev baseline with Qwen3-ASR-0.6B-bf16 is **15.86% corpus WER per repeat**: 1,097 substitutions, 939 deletions, and 405 insertions over 15,394 normalized reference words. Using the server-reported decoded durations (not MP3 container metadata), aggregate corpus RTF was 0.0278 after a full-call warm-up, with per-call RTF ranging from 0.0172 to 0.0372. Performance comparisons must therefore be paired and repeated; quality output was deterministic across all three runs. The result remains local under `benchmarks/data/`. The companion `earnings21-batch-chunking.json` records paired VAD-versus-automatic subtitle-structure counters for the same two development calls; those counters are not timestamp-quality gold.

Earnings-21 text files are CC BY-SA 4.0. The repository does not state an equivalent license for the audio, so audio remains local and must not be redistributed. Angle-bracket placeholders such as `<inaudible>` and `<crosstalk>` are excluded while building spoken-word references; exclusion counts are recorded beside each generated reference.

Do not run or inspect the held-out hypotheses while choosing implementations. After five decision uses, replace the held-out selection.

## MuST-Cinema and human timing

MuST-Cinema remains the planned subtitle-break layer, but unattended download is currently blocked:

- the original FBK download URL now resolves to a generic site;
- the available OpenDataLab mirror requires manual or authenticated access;
- the mirror labels the corpus CC BY-NC-ND 4.0.

Check the recorded status:

```bash
uv run benchmarks/cli.py asr prepare-long-form must-cinema-status
```

Do not substitute automatically aligned timestamps as human gold. Once the source archive is available, import three development talks and the five-talk test set, then hand-time roughly ten one-minute spans for absolute word-boundary evaluation.
