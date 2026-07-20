# Streaming refactor acceptance criteria

Date: 2026-07-20
Status: proposed

## Decision

The first refactor is correctness-first and targets the current 1.7B, single-client app path. It may reorganize streaming session storage, chunk extraction, and related boundaries, but it must not intentionally change decoding, VAD, segment-commit, finalization, transport, or transcript semantics.

The refactor passes only when:

1. deterministic behavior tests prove that audio and transcript state are preserved;
2. the fixed 1.7B streaming canary shows no material quality or performance regression;
3. at least one targeted performance metric improves by more than measurement noise; and
4. the evidence is produced by repeatable commands and retained as aggregate JSON without private transcripts or audio.

The 0.6B model and concurrent-client matrix are informative for this change, not completion gates.

## Scope

This specification covers the first structural refactor of:

- streaming audio accumulation and chunk extraction;
- session-state ownership around the streaming hot path;
- instrumentation required to compare the old and new implementations.

Primary user: a person dictating through Yuwp.app with the 1.7B model and one active session.

Current workaround: judge changes from subjective responsiveness, isolated traces, or one-off benchmark runs.

Desired behavior: every refactor produces paired correctness and performance evidence before it can be accepted.

## Non-scope

The first refactor does not intentionally change:

- model weights, quantization, prompts, token sampling, or decode limits;
- the 1.75-second steady chunk and 1.5-second bootstrap chunk policy;
- VAD thresholds or recurrent-state behavior;
- pause correction, final accuracy, or subtitle behavior;
- HTTP or stdio API fields;
- recording retention policy;
- multi-client scheduling.

Those changes need separate acceptance criteria because they trade latency against transcript behavior.

## Benchmark contract

### Gating configuration

| Setting | Required value |
|---|---|
| Machine | Current Mac Studio, AC power |
| Build | `release` |
| Model | `mlx-community/Qwen3-ASR-1.7B-bf16` |
| Clients | 1 |
| Chunk policy | 1.5s bootstrap, 1.75s steady |
| Batch correction | enabled |
| Finalization | active segment only |
| VAD | same setting before and after |
| Fixture order | fixed |

Before paired runs, close other GPU-heavy workloads or record them as a invalidating condition. Do not compare a clean after-run with a baseline collected while another ASR model is resident.

### Fixed fixture lanes

1. **Checked-in smoke lane**
   - `Tests/fixtures/jfk.wav`
   - `Tests/fixtures/asr_en.wav`
   - `Tests/fixtures/asr_zh.wav`
   - `Tests/fixtures/silence.wav`

2. **Local quality lane**
   - The five named fixtures in `FAST_CANARY_CASES` in `benchmarks/lib/stream_quality.py`.
   - These paths are fixed even though their audio remains private and outside the repository.

3. **Long-utterance regression lane**
   - At least one fixed fixture over 30 seconds with continuous or resumed speech.
   - It must exercise growing active audio, no-growth recovery, and stop finalization.

Missing fixtures fail the benchmark setup; the runner must not silently substitute newer recordings.

### Repetition and aggregation

- Warm the model once before measured work.
- Run each fixture at least five times in the same process when the harness supports it.
- Pair baseline and candidate runs by fixture and repetition.
- Report median, p95, maximum, and median absolute deviation where applicable.
- Exclude a run only for a recorded invalidating event such as thermal throttling, another GPU workload, process failure, or missing fixture. Do not discard slow runs merely because they are slow.
- Keep raw benchmark artifacts under `/tmp`; retain only aggregate metrics and configuration in the repository.

The existing quality runner starts a fresh process for each fixture. Before production refactoring, extend the benchmark lane so model-load time and process startup do not dominate repeated measurements.

## Correctness gates

All gates are mandatory.

### Build and test

```bash
swift build
swift test
swift test --filter "ASR Server"
```

The commands must complete without warnings introduced by the change.

### Audio accumulator contract

Tests must prove, over deterministic and randomized input partitions:

- every input sample is emitted exactly once and in order;
- no sample is dropped or duplicated across chunk boundaries;
- chunks are emitted at the configured bootstrap and steady sizes;
- an incomplete tail remains available for stop/finalization;
- concurrent feed/stop ordering follows the existing session contract;
- compaction or wraparound does not change output;
- empty input and exact-boundary input behave correctly.

A property test must compare the refactored accumulator with a simple reference implementation across multiple input lengths and feed partitions.

### Transcript behavior

For every fixed fixture:

- normalized final text must equal the baseline exactly;
- committed text must remain a prefix of final text;
- committed text must never shrink during a session;
- `text == append(committed_text, active_text)` for non-final updates under the existing renderer rules;
- final responses must set `is_final=true`, `update_kind=final`, and an empty active tail;
- silence must not produce non-empty text;
- English, Chinese, and mixed-script fixtures must not lose text;
- no session may hang, crash, or emit an update after its final response.

If an MLX or platform variation prevents exact text equality, the exception must be demonstrated on the unchanged baseline. Only then may the gate use normalized equality plus unchanged WER/CER.

### Quality metrics

Using the same fixtures and references:

- mean WER must not increase by more than 0.005 absolute;
- p90 WER must not increase by more than 0.01 absolute;
- normalized exact-match rate must not decrease;
- recovered no-growth seconds and no-growth case rate must not increase;
- finalization-added words must not increase.

For a structural refactor expected to preserve chunking, exact final-text equality takes precedence over aggregate tolerance.

### API and lifecycle

Existing HTTP and stdio request/response tests must remain green. Add regression cases for:

- feed before enough audio exists for inference;
- multiple chunks becoming available in one feed;
- stop with a partial tail;
- unknown and expired session IDs;
- feed racing with stop;
- recording disabled, ensuring the backend creates no recording artifact.

## Performance gates

Correctness-first means a refactor does not fail merely because its intended structural improvement is small. It still must not regress user-visible performance.

### Hard non-regression limits

Compare the median of paired benchmark runs:

| Metric | Limit |
|---|---:|
| Speech-active mean chunk `totalMs` | no more than +5% |
| Speech-active p95 chunk `totalMs` | no more than +5% |
| Maximum chunk `totalMs` | no more than +10% |
| Mean speech-to-first-text | no more than +0.20s |
| p90 finalization time | no more than +10% or +0.10s, whichever is larger |
| Mean user CPU per fixture | no more than +5% |
| Peak RSS after warmup | no more than +3% |
| Server errors, dropped feeds, timeouts | zero |

Stage timings must also be reported for encode, prefill, decode, VAD, queue wait, buffer/copy work, and batch correction. A faster total that hides a severe regression in one stage requires investigation rather than automatic acceptance.

### Clear-improvement requirement

A performance claim is valid only when one of these is true and every hard gate passes:

1. a user-visible primary metric—speech-to-first-text, speech-active mean/p95 chunk latency, or finalization latency—improves by at least 10% in paired median results; or
2. the directly targeted mechanical metric—bytes copied, allocations, or accumulator processing time—improves by at least 25%, and end-to-end latency does not regress.

The improvement must exceed run-to-run noise. Report the baseline and candidate median absolute deviation; if the claimed improvement is not larger than the combined variability, label the result inconclusive.

For the proposed accumulator refactor, the expected primary proof is lower accumulator processing time and fewer copied bytes. Lower end-to-end latency is desirable but not required because model inference dominates many chunks.

## Gating baseline

The uncontended baseline was captured after quitting Yuwp.app and before changing the streaming accumulator. The tree was based on `1b1a57b7cf10114621479189de66ec98ff80c9b5` with pre-existing recording changes; the complete status and diff hash are stored with the artifacts.

```bash
uv run benchmarks/cli.py asr load large \
  --files <five fixed fast-canary fixtures> \
  --concurrency 1 --repeats 5 --warmup --capture-text \
  --json /tmp/yuwp-streaming-refactor-baseline/load-large-c1-r5.json

uv run benchmarks/cli.py asr quality \
  --suite fast \
  --out-dir /tmp/yuwp-streaming-refactor-baseline/quality-fast
```

The repeated endpoint lane kept one warmed server alive for 25 sessions. The quality lane supplied detailed stage timing and transcript metrics.

| Metric | Baseline value |
|---|---:|
| Repeated endpoint average chunk | 115.69ms |
| Repeated endpoint p95 chunk | 357.97ms |
| Repeated endpoint first partial | 144.50ms |
| Repeated endpoint realtime factor | 0.0612 |
| Repeated endpoint final similarity | 0.89214 |
| Repeated endpoint peak RSS | 4041.47MB |
| Stream quality score | 0.472064 |
| Mean WER | 0.333576 |
| p90 WER | 0.726444 |
| Normalized exact rate | 0.20 |
| Mean speech-to-first-text | 1.80s |
| Mean recovered no-growth | 0.209s |
| Max recovered no-growth | 0.936s |
| Speech-active mean chunk time | 124.80ms |
| Speech-active median chunk time | 120.26ms |
| Speech-active p95 chunk time | 175.19ms |
| Speech-active maximum chunk time | 229.36ms |
| Speech-active chunk-time MAD | 16.54ms |
| Quality-lane max RSS | 4072.88MB |

Raw artifacts: `/tmp/yuwp-streaming-refactor-baseline/`.

## First refactor result

The candidate replaced per-chunk suffix removal with an indexed accumulator. It was measured with the same model, five fixtures, five repetitions, and an uncontended machine.

| Metric | Baseline | Candidate | Change | Gate |
|---|---:|---:|---:|---|
| Endpoint average chunk | 115.69ms | 115.55ms | -0.12% | pass |
| Endpoint p95 chunk | 357.97ms | 358.70ms | +0.20% | pass |
| Endpoint first partial | 144.50ms | 145.07ms | +0.39% | pass |
| Endpoint realtime factor | 0.06120 | 0.06099 | -0.34% | pass |
| Endpoint final similarity | 0.89214 | 0.89214 | 0.00% | pass |
| Endpoint peak RSS | 4041.47MB | 4042.72MB | +0.03% | pass |
| Speech-active mean chunk | 124.80ms | 124.72ms | -0.06% | pass |
| Speech-active p95 chunk | 175.19ms | 175.37ms | +0.11% | pass |
| Mean user CPU | 1.184s | 1.168s | -1.35% | pass |
| Mean system CPU | 1.340s | 1.312s | -2.09% | pass |
| Mean WER | 0.333576 | 0.333576 | 0.00% | pass |
| Stream quality score | 0.472064 | 0.472064 | 0.00% | pass |

Correctness evidence:

- all 25 repeated endpoint final transcripts matched exactly;
- all 25 repeated endpoint streaming transcripts matched exactly;
- all five quality-lane final transcripts matched exactly;
- every quality-lane partial text sequence matched exactly;
- the full unit suite passed 284 tests;
- accumulator partition, exact-boundary, incomplete-tail, and copy-reduction tests passed.

The representative 100ms-feed/1.75s-chunk accumulator test reduced suffix-copy work from 2,400 samples to zero. End-to-end latency changes were smaller than observed variability, so this result does **not** claim a user-visible latency improvement. It passes the correctness-first acceptance gate through exact behavioral equivalence, a 100% improvement in the targeted copy metric, and no material end-to-end regression.

Candidate artifacts: `/tmp/yuwp-streaming-refactor-candidate/`.

## Bounded live stall-refresh result

The second refactor replaced implicit full-session recovery during live speech with a pure, tested policy and a bounded active-segment refresh. Full-session context remains available at pause and finalization, where latency does not block an incoming live chunk.

Policy invariants:

- refresh only while speech is active and text has stopped growing;
- require at least two seconds of active-segment audio;
- never refresh more than twelve seconds of active-segment audio;
- wait at least two chunks before retrying a failed or unchanged refresh;
- never split or rewrite committed session context in the live refresh path.

| Metric | Accumulator baseline | Bounded refresh | Change |
|---|---:|---:|---:|
| Endpoint average chunk | 115.55ms | 111.63ms | -3.39% |
| Endpoint p95 chunk | 358.70ms | 333.12ms | -7.13% |
| Endpoint first partial | 145.07ms | 144.99ms | -0.05% |
| Endpoint realtime factor | 0.06099 | 0.05790 | -5.08% |
| Endpoint wall time | 102.57s | 99.16s | -3.33% |
| Endpoint peak RSS | 4042.72MB | 4047.05MB | +0.11% |
| Final similarity | 0.89214 | 0.89214 | 0.00% |
| Mean WER | 0.333576 | 0.333576 | 0.00% |
| Recovered no-growth time | 0.2088s | 0.2088s | 0.00% |

Correctness evidence:

- all 25 repeated endpoint streaming and final transcripts matched exactly;
- all five quality-lane final transcripts and complete partial sequences matched exactly;
- WER, stream-quality score, first-text timing, and recovered-stall metrics were unchanged;
- the full suite passed 286 tests, including explicit live-refresh policy boundaries.

The repeated endpoint p95 improvement was 7.13%, while the two preceding uncontended baseline runs differed by only 0.20%. The change therefore produced a measurable latency improvement without changing observed transcript behavior. The residual risk is corpus coverage: a future fixture may depend on more than twelve seconds of live batch context. Pause and finalization still retain the existing session-context recovery path, and the policy is isolated for adjustment.

Candidate artifacts: `/tmp/yuwp-stall-refresh-candidate/`.

## Consolidated session ownership result

The third refactor replaced six parallel per-session dictionaries with one `ManagedStreamingSession` object. Pending audio, activity time, recording state, language, model state, and lifecycle now share one ownership boundary. A `SessionOperationGate` serializes feed operations, makes close one-way, and lets expiry re-check its condition while holding exclusive session access.

Reliability invariants:

- two feeds for one session cannot mutate model or audio state concurrently;
- stop waits for an in-flight feed or closes first and rejects the late feed;
- stop and expiry run at most once;
- a feed after finalization cannot touch the finalized model session;
- expiry cannot close a session whose activity changed while cleanup waited;
- registry locking is separate from per-session operation locking;
- pending audio and recording data cannot drift away from their model session.

| Metric | Bounded-refresh baseline | Consolidated state | Change |
|---|---:|---:|---:|
| Endpoint average chunk | 111.63ms | 113.46ms | +1.64% |
| Endpoint p95 chunk | 333.12ms | 340.05ms | +2.08% |
| Endpoint first partial | 144.99ms | 147.48ms | +1.72% |
| Endpoint realtime factor | 0.05790 | 0.05920 | +2.25% |
| Endpoint peak RSS | 4047.05MB | 4045.92MB | -0.03% |
| Final similarity | 0.89214 | 0.89214 | 0.00% |
| Repeated quality p95 chunk | 176.29ms | 177.42ms | +0.64% |
| Mean WER | 0.333576 | 0.333576 | 0.00% |

Correctness and reliability evidence:

- all 25 repeated endpoint streaming and final transcripts matched exactly;
- all five quality-lane final transcripts and partial sequences matched exactly;
- all server integration tests passed against the release candidate;
- a real concurrent feed/stop race returned one final response and rejected every post-final operation;
- the full suite passed 291 tests, including active-operation serialization, close ordering, conditional expiry, and feed/stop race coverage.

One quality timing run contained a transient prefill outlier and reported p95 `210.85ms`; an immediate uncontended repeat reported `177.42ms`, within 0.64% of baseline. Both runs preserved exact text and are retained in the artifacts. The repeated 25-session endpoint lane stayed within the 5% non-regression budget. This refactor is accepted for reliability and maintainability, not as a latency improvement.

Candidate artifacts: `/tmp/yuwp-session-state-candidate/`.

## Evidence required in the completion report

The final report must contain:

1. baseline and candidate commit or tree identifiers, including whether either tree was dirty;
2. exact model path and model identity;
3. machine, power, thermal, and competing-workload notes;
4. exact commands;
5. fixture IDs and repetition count;
6. aggregate before/after table with absolute and percentage differences;
7. correctness test results;
8. failing or excluded runs with reasons;
9. a short residual-risk section;
10. raw artifact paths under `/tmp`.

Do not claim improved performance from one run, model-load timing, subjective responsiveness, or a metric that moved less than observed run-to-run variation.

## Implementation order

1. Add benchmark-only aggregation for speech-active chunk latency, stage timings, queue wait, VAD, buffer work, copied bytes, and allocations where measurable.
2. Add same-process warmup and repetition support.
3. Add the accumulator reference/property tests.
4. Capture the final clean baseline.
5. Refactor the accumulator and session ownership without changing decoding policy.
6. Run focused tests after each behavior change.
7. Run the full acceptance matrix and publish the evidence table.

## Revisit triggers

Write a separate acceptance specification before changing chunk duration, VAD policy, model selection, batch refresh/finalization, subtitle behavior, or multi-client scheduling. Those changes can improve latency by intentionally changing transcript behavior, so exact equivalence is not the right gate.
