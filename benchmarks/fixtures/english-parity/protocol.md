# English ASR paired dictation protocol v1 — preparation freeze candidate

## Decision and scope

This protocol is for the delivery owner and independent reviewer. It freezes the public fixtures and proposes the complete measurement rules **before any inference results**. Owner approval of this document and the pre-run configuration receipt is required before candidate measurements. Preparation does not authorize inference, integration, installation, or service changes.

**Current coverage verdict: general English dictation parity is blocked.** The materialized data can establish only a bounded read-speech comparison and diagnostic findings. A numerical win cannot override the coverage gaps below. Missing evidence is inconclusive, not a pass. Qwen stays the default.

Compare isolated release builds of current Qwen3-ASR-1.7B-bf16 with final accuracy ON, Nemotron English at 560 ms, and Parakeet TDT **v2 English** with bounded visible previews. No live service, private recordings, old inference outputs, candidate weight downloads by this preparation task, source implementation, or delegation is part of preparation. Batch RTF is supplemental, not a dictation-parity substitute.

## Frozen inputs and lineage

`dev.jsonl`, `heldout.jsonl`, and `diagnostics.jsonl` use the existing `asr_evaluate.load_manifest` keys: `id`, `audio`, `language: en`, `metric: wer`, and `reference`. Additional fields record source components, speaker clusters, sample counts, transformations, and SHA-256. Audio is local-only under `/tmp/yuwp-english-parity/data/audio`; no tracked audio. `materialization.json` records archive, manifest, materializer, and normalizer hashes. `sources.json` records provenance and exclusions. `model-sources.json` pins the selected artifact file metadata, including available LFS SHA-256, size, and Git blob identifiers.

| System | Pinned source/artifact | Required behavior |
| --- | --- | --- |
| Qwen baseline | `mlx-community/Qwen3-ASR-1.7B-bf16` at `e1f6c266914abc5a46e8756e02580f834a6cf8a7` | BF16, English, final accuracy ON, chunk 1.75 s, live VAD ON, automatic final batch chunking; no aligner/context prompt/hotwords |
| FluidAudio | `5c19d5e12320e22bbfb7a1877b089d2665a69add` | Offline local-only load, serial ordered session operations; verified errors are not empty success |
| Nemotron | `FluidInference/nemotron-speech-streaming-en-0.6b-coreml` at `e673531caa6d25ab7baf5a8c14c9b99ba1551838`, `nemotron_coreml_560ms/` | Explicit 560 ms; no inherited 2240 ms default; converted encoder is int8 according to card |
| Parakeet | `FluidInference/parakeet-tdt-0.6b-v2-coreml` at `ee09c569f73759e6d44c9bd16766f477b2b36d39` | `Preprocessor`, `Encoder`, `Decoder`, `JointDecision` compiled models plus `parakeet_vocab.json`; not v3 |

Parakeet's selected `Encoder.mlmodelc/metadata.json` describes **enc6bit-palettize quantized - encoder**. Do not call this artifact uniformly FP16 or int8. Nemotron and Parakeet cards identify NVIDIA upstream model names but do not prove exact upstream checkpoint revisions or conversion equivalence. The converted artifact revision is the executable input pin; missing precision/lineage verification must remain visible in the final report. Do not choose alternate similarly named model files without a new pre-results receipt.

The baseline uses the app's stdio path. Source inspection shows this path has no separate batch-VAD object: record batch VAD as `nil / not applicable to stdio`, not falsely as ON. This is different from the current HTTP info configuration. Preserve the applicable current final-pass behavior and record the actual transcriber constructor and CLI arguments. A baseline that quietly changes model size, final accuracy, live VAD, or final chunk policy invalidates the comparison.

Before any model run, archive: Yuwp source SHA and dirty diff hash, adapter source hash, Swift/compiler/macOS/SDK versions, release binary and metallib hashes, Package.resolved hashes, exact flags, selected model-file SHA-256 inventories, Core ML compute-unit settings per component, readiness response, and protocol/manifest hashes. Verify local-only readiness without altering the live app/preferences or service. Model acquisition/load/compatibility failure cannot pass.

## Data and coverage

| Split | Speech clips | Independent speaker clusters | Audio | Shape |
| --- | ---: | ---: | ---: | --- |
| Development | 12 | 6 | 203.050 s | 6 short (3–10 s), 6 long (20–40 s) |
| Heldout | 40 | 20 | 642.5951875 s | 20 short, 20 long |
| Supplemental diagnostics | 21 | Reuses parent speakers; not 21 new independent samples | See manifest | 15 dev, 6 heldout |

Source: [LibriSpeech SLR12](https://www.openslr.org/12), public read English audiobooks, CC BY 4.0. Archive SHA-256 is `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23`; publisher MD5 was verified. Human-authored book references were corpus-segmented/aligned; they are not new human dictation transcriptions.

Selection is result-blind: sort test-clean speakers by SHA-256 of UTF-8 `english-parity-v1:` plus speaker ID; first six are dev, next twenty are heldout. Use lexical source-file order. Per speaker choose the first 3–10 s utterance and the first same-chapter concatenation of whole other utterances reaching 20–40 s, with exactly 250 ms of zero PCM between components. Do not truncate words to make the duration target. Source component IDs, sample positions, and hashes are in each row. No component or speaker crosses splits. No model outputs were used to select samples. Long cases are explicitly concatenated utterances, not natural continuous long dictation.

Diagnostics per split use its first long case: 1 s leading plus 2 s trailing silence; 3 s pause inserted at the second component's start; additive deterministic uniform noise at 20 dB whole-clip RMS SNR; last 320 ms removed; 5 s silence-only; and 5 s noise-only. LCG32 recurrence, seeds, scale, clipping count, sample positions, and resulting hashes are recorded in the materializer/manifest. The cutoff reference is deliberately empty and marked **lexical gold unavailable**, not a silence reference: it must not enter WER or silence-hallucination scoring. This probe checks finish/error/stability behavior only until a human verifies audible cutoff words. Synthetic noise is not evidence about real microphone noise.

Nine nonempty human-segment cuts from the existing public AMI EN2002b speaker D excerpt are **dev-only** supplemental probes. Exact sample cuts preserve the existing human segment boundaries and all reference hesitation/repetition text; the punctuation-only final segment is excluded. AMI is CC BY 4.0; source audio/reference hashes and original URLs are recorded. These probes cover spontaneous hesitation, repetition and UI/software vocabulary, but only one previously public close-talk speaker. They do not close the heldout spontaneous-speech gap. Forced word alignment is not word-timing gold.

Coverage blockers that prevent general parity even if all numerical bounds pass:

- No verified accent strata; speaker count or apparent names do not prove accent diversity.
- No heldout spontaneous dictation/self-correction reference set. AMI is dev-only and single-speaker.
- LibriSpeech references are uppercase and generally unpunctuated. They cannot establish punctuation, case or numeric-formatting correctness. Names/numbers in book text are not a validated dictation entity/formatting challenge set.
- Noise, pause and cutoff transformations are synthetic. Cutoff has no lexical gold; actual endpoint and environmental-noise coverage is missing.
- Read-speech benchmark contamination is possible; model cards already report LibriSpeech evaluation. Neither training disjointness nor real-user generalization is established.
- Twenty heldout clusters can have weak power at zero margins, especially for p95. Do not widen margins or add heldout trials after viewing outputs.

Existing Earnings-21 references have empty `ts`/`endTs` fields in the inspected source; safe short cuts cannot be derived without new alignment. Old earnings results were not used for selection. Public FLEURS and MINDS14 cards were inspected only: FLEURS is read speech; MINDS14 offers 8 kHz English AU/GB/US banking-intent queries but requires a new verified split/materialization. Neither is silently counted as coverage. Bounded preparation stops here. Any coverage expansion needs a separately approved, pre-results freeze; these files must not be adjusted to rescue a result.

## Common transport, audio clock and endpoint

1. Use existing `Sources/ASRIPC/Protocol.swift` binary stdio framing for **all** systems: two UInt32 **big-endian** lengths (JSON bytes, binary bytes), JSON metadata, then PCM. PCM is signed 16-bit **little-endian**, mono, 16,000 Hz. No JSON-lines substitution. Commands are `info`, `create`, `feed`, `stop`. Request IDs must match replies. `create` must return `session_id`; associate feed/stop with that client session and validate an echoed session ID only when present (Qwen omits it on feed/stop). UTF-8 JSON keys and transcript fields follow the existing protocol.
2. Decode/resample fixtures once during preparation. Feed exactly their frozen PCM, without per-model gain, denoising, VAD clipping, resampling, leading/trailing silence, or frontend packet changes. Adapter-internal model preprocessing is part of that system and must be recorded. No extra common 1.5/1.75 s buffer ahead of native tiers.
3. After `create` succeeds, choose monotonic `t0`. Packet k contains at most 1,600 samples (100 ms). Release it at `t0 + cumulative_samples/16000`, the packet **end**, including the final shorter packet. There is at most one in-flight feed. Wait for its full reply before sending the next eligible packet. Never shift `t0` to hide a slow backend, drop samples, or parallelize a single stream. Drain late packets in order without artificial catch-up sleeps; preserve all original deadlines.
4. Define user stop time `t_stop = t0 + total_samples/16000`. After the final feed reply, send `stop` immediately. No extra trailing padding or sleep. Stop-to-final is measured from **t_stop**, so queued feeds, backpressure and transport delay count. Also report stop-request-to-final separately; it is not the primary stop latency.
5. Visible text is the client-decoded reply's complete `text` (normalize hypotheses with `strip_annotations=False`), with `committed_text`, `active_text`, `update_kind`, `batch_corrected` and `is_final` recorded. This is the client visibility boundary, not SDK token time; actual AppKit display and text injection remain human/product QA. Do not claim callbacks invisible to the client as first text. Nemotron's empty `process` return is not a valid substitute for verified cumulative callbacks/getter output.
6. Record scheduled deadline, write start/end, reply completion, full raw response, text fields, sample count, pending audio seconds, stop intent/request/reply, process exit and errors per event. Clock is one client monotonic clock for every system; record its resolution. Maximum request timeout is 30 s; readiness timeout 180 s; trial timeout is audio duration + 60 s after successful create. A timeout/error marks the case failed, never an empty successful transcript.
7. Backend-induced lateness is a measured outcome, not grounds to remove a slow trial. Record max/p50/p95 packet lateness and end backlog. Client wake lateness measured before acquiring the feed-operation slot must also be logged; external scheduler stalls are load-control violations, not unexplained filtering.

Committed text is immutable across feed, EOU, stop and final correction. EOU is not session end. Each stream starts with clean state; if reset swallows errors, recreate the manager rather than assume a clean state. Do not share one Fluid manager across streams or hold locks across await. A missing required final/text field must be explicitly diagnosed, not synthesized as success. Optional `batch_corrected` omission means false, as in current Qwen.

## Bounded development configurations

The owner accepted the P1/P2 cadence and 45 s cap during preparation. Exact implementation settings and compute units still require a pre-run receipt and independent protocol review. These two-slot limits do not permit heldout tuning:

- **N1:** Nemotron 560 ms at pinned SDK defaults with explicit tier and recorded compute units. **N2:** optional same-560-ms fused-decoder enabled/disabled comparison, only if the selected pinned assets include the fused decoder; declare the exact toggle before N2. The owner prefers no second configuration unless dev evidence justifies it. No 1120 ms or other tier change in this protocol.
- **P1:** Parakeet v2 full-received-prefix previews first at 1.0 s, then every additional 1.0 s, hard 45 s input cap, one final full-input pass. **P2:** only change preview cadence to 0.5 s. No external punctuation model or Qwen cleanup. Prefix work and final pass remain measured. A full-input cap is bounded buffered preview, not unlimited streaming support.
- If the existing candidate design needs different preview windows, overlap or commit rules, the owner must replace this proposal **before the first development output**, with an exact bounded rule and second-slot change. No implicit SDK buffering.
- Development uses only the 12 dev clips and 15 dev diagnostics, one warm pass per attempted configuration. Declare the selected slot per candidate before revealing any heldout outputs. At most two candidate configurations, no optimize-until-green loop. Adapter bug repairs after heldout exposure cannot reuse this heldout set to claim confirmation.

The 45 s proposal accommodates every current fixture, including transforms, but is itself a scope limitation for general dictation. No-preview/final-only behavior cannot pass first-visible parity. Active text can revise; immutable commits must never revise. If the adapter offers no useful pre-stop text on a required speech clip, first-visible latency is unavailable for that clip and the full gate cannot pass.

## Serial order, warm/cold and machine controls

Use this same physical mac-studio, AC power, no sleep or low-power mode. Record hardware/chip, CPU count, memory, OS and thermal state. Do not stop personal inference, other agents, simulators or applications. No concurrent candidate or baseline inference; no benchmark against the live 7936 service. Downloads/builds must finish before measurement.

Before every system trial, record 30 s of 1 Hz load/CPU/memory/thermal telemetry plus process activity. Require normalized 1-minute load average <=0.5 per logical CPU, aggregate idle >=85% throughout the window, nominal thermal state, no swapout increase, and no known external inference/build/simulator workload active. Record GPU/ANE activity if supported; inability to exclude concurrent accelerator inference is **no load control**, not a pass. `uptime`, `top`, `vm_stat`, `pmset -g therm`, and read-only process inspection are practical evidence sources; document unavailable counters. Telemetry remains on throughout the paired block. A single preparation snapshot is not clearance: observed load1 was 73.41 with active unrelated simulators/diagnosticd.

If clearance fails, defer; never kill other jobs. Wait at most 20 minutes for an acceptable window, then return `blocked-load-control`. If an external load/thermal violation starts mid-block, stop measurement and retain the affected complete block as invalid evidence. Do not silently rerun or filter a slower system. Heldout confirmation is one scheduled execution; a load-interrupted execution is incomplete/inconclusive and needs a separately reviewed new experiment rather than result-driven retries.

Use fixed order based on ascending SHA-256 of `english-parity-order-v1:<phase>:<repeat>:<clip-id>`. For each clip, select one of the six permutations of [Qwen, Nemotron, Parakeet] using the first eight bytes of SHA-256 of `english-parity-system-v1:<phase>:<repeat>:<clip-id>` interpreted big-endian, modulo six; enumerate permutations lexically. Run each three-system block serially. Complete all systems for a clip before the next. No candidate gets a different fixture order. If a candidate cannot load, record its missing trials; do not redefine success around the survivors.

- **Warm:** two heldout repetitions per system, grouped by clip. For each system trial use a fresh isolated process, wait for readiness, then one unscored warm-up of the fixed first dev short clip; create a new clean session for the measured clip. This standardizes process/model warm-up without keeping competing models resident. Run all warm blocks before cold blocks. Include six heldout diagnostic clips once per system after warm speech trials.
- **Process-cold:** one heldout repetition of all 40 clips per system. Fresh process, no unscored model inference; measure spawn-to-ready, create latency, then real-time feed metrics. OS file caches and Core ML compilation caches are **not** purged. This is process-cold, not first-install/device-cold. Report load time separately and startup-inclusive first-use latency as spawn-to-first-useful-visible. No cold/warm pooling.
- This is 240 warm speech trials, 120 cold speech trials, and 18 heldout diagnostic trials, plus prescribed warm-ups. Total run budget is four wall-clock hours excluding authorized preparation/build/model acquisition. Budget exhaustion means incomplete, not a partial-data pass. Report actual counts and wall time.

## Metrics and paired confidence analysis

Preserve raw transcripts and events. Use `benchmarks/lib/transcript_metrics.py` at SHA-256 `9c8ce3b3efffb4091874a6079d554cde0ed3c4a6785416ead9a89fcefc56f1e0` for primary WER: NFKC/casefold, common punctuation handling, reference-only annotation stripping, and deterministic C/S/D/I. Hypothesis annotations remain scoreable. Do not add number normalization, filler deletion, entity fixes or candidate-specific cleanup. Report raw strings beside normalized text so WER does not conceal formatting defects.

For each speech trial define:

- **Final WER:** final transcript corpus micro-WER (sum errors / sum reference words), plus per-case C/S/D/I and macro-WER. Warm quality uses mean error counts across its two repetitions; also report each repeat and all nondeterministic differences.
- **First useful visible:** earliest pre-stop client reply whose normalized visible text begins with the first three reference words, or the whole reference when shorter. Latency is reply time minus t0. This excludes empty strings and arbitrary wrong first tokens. Also report first nonempty text descriptively; it is not the gate metric.
- **Stable useful text:** earliest occurrence of that same correct prefix after which those visible prefix words never change through final. Latency is reply time minus t0. Report first nonempty immutable commit separately. This is stable-prefix latency, **not** per-word acoustic-alignment lag; the fixtures lack verified word-time gold.
- **Stop-to-final:** first successful final reply completion minus t_stop. A final reply must contain the real finalized text and `is_final=true`. Include all pending input and final-pass work. Report stop-request-to-final and final-pass runtime separately.
- **Warm per-case latency:** median of the two repetitions, then compute corpus p50 and p95 over the 40 per-case values. Cold uses each case's one observation. Also report raw-trial p50/p95 to expose repeat variability. Quantiles use linear interpolation at `(n-1)*p` on sorted values.

Paired confidence plan, fixed before outputs:

1. Bootstrap **20 speaker clusters**, not 40 independent clips. Draw 20 cluster IDs with replacement; keep both short and long cases and all repetitions of each chosen speaker. Use identical sampled indices for candidate and Qwen. Diagnostics never inflate the bootstrap sample size.
2. Use 20,000 draws with deterministic Python `random.Random(20260908)` and `randrange(20)` per cluster position. Record Python version and draw-index artifact hash. Compute WER candidate-minus-Qwen and candidate/Qwen ratios of p50 and p95 for first-useful, stable-useful and stop-to-final, separately for warm and cold. Cold additionally compares p50/p95 startup-inclusive first-useful latency. Reference-weighted WER must be recomputed inside every draw, not averaged from utterance WER.
3. Report paired point estimates, two-sided descriptive 90% intervals and the one-sided 95% upper percentile bound. The one-sided upper bound uses the same linear 0.95 quantile rule. Strict evidence requires **WER-difference upper bound <=0**, and **every key latency-ratio upper bound <=1**, separately in warm and cold. Exact equality can pass a numerical endpoint; failure to reject regression cannot. No epsilon margin or post-hoc rounding to pass.
4. The stated bounds are per candidate, all required endpoints jointly satisfied (intersection rule), not a familywise claim about either candidate selected after results. Report both candidates and all endpoints; do not select the best endpoint/phase. No tuning or configuration choice after heldout is opened.
5. A ratio with zero/undefined baseline, missing useful text, zero-word speech reference, missing trial, malformed output, timeout or failure is **undefined/blocking**. Never substitute 1, drop the pair, use a timeout as an exact observation, or exploit empty output as low WER. Valid successful empty speech output has deletion WER, but lacks useful-text latency and cannot pass. Silence diagnostics legitimately have no WER/first-text endpoint; score their hallucination count instead.
6. If the point estimate regresses, label that endpoint fail; if the point estimate meets the margin but the upper bound does not, label it inconclusive. Degenerate valid identical paired observations can yield a zero-width interval; this is only evidence for the sampled corpus, never resolution of coverage gaps. No additional heldout repetitions to reduce uncertainty.

## Guardrails and verdict

All guardrails are conjunctive with numerical bounds. Any unavailable required gold/evidence blocks promotion.

- **Formatting:** report whitespace-collapsed, otherwise raw character edit rate; punctuation-only sequence edit counts; case mismatch counts on casefold-equal aligned words; and literal numeric-span correctness against a preapproved human formatting reference. No model-specific inverse text normalization. Candidate errors must not exceed baseline, both overall and per audited formatting case. Current LibriSpeech gold cannot support these judgments, so this guardrail is **unresolved**, not zero errors. AMI dev punctuation is diagnostic, not heldout formatting gold.
- **Tail:** report normalized last-five-word sequence edit counts between reference and hypothesis suffixes (or full shorter sequences). Candidate aggregate tail errors and per-clip missing suffix words must not exceed Qwen. Flag excess repetitions of the reference's last-five-word phrase relative to its occurrence count in the reference. A committed suffix lost during finalization, dropped feed, or duplicated transport segment is an absolute failure. Cutoff lexical loss is unscorable until separately human-verified; do not apply full-reference tail scores to truncated audio.
- **Silence/noise-only:** zero normalized words at every visible event and final; any candidate hallucination is a failure even if baseline also hallucinates. Do not treat a nonresponse as correct silence.
- **Stability:** zero changed/removed committed characters, correct reconstruction using the existing nonempty-parts space-joining rule, and no commit/final disagreement. Record active-text rollback as the sum of removed suffix words relative to consecutive normalized-text longest common prefixes, divided by final reference words. Candidate aggregate rollback rate must be <=Qwen; report worst case. These event/commit invariants are absolute even when WER is better.
- **Operational:** no load-control waiver, missing output, hidden fallback, unsupported-language lie, unverified model inventory, unbounded preview, or error-to-empty conversion can pass. Report cancel/reset/offline-readiness proof separately; preparation has not executed it.

Verdicts are `pass-bounded-corpus`, `fail`, `inconclusive`, or `blocked`. **No general-parity pass is possible with the current coverage blockers.** Passing a batch-only or capped-preview comparison does not authorize production integration. Actual hotkey → microphone → text injection and Chen's qualitative preference remain human leftovers.

## Owner freeze checklist and preparation receipt

Before inference the owner and reviewer must approve: exact adapter preview/commit bounds and both allowed config slots; release/config/compute-unit hashes; baseline stdio VAD semantics; load-control instrumentation availability; exact event implementation of useful/stable text; and the limited claim permitted by current data. Record approval with this protocol hash and all manifest/model-source hashes. No candidate output has been generated by preparation.

Reproduction: download the pinned public test-clean archive to `data/source`, verify its recorded SHA-256 and publisher MD5, extract beneath `data/source/extracted`, then run `uv run --python 3.14 --no-project /tmp/yuwp-english-parity/data/materialize.py` from the worktree root. The local materializer hash and tool versions are in `materialization.json`; all generated WAV/reference/component hashes are in the manifests. Reproduction needs the existing tracked AMI fixture. Metadata inspection files are retained under `data/source`; no candidate weights were fetched. Paths and hashes, not a future reconstruction from mutable HF `main`, are the freeze inputs.
