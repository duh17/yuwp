# English ASR delivery result — 8 September 2026

**Neither candidate qualifies for production. Qwen remains unchanged and default.** Native Nemotron and repaired Parakeet both ran, but their development results were mixed: both made more word errors on the fixed clean-speech set, despite fewer errors on the small spontaneous-speech set. Parakeet's buffered previews revised substantially more text. No heldout confirmation or controlled latency gate was completed.

This is a completed **NON-ACCEPTANCE development experiment**, not a claim of English parity. No production backend/catalog/UI change, live service restart, installation, push or release occurred. The parent owns any local landing of reviewed benchmark tooling and this report.

## What ran

- Original frozen run: **81/81 trials**, 26.84 minutes, Qwen/Nemotron/Parakeet interleaved serially per case. All 27 Parakeet trials failed local readiness before audio; the original artifacts remain intact. Qwen's noise-only hallucination was also retained as a failure.
- Separately authorized resolver repair: eight model-free tests, independent parent review, then local-only readiness. Only the SDK directory basename changed; weights, precision, decoder, compute units and preview cadence did not.
- Separately frozen repaired Parakeet run: **27/27 trials**, 13.05 minutes, all transport/invariant checks passed. This was a later, **noninterleaved** pass, not a replacement for the original failures or a controlled latency comparison.
- Heldout: **zero inference trials**. Forty speech clips and six heldout diagnostics were materialized but never fed to a model. No N2/P2 tuning or third experiment followed.

Development strata: 12 LibriSpeech clips from six speakers (six short, six whole-utterance concatenations; 558 reference words), nine spontaneous AMI segments from one speaker (531 words), and six synthetic probes. Transformed repeats are not independent speech samples. The datasets, selection, hashes and attribution are frozen in [sources.json](sources.json), [materialization.json](materialization.json) and the manifests.

## Final word error rate

All 21 speech trials per measured system returned a final response. Empty speech output remains deletion-scored, not excluded. These are descriptive corpus WERs from the common unchanged normalizer; no confidence/parity inference is attached.

| Speech stratum | Qwen baseline | Nemotron N1 | Parakeet P1 repaired |
|---|---:|---:|---:|
| LibriSpeech dev, 558 words | **11 errors; 1.971%** | 16; 2.867% (**+0.896 pp**) | 15; 2.688% (**+0.717 pp**) |
| AMI spontaneous dev, 531 words | 61; 11.488% | 53; 9.981% (**−1.507 pp**) | 48; 9.040% (**−2.448 pp**) |

Nemotron returned an empty transcript for the one-word AMI reference **“Um”**: one deletion, not a lost long utterance. Do not turn the better AMI corpus WER into a claim of better English quality overall. LibriSpeech references also cannot adjudicate punctuation, case or number-formatting correctness.

## Observed client latency

Seconds, **p50 / p95**, from the real-time framed-stdio client. Packets were released at their 100 ms audio-end deadlines. Stop-to-final starts at scheduled audio end and includes queued feeds, transport and finalization—not just stop-request RTT. These are sample quantiles, not model-card chunk delays or batch RTF. First nonempty text can arrive only at finalization on a short clip; that metric alone does not prove a pre-stop preview.

| Stratum / system | First nonempty visible text | Stop-to-final |
|---|---:|---:|
| LibriSpeech / Qwen | 1.663 / 1.718 (12/12) | 0.606 / 1.154 (12/12) |
| LibriSpeech / Nemotron | 1.239 / 1.473 (12/12) | 0.041 / 0.055 (12/12) |
| LibriSpeech / Parakeet repaired | 1.108 / 1.153 (12/12) | 0.172 / 0.545 (12/12) |
| AMI / Qwen | 1.675 / 1.718 (9/9) | 0.921 / 1.510 (9/9) |
| AMI / Nemotron | 1.246 / 1.733 (**8/9**) | 0.044 / 0.061 (9/9) |
| AMI / Parakeet repaired | 1.098 / 1.114 (9/9) | 0.143 / 0.388 (9/9) |

On the common LibriSpeech pairs, Nemotron-minus-Qwen observed deltas were **−0.425 / −0.245 s** for first nonempty text and **−0.565 / −1.099 s** for stop-to-final. On AMI, the corresponding deltas were −0.432 / +0.013 s on eight first-text pairs, and −0.877 / −1.449 s on nine finalization pairs. These arithmetic differences do not establish candidate-caused speedups under uncontrolled load. **No latency-parity comparison is made for the later Parakeet pass.**

First nonempty text is distinct from the protocol's correct-prefix “useful” text. The latter requires the first three reference words (or the whole shorter reference) before stop; errors can make it undefined. Reported useful/stable quantiles therefore have different denominators and must not be read as interchangeable populations:

| Stratum / system | First useful, p50 / p95 (available) | Stable useful, p50 / p95 (available) |
|---|---:|---:|
| LibriSpeech / Qwen | 1.660 / 1.720 (11/12) | 1.660 / 1.720 (11/12) |
| LibriSpeech / Nemotron | 1.739 / 1.751 (10/12) | 1.739 / 1.751 (10/12) |
| LibriSpeech / Parakeet repaired | 2.084 / 2.118 (11/12) | 2.087 / 2.118 (10/12) |
| AMI / Qwen | 1.697 / 20.104 (7/9) | 1.697 / 20.104 (7/9) |
| AMI / Nemotron | 1.740 / 3.328 (5/9) | 1.740 / 3.328 (5/9) |
| AMI / Parakeet repaired | 2.105 / 3.770 (8/9) | 2.605 / 10.109 (6/9) |

On the ten common LibriSpeech useful-prefix pairs, Nemotron's observed p50/p95 deltas were +0.076/+0.029 s; missing endpoints remain blocking, not zero. Stable useful means a correct prefix that survives through final, not word-aligned acoustic lag. Startup/create costs are recorded separately; the runner's startup-inclusive metric contains its prescribed warm-up and is **not cold-start latency**. Actual AppKit display/injection and isolated final-pass execution time were not measured.

## Guardrails and failures

- **Qwen noise hallucination:** the 5 s synthetic noise-only probe finalized as **“I'm sorry.”** (two normalized words). This is retained as a failed diagnostic; it is not hidden by success-only summaries. All three systems returned no words for pure silence. Nemotron and repaired Parakeet returned no words for noise-only.
- Three reference-bearing synthetic transforms contributed 255 repeated reference words: Qwen 0 errors, Nemotron 1, Parakeet 4. These are diagnostics, not another independent speech-quality sample. Cutoff has no lexical gold and is not scored as silence or against the uncut transcript.
- Normalized active-text rollback words/reference word, LibriSpeech then AMI: Qwen **0.025 / 0.620**; Nemotron **0.073 / 0.019**; Parakeet **0.525 / 2.032**. This counts removed word suffixes between consecutive drafts, including unfinished-word revisions. It is not changed committed text. Parakeet's preview instability does not meet the no-increased-rollback guardrail in this dev sample.
- Last-five-word suffix edit counts, LibriSpeech then AMI: Qwen **2 / 5**; Nemotron **3 / 7**; Parakeet **2 / 2**. Nemotron's observed tail errors increased in both speech strata.
- No changed committed-prefix, reconstruction or final-consistency violation was recorded in the completed measured streams. Both prototypes keep all text revisable until final; this is **not proof of production mid-stream commit/injection support**. Both enforce a 45 s input cap.
- Original Parakeet failure was an adapter integration defect, not bad recognition: HF basename `parakeet-tdt-0.6b-v2-coreml` versus SDK local basename `parakeet-tdt-0.6b-v2`. The reviewed repair uses `Repo.parakeetV2.folderName` and a separate alias to identical assets. Offline readiness then succeeded in 16.358 s. All original failed trials remain recorded alongside the later successful pass.

## Why neither qualifies

The zero-margin heldout WER and latency confidence gates were **not evaluated**. There are no affirmative parity confidence bounds to report. A 244 s clearance attempt had **zero qualifying samples**: idle fraction 5.2–77.7% (required at least 85%) and load1 25.8–46.8 on 32 CPUs (required at most 16). Accelerator/external-workload exclusion was unavailable. Continuous telemetry was retained; unrelated work was not stopped.

The frozen heldout set also lacks adequate spontaneous and human punctuation/case/number-formatting gold. The original confirmation schedule exceeded its four-hour budget even before inference overhead. Parent and independent review therefore blocked that schedule **before candidate output**, narrowed execution to development diagnostics, and left heldout untouched. No margin was widened and no hard fixture was removed.

Thus **Nemotron: not qualified; mixed development quality with clean-WER/tail regressions. Parakeet: not qualified; mixed development quality with clean-WER and preview-rollback regressions.** Neither is promoted on promising finalization timings. Confidence/power, representative heldout coverage, controlled load, uncapped session behavior and real dictation QA remain unresolved.

## Exact baseline, models and build

- **Qwen3-ASR-1.7B-bf16**, HF `mlx-community/Qwen3-ASR-1.7B-bf16` revision `e1f6c266914abc5a46e8756e02580f834a6cf8a7`. Same transcriber passed as `--model` and `--batch-model`; final accuracy **ON**, English, 16 kHz, 1.75 s chunks, automatic final policy, no aligner/context hints. All 27 stdio readiness replies confirmed model ID, final accuracy, sample rate and chunk size.
- Live VAD was **configured ON** by unchanged CLI defaults. Separate isolated HTTP readiness reported live and batch VAD true. Per-trial stdio VAD load state is not independently observable: stdio info omits it, and the frozen commands did not enable diagnostic logging. StdIO has no separate batch-VAD object by design. This fidelity limitation is explicit, not a claimed stderr proof.
- **Nemotron N1:** FluidAudio `5c19d5e12320e22bbfb7a1877b089d2665a69add`; converted HF revision `e673531caa6d25ab7baf5a8c14c9b99ba1551838`, explicit **560 ms**, int8 encoder, required fused decoder, CPU+ANE configuration. Fresh throwing manager initialization per stream; cumulative getter, not the empty `process()` return.
- **Parakeet P1:** same SDK; `FluidInference/parakeet-tdt-0.6b-v2-coreml` revision `ee09c569f73759e6d44c9bd16766f477b2b36d39`. Selected encoder metadata identifies **6-bit palettization**, not blanket FP16/int8. Preprocessor/decoder/joint CPU-only, encoder CPU+ANE; internal chunk concurrency one; full-prefix previews every 1 s, 45 s cap, final full-prefix pass. Compute-unit configuration is not measured hardware residency. Upstream-to-conversion equivalence remains unproved.
- Baseline source `45cfcbcb8089f02aa3c2cb4c2a2709c32e57c022`; Apple Swift 6.3.3, release builds, macOS 27.0 (26A5416b), Mac15,14/M3 Ultra, 32 CPUs, 512 GiB. All nine Qwen and 43 selected candidate asset files were hashed; repaired P1's 21 files were reverified unchanged.
- Fresh Qwen Swift/C++ executable SHA-256 `b4ea2d713d8b7babe24da740b8ab9db73cf9c40af9c66592ad22fd6e5b133bd1`. Metal compilation failed because the toolchain was absent; only the parent-authorized **source-matched shader cache** was reused. Shader SHA-256 `67412110099272b343e14ccec0b42ad495f54ce33f56d4a999e7dbe36f6134ef`; no system component was installed.
- Original adapter SHA-256 `ee21934dedf0418c778a6ab95e841823ab7a132bce95442be4b0f6e611c5f434`; repaired adapter `00136ef255d812f0bf3263a2e0eba2a7170e17f70c8f4190c795fcff4bb0aedd`. Full source, build, command and asset hashes are in the receipts.

## Evidence, reproduction and review

- [Compact per-trial transcripts/metrics, including every failure](receipts/case-results.jsonl)
- [Stratified descriptive summaries](receipts/combined-dev-summary.json) and [run counts/raw hashes](receipts/runs.json)
- [Baseline/config/build receipt](receipts/pre-run-receipt.json), [HTTP readiness](receipts/qwen-http-readiness.json), [stdio fidelity caveat](receipts/qwen-stdio-fidelity.json)
- [Repaired P1 readiness](receipts/parakeet-repaired-readiness.json) and [separate run receipt](receipts/parakeet-retry-receipt.json)
- [Protocol review](receipts/protocol-review.md), [exact-source pre-run approval](receipts/pre-run-review.md), [scope lock](dev-only-addendum.md), [readiness exception](parakeet-readiness-repair.md), [P1-only scope](parakeet-retry-scope.md)
- [Durable source and staging recipe](reproduction/README.md): exact materializer, original adapter/tests, resolver patch, pinned acquisition logic and analysis scripts. No large audio/model/build blobs are tracked.

Raw framed-event runs, per-trial stderr and continuous load telemetry were also copied out of `/tmp` into this worktree's ignored `.pi/research/english-parity-20260908/artifacts/`. Their hashes are in `runs.json`; preserve that directory when archiving the worktree. The compact committed case evidence permits independent WER/count/quantile checks without temporary raw files.

Validation: fresh release Qwen and Fluid builds; isolated HTTP readiness and actual stdio replay; **54 benchmark unit tests** (the earlier 37-test focused subset includes the explicit under-load execution regression; three later model-free tests cover reproduction staging safety); original adapter seven and repaired adapter eight model-free Swift tests; source-identity and resolver-patch reconstruction checks. Upstream Fluid warnings were recorded, not called a warning-free dependency build. Full app/Swift suite and hotkey tests were not run because production app/server source was unchanged.

Delegated children were all `openai-codex/gpt-6-astra`, medium, and are stopped: protocol preparation `92c6d7c1-3a3d-4903-b6f2-fd10c4639cb6`; protocol review `3fbac477-663a-4ba1-8078-e6fba65ab23a`; native prototype `6e85e15b-0d13-4ee7-83b1-908f1378b0a0`; evaluator `69acfc45-a3da-4c96-bd00-235776212b89`; exact pre-run review `0d729ef7-5305-4fd1-82fd-0e880b68efab`; final source/report/reproduction review `291cd60e-7a53-4be4-b5c1-64f27cbf1f10`. The final review independently reproduced all compact/raw metrics and found one staging-shell safety issue; fail-fast guards and mocked sentinel/checksum tests corrected it. The independent parent approved the exact resolver repair and thin supplemental driver before their authorized runtime steps. No local chat/orchestration model was launched.

**Human leftover:** hotkey → real microphone → text injection in Chen's apps, qualitative formatting/preview preference, real noise/accents/cutoff gold, and lower-tier Mac behavior. None is represented as Chen-tested. All task-owned inference processes were closed; live Yuwp services were left running unchanged. This experiment ends here.
