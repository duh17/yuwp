## Verdict: REVISE; diagnostic addendum approved in principle

The proposed fallback is statistically honest **only as non-acceptance diagnostics**. It cannot waive load control for acceptance, establish quality parity, or authorize production integration.

### Required finite corrections

1. **Freeze the fallback before outputs.** After one unsuccessful ≤20-minute clearance attempt: only Qwen + N1 + P1, each on 12 dev clips and **15 dev-filtered diagnostics**—81 scored trials maximum. No N2/P2, cold phase, retries, heldout speech, or six heldout diagnostics. Preserve the four-hour total cap, fixed serial ordering and prescribed warm-up; count startup/warm-up overhead. Finish incomplete if exhausted.

2. **Make the waiver narrow.** Record load throughout; label every fallback artifact `NON-ACCEPTANCE`, with acceptance status `blocked-load-control`. Report descriptive quality/runtime observations, not inferential parity or candidate-caused latency differences. Do not later relabel quieter trials as acceptance evidence. Existing non-interference and operational safeguards remain applicable.

3. **Correct transport assertions to match existing Qwen.**
   - `create` returns `session_id`; current feed/stop replies omit it. Validate request IDs and client session association; validate an echoed session ID when present.
   - Reconstruction is the existing space-joining rule, not literal `committed_text + active_text`.
   - `batch_corrected` can be omitted when false. Distinguish optional omission from missing required final/text evidence.
   
   Otherwise the harness could falsely reject the faithful baseline.

4. **Complete the pre-run receipt.** Exact adapter/commit rules, resolved N1 decoder toggle, P1 scheduling, compute units, release/assets/config hashes and actual Qwen VAD readiness remain prerequisites—not approved implementation evidence today. Explicitly normalize visible hypotheses with `strip_annotations=False`. Exclude cutoff/silence/noise-only from WER despite their manifest `metric: wer`; retain their dedicated diagnostics.

5. **Repair the controlled-confirmation budget before authorizing it.** The current plan cannot fit four hours: 378 × 30-second clearance windows plus speech playback alone total approximately **4.76 hours**, before warm-ups, diagnostic audio, startup or inference overhead. Keep controlled confirmation blocked pending a separately frozen feasible budget/schedule. This does not block the smaller diagnostic fallback.

### Nonblockers / strengths

- Verified 12/40 speech counts, 6/20 speaker clusters, duration/reference-hash consistency, and no cross-split speaker/component overlap.
- Speaker-cluster paired bootstrap, zero margins, conjunctive endpoints and undefined-pair blocking are defensible; 20-cluster p95 uncertainty remains substantial.
- Stop latency correctly begins at scheduled audio end and includes queued feeds/backpressure. Existing `benchmark_asr_transport.py` does **not** implement that clock—it measures stop-request RTT and feeds without real-time pacing.
- Coverage, formatting, cutoff-gold and artifact-lineage limitations are candid. Numerical wins cannot remove them.

### Exact approval scope

**Conditional methodological approval for the dev-only fallback after corrections 1–4 are recorded and the pre-run receipt is approved.** No approval for heldout execution, acceptance claims, adapter correctness, production support or landing. The proposed addendum is currently message text, not a reviewed hashed file.

### Reviewed SHA-256

```text
delivery-contract.md
7c8c80af3e79970cce8bad1e6ffa218b3e7e5c2af444907b87190c613959e09d
protocol.md
d50da0bd95da15bd49e08e19b99b6ac83460e3214da4d6aacf5169714302d473
dev.jsonl
3ca6c291ced53baa402e14fd81c8466431384f66d45ad4b60de0859e94f80cb8
heldout.jsonl
e25a0662d1b9f31b7ecba202babaf822af43ab572c674057ee1ac925684a4727
diagnostics.jsonl
73106af953a79192c2246c05d5b26afc275ec086f651541e02f544b9a8876414
materialization.json
fbbaabfcfde270124a5fc2b01a126a61a1612a635025c29b45e7c406b3192de0
sources.json
4b16545034cb76f89817222d1b4d9d3ed143c281fa42eb1fd15ea27479cb2b4f
model-sources.json
4914ae82919b85b95138a82fb91997b1b9d083c221f9bf24dcc142d9a9b8554a
benchmarks/lib/transcript_metrics.py
9c8ce3b3efffb4091874a6079d554cde0ed3c4a6785416ead9a89fcefc56f1e0
Sources/ASRIPC/Protocol.swift
5b17446695fdf18b3523977d87c062c0935132defbdf27e7f4dc40cbc0ea785c
Sources/ASRServerSupport/Runner.swift
048494f59c260e2a81df6b6c9d05ba72cd447f31cc59623b568eec639c7b8e51
Sources/ASRServerSupport/ServerRuntime.swift
29b87a73994cd16b84f2fc84f799d50787ffc1363ad518ef2dc08c3e405f1cd4
Sources/NativeASR/StreamingSession.swift
973fc94722a2fdc584269df8e95857e144fe676a94f1d0f10d8c5638f255adcd
scripts/benchmark_asr_transport.py
dcf839a7d4689a0a07f40bcbef5779c9b4b501b17578f7560e748563b78f688e
```

Source HEAD: `45cfcbcb8089f02aa3c2cb4c2a2709c32e57c022`. Source excerpts reviewed as needed; no audio-byte or model-runtime validation. No edits, inference, downloads, delegation or commits. Review complete.
