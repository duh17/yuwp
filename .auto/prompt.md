# Autoresearch: Qwen3-ASR Inference Speed

## Objective

Reduce wall-clock inference time (RTF) of Yuwp's native MLX Qwen3-ASR backend
on Apple Silicon without regressing transcription quality (WER/CER).

The workload is batch transcription via `yuwp-asr transcribe` and the streaming
chunk pipeline in `StreamingSession`. The model is Qwen3-ASR 0.6B bf16 loaded
through MLX Swift. The hot path is: mel spectrogram → audio encoder (Conv2d stem
+ 24-layer windowed-attention transformer) → embedding fusion → LLM prefill
(28-layer GQA decoder with KV cache) → autoregressive token decode.

## Metrics

- **Primary**: `median_rtf` (unitless, lower is better) — median real-time factor
  across benchmark fixtures. RTF = wall_time / audio_duration.
- **Secondary**:
  - `en_rtf` — RTF on jfk.wav (11s English)
  - `long_rtf` — RTF on asr_en.wav (15s English)
  - `zh_rtf` — RTF on asr_zh.wav (4s Chinese)
  - `xl_rtf` — RTF on asr_en_long.wav (26s English)
  - `stream_rtf` — end-to-end core streaming RTF on asr_en.wav
  - `stream_prefill_ms` — median per-chunk streaming prefill latency on asr_en.wav
  - `stream_reuse_pct` — median structural prefix reuse on asr_en.wav
- **Quality gates** (run by `.auto/checks.sh`):
  - exact batch transcripts for the English JFK and Chinese fixtures
  - stream-vs-batch WER ≤1% on JFK and ≤7% on asr_en.wav, with batch correction disabled

## How to Run

`./.auto/measure.sh` — outputs `METRIC name=number` lines.

Runs `yuwp-asr transcribe --format json` on four fixtures and runs the core
streaming path on asr_en.wav with batch correction disabled, 3 repetitions each.
It reports batch RTF plus streaming RTF, prefill latency, and reuse. Both binaries
must be pre-built and newer than all tracked NativeASR sources:

```bash
swift build -c release --product yuwp-asr
swift build -c release --product asr-stream-test
```

## Files in Scope

| File | What it does | Optimization notes |
|------|-------------|-------------------|
| `Sources/NativeASR/KVCache.swift` | Per-layer KV cache (concat-based) | **High impact.** Every `update()` does `MLX.concatenated` which copies the full cache. A pre-allocated buffer with in-place writes would eliminate O(n²) total copy cost over a decode sequence. |
| `Sources/NativeASR/TextDecoder.swift` | Qwen3 LLM decoder (GQA + RoPE + SwiGLU) | Attention mask mode selection, layer fusion, eval batching. The `maskMode` branch is already optimized (.causal vs .none). Look at reducing per-token overhead in attention. |
| `Sources/NativeASR/Qwen3ASRTranscriber.swift` | Batch transcription pipeline | Double-buffer decode loop is already good. Look at: token cap heuristic, eval placement, mel→encoder→decode pipeline overlap. |
| `Sources/NativeASR/StreamingSession.swift` | Streaming chunk pipeline | `computeReuseLength` does a full element-wise diff with eval sync every chunk. `encodeIncremental` re-encodes tail every chunk. `buildInputEmbeds` rebuilds full prompt embeddings. |
| `Sources/NativeASR/AudioEncoder.swift` | Conv2d stem + windowed transformer | Block mask rebuilt every call. Conv stem loops over samples. `makeBlockMask` builds O(N²) mask on GPU. Windowed attention mask could be cached for fixed chunk sizes. |
| `Sources/NativeASR/MelSpectrogram.swift` | STFT + mel filterbank | `reflectPad1D` creates multiple intermediate arrays via index-flip. Could use vDSP or a single padded allocation. The STFT itself uses `MLX.rfft` which is fine. |
| `Sources/NativeASR/Qwen3ASRModel.swift` | Model wrapper, embedding fusion | `buildInputsEmbeds` fast path is good. `callAsFunction` creates cache list if nil — minor. |
| `Sources/NativeASR/Qwen3ASRConfig.swift` | Config structs | Read-only reference. |
| `Sources/NativeASR/Qwen3ASRTokenizer.swift` | Tokenizer + prompt building | `buildPrompt` called every chunk in streaming. Could cache prompt template. |

## Off Limits

- `Sources/NativeTTS/` — TTS code, not relevant
- `Sources/App.swift`, `Sources/DictationSession.swift`, UI files — app shell
- `Sources/ASRServerSupport/` — HTTP/stdio server plumbing (not the hot path)
- `Tests/` — do not modify tests (checks.sh runs them)
- `benchmarks/` — benchmark tooling (measure.sh is separate)
- Model weights / quantization scheme — we're optimizing the runtime, not the model
- `Package.swift` — no new dependencies

## Constraints

1. `swift build -c release --product yuwp-asr` must succeed with no errors
2. `swift test` must pass
3. WER on jfk.wav must not regress more than 1% absolute from baseline
4. No new package dependencies
5. Swift 6 strict concurrency — no new warnings
6. MLX is NOT thread-safe — all inference through inferenceLock (don't break this)
7. Keep the public API stable (TranscriptionResult, ChunkResult, StreamConfig)

## Architecture Notes

### KV Cache (biggest opportunity)
Current: `MLX.concatenated([existing, newKeys], axis: 2)` on every token.
For a 100-token decode, this copies 1+2+3+...+100 = 5050 position-slots total.
A pre-allocated cache (allocate maxLen upfront, write at offset) would make each
update O(1) instead of O(n). MLX supports `array[offset..offset+new] = newValues`
via slice assignment. Check if MLX Swift supports in-place slice update.

### Streaming Reuse Length
`computeReuseLength` evaluates `MLX.abs(prev - new).sum()` + `argMax` with a full
GPU sync every chunk. For typical chunks where reuse is high, a cheaper heuristic
(e.g., compare only the first/last few positions, or track audio buffer growth)
could avoid the sync.

### Audio Encoder Mask
`makeBlockMask` builds a full (seqLen × seqLen) attention mask every encode call.
For streaming with fixed window sizes, this mask is identical across chunks and
could be cached.

### Eval Placement
MLX is lazy — `eval()` forces GPU execution. Unnecessary evals create sync points.
But too-few evals mean the GPU pipeline can't overlap. The current code has
reasonable eval placement; audit for any that can be removed or batched.

## What's Been Tried

### Profiling Data (0.6B bf16, warm, jfk.wav 11s)
- mel: ~3ms | encoder: ~8-10ms | prefill: ~14ms | decode: ~3.5ms/token
- Decode is memory-bandwidth-bound (~3ms = full model weight read at 400GB/s)
- Encoder/prefill are compute-bound, already near hardware limits
- Total warm: ~35-50ms for 11s audio

### Kept
1. **reflectPad1D simplification** (iter 1): single index gather replaces multi flip+concat. ~1% gain, cleaner code.
2. **LM head last-position-only** (iter 2): slice hidden to last token before vocab projection. ~0.7% batch gain, bigger for 1.7B.
3. **Merged delta+last forward pass** (iter 4): two 28-layer decoder calls → one. Streaming quality gates cover the merged path.
4. **Structural reuse estimation** (iter 5): replaced GPU-sync computeReuseLength with integer tracking. Cache-origin tracking makes eviction explicit.

### Discarded
5. **Unmasked SDPA for single-window encoder** (iter 6): 4% regression on zh fixture. MLX masked SDPA with all-zero mask is faster than .none mode for small sequences.

### Superseded
- **skipLMHead for streaming delta** (iter 3): removed after the merged forward pass made the separate delta-only call obsolete.

### Dead Ends
- KV cache pre-allocation: MLX scatter creates new array internally, no benefit for short sequences (~30 tokens)
- Decode loop CPU overhead: <0.1ms per token vs 3.5ms GPU time
- Mel spectrogram: only 3ms, not worth optimizing
- Prompt template caching: 9 tokens × 1024 dims, negligible
