# Ideas Backlog

## High Impact

- **Pre-allocated KV cache**: Replace `MLX.concatenated` in KVCache.update() with a pre-allocated buffer and in-place slice writes. Eliminates O(n²) total copy cost over decode. Check if MLX Swift supports `array[0..., 0..., offset..<offset+new, 0...] = newValues`.
- **Cache streaming attention mask**: `makeBlockMask` in AudioEncoder rebuilds the full (N×N) mask every encode call. For fixed window sizes in streaming, cache it once.
- **Cheaper reuse-length check**: `computeReuseLength` does full element-wise diff + GPU sync. Track audio buffer growth instead — if only new audio was appended, reuse length = previous embed length.

## Medium Impact

- **Avoid tail re-encode**: `encodeIncremental` re-encodes the partial tail window every chunk. Cache the tail and only re-encode when it grows past a threshold.
- **Prompt template caching**: `buildPrompt` + `embedTokens` for the system/user header is identical every chunk. Cache the prefix embeddings.
- **vDSP mel spectrogram**: Replace index-based `reflectPad1D` + MLX rfft with Accelerate/vDSP STFT. The mel computation is a small fraction of total time but has unnecessary intermediate allocations.
- **Batch eval in decode**: Audit eval placement in the decode loop — ensure asyncEval overlap is actually happening and no hidden syncs.

## Low Impact / Exploratory

- **Fused QK-norm + RoPE**: Currently two separate passes. A custom MLX kernel could fuse them, but this requires Metal shader work.
- **Speculative decoding**: Use a smaller draft model for token prediction. Probably not worth the complexity for ASR (short outputs).
- **Encoder layer pruning**: Skip later encoder layers for streaming partials (quality risk — needs careful evaluation).
- **Quantized encoder**: The audio encoder runs in bf16. Quantizing just the encoder FFN layers to 8-bit could reduce memory bandwidth.
