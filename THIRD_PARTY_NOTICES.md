# Third-Party Notices

This document separates third-party material into three buckets:

1. Bundled in the repository and/or packaged app
2. Downloaded at runtime, but not redistributed in the repository or DMG
3. Acknowledged implementation references that informed Yuwp but are not bundled

The packaged macOS app also ships these notices plus the vendored license files under
`Contents/Resources/OpenSource/`.

## Bundled In The Repo Or App

### Sparkle

- Upstream: <https://github.com/sparkle-project/Sparkle>
- What Yuwp ships: `Sparkle.framework` inside the packaged app bundle
- License: MIT, plus Sparkle's bundled third-party license notices
- Vendored license text: `third_party/licenses/Sparkle-LICENSE.txt`

### MLX Swift / MLX Runtime Support

- Upstream: <https://github.com/ml-explore/mlx-swift>
- What Yuwp ships:
  - Swift code linked into the app and server binaries via SwiftPM
  - `mlx.metallib`, compiled from the checked-out MLX Metal kernels by `scripts/build_mlx_metallib.sh`
- License: MIT
- Vendored license text: `third_party/licenses/mlx-swift-LICENSE.txt`

### MLX Swift LM

- Upstream: <https://github.com/ml-explore/mlx-swift-lm>
- What Yuwp ships: Swift helper code linked into the native TTS test/tooling path via SwiftPM
- License: MIT
- Vendored license text: `third_party/licenses/mlx-swift-lm-LICENSE.txt`

### Tencent AuK (AuK-Flash and AuK Base native MLX port)

- Upstream: <https://github.com/Tencent-Hunyuan/AuK>
- Reference revision: `feat/mlx-apple-silicon` at `6943a1e967409e8c73139a7a345f2a611cfb3dd6`
- What Yuwp ships: a native Swift/MLX inference port of the official AuK-Flash and AuK Base MLX backend in `Sources/NativeTTS` (`AuK*.swift`), plus a one-time PyTorch→MLX converter invoked as `yuwp-tts convert-auk`. Runtime inference does not bundle or call Python.
- License: MIT
- Vendored license text: `third_party/licenses/auk-LICENSE.txt`
- Credit: Tencent / Hunyuan for AuK training and inference code, including the official MLX Apple Silicon backend

### MLX Audio Swift Qwen3-TTS Port

- Upstream: <https://github.com/Blaizzy/mlx-audio-swift>
- What Yuwp ships: selected Swift source files ported into `Sources/NativeTTS` as the initial Qwen3-TTS implementation reference
- License: MIT
- Vendored license text: `third_party/licenses/mlx-audio-swift-LICENSE.txt`

### Swift Transformers / Hugging Face Swift

- Upstreams:
  - <https://github.com/huggingface/swift-transformers>
  - <https://github.com/huggingface/swift-huggingface>
- What Yuwp ships: tokenizer / Hugging Face support code linked through SwiftPM for NativeTTS
- License: Apache License 2.0
- Vendored license text:
  - `third_party/licenses/swift-transformers-LICENSE.txt`
  - `third_party/licenses/swift-huggingface-LICENSE.txt`
  - `third_party/licenses/Apache-2.0.txt`

### Swift Numerics

- Upstream: <https://github.com/apple/swift-numerics>
- What Yuwp ships: transitive SwiftPM dependency used through the MLX Swift stack
- License: Apache License 2.0
- Vendored license text:
  - `third_party/licenses/swift-numerics-LICENSE.txt`
  - `third_party/licenses/Apache-2.0.txt`

### Silero VAD CoreML Artifact

Yuwp bundles a compiled CoreML VAD model at:

- `Sources/NativeASR/Resources/silero_vad.mlmodelc`

Yuwp loads that model through:

- `Sources/NativeASR/SileroVAD.swift`

Provenance:

1. Original model: `snakers4/silero-vad`
   - Upstream: <https://github.com/snakers4/silero-vad>
   - License: MIT
   - Copyright: Silero Team
2. Apple-platform CoreML conversion used for the bundled artifact:
   - Upstream: <https://huggingface.co/FluidInference/silero-vad-coreml>
   - Model card attribution: "Developed by: Silero Team (original), converted by FluidAudio"
   - License: MIT
   - Credit: FluidInference / FluidAudio for the CoreML conversion that Yuwp packages

The bundled repository artifact is the compiled CoreML form of that converted model, not the
original PyTorch release.

Vendored notice files:

- `third_party/licenses/silero-vad-LICENSE.txt`
- `third_party/licenses/silero-vad-coreml-NOTICE.txt`

## Runtime-Downloaded Models (Not Bundled In The Repo Or DMG)

Yuwp can download or use locally cached ASR / aligner models at runtime. Those model weights are
not redistributed in this repository or the packaged macOS app, but they are first-class third
party dependencies of the product.

### Qwen3-ASR

- Upstream model card: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- Typical Yuwp defaults also include MLX-community converted variants such as
  `mlx-community/Qwen3-ASR-0.6B-4bit`
- License: Apache License 2.0
- Vendored Apache 2.0 text: `third_party/licenses/Apache-2.0.txt`
- Credit: Qwen team for the underlying ASR model family

### AuK-Flash, AuK Base, And Qwen2.5-Omni-3B

- Upstream model cards / repositories:
  - <https://github.com/Tencent-Hunyuan/AuK>
  - <https://huggingface.co/tencent/AuK>
  - <https://huggingface.co/tencent/AuK-Flash>
  - <https://huggingface.co/Qwen/Qwen2.5-Omni-3B>
- Role in Yuwp: runtime-downloaded weights for native AuK-Flash / AuK Base TTS and audio editing (`yuwp-tts`). The Thinker encoder is Qwen2.5-Omni-3B; AuK DiT + VAE weights are converted locally to MLX safetensors.
- License: MIT (AuK weights/code as published by Tencent); Apache License 2.0 (Qwen2.5-Omni)
- Vendored license text:
  - `third_party/licenses/auk-LICENSE.txt`
  - `third_party/licenses/Apache-2.0.txt`

### Confucius4-R2T2

- Upstream repository: <https://github.com/netease-youdao/Confucius4-R2T2>
- MLX-community conversions Yuwp can download:
  - <https://huggingface.co/mlx-community/Confucius4-R2T2-8bit>
  - <https://huggingface.co/mlx-community/Confucius4-R2T2-bf16>
- Role in Yuwp: optional runtime-downloaded streaming ASR weights. Same Qwen3-ASR graph; Yuwp selects the 160 ms longest-stable-prefix loop when the model path contains `r2t2`.
- License: NetEase Youdao Model Use License (personal / small commercial use as published with the weights). Not Apache 2.0.
- Weights are not bundled in the DMG; users download them from Hugging Face.

### Qwen3 Forced Aligner

- Official upstream model: <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>
- Default Yuwp aligner path today points at the MLX-community conversion:
  <https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit>
- License: Apache License 2.0
- Vendored Apache 2.0 text: `third_party/licenses/Apache-2.0.txt`
- Credit:
  - Qwen team for the original forced aligner model
  - MLX Community for the MLX conversion / quantized release Yuwp uses by default

## Acknowledgments / Implementation References

The following projects informed the implementation, evaluation, or streaming design, but Yuwp does
not bundle their source code unless separately noted above.

### qwen-asr

- Upstream: <https://github.com/antirez/qwen-asr>
- Role in Yuwp: streaming / rollback / reference-implementation ideas during native ASR work
- License: MIT
- Redistribution status: acknowledged reference, not bundled as a shipped library in Yuwp

### Qwen Official Tooling And Model Cards

- Upstream: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- Role in Yuwp: upstream model architecture, tokenizer/config expectations, and model-level docs
- Redistribution status: runtime-downloaded models only, not packaged in the app bundle
