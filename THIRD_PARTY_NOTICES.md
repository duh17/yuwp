# Third-Party Notices

## Silero VAD

Yuwp bundles a compiled CoreML Silero VAD model at:

- `Sources/NativeASR/Resources/silero_vad.mlmodelc`

Yuwp also includes a local Swift wrapper and chunking helpers at:

- `Sources/NativeASR/SileroVAD.swift`

The current provenance we can trace in-repo is:

1. **Original model:** [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad)
   - License: MIT
   - Copyright: Silero Team
2. **Bundled CoreML conversion:** [`FluidInference/silero-vad-coreml`](https://huggingface.co/FluidInference/silero-vad-coreml)
   - The model card declares `license: mit`
   - The bundled model metadata in this repo lists the author as `Fluid Infernece + Silero Team`

`SileroVAD.swift` itself is local Yuwp code that loads the bundled CoreML model and implements the VAD/chunking helpers we use for long-form batch transcription and subtitles.

### Silero VAD license

Source: <https://github.com/snakers4/silero-vad/blob/master/LICENSE>

```text
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
