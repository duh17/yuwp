# Reconstruct the frozen development experiment

These are **non-production experiment sources**, not installed ASR backends. The checkout contains the fixture generator, pinned asset manifest, original native adapter, narrow resolver patch, diagnostic driver and scoring scripts. Audio, model weights, executables and machine caches are not in git. Do not describe the benchmark as runnable until those inputs are staged.

The sources retain the experiment's explicit `/tmp/yuwp-english-parity` layout. Use a fresh isolated host/staging root; **do not overwrite an existing experiment at that path**. Replaying inference needs separate authorization and is not part of a unit test. Neither this recipe nor the archived driver authorizes heldout inference or another run.

## Public audio

Requirements: Python 3.14 through `uv`, FFmpeg 8.0.1 (the recorded version), `curl`, and `tar`. Run from the Yuwp repository root. The original materializer is preserved byte-for-byte: SHA-256 `4406c6611e4d7526386dfbe2f311944ca807c6f6892e05f297d20f9eaa109f11`.

```bash
(
set -euo pipefail
# A subshell failure stops this entire block, even in a normal interactive shell.
if [[ -e /tmp/yuwp-english-parity || -L /tmp/yuwp-english-parity ]]; then
  printf '%s\n' 'Refusing to overwrite existing experiment staging.' >&2
  exit 1
fi
mkdir -p /tmp/yuwp-english-parity/data/source/extracted
curl --fail --location https://www.openslr.org/resources/12/test-clean.tar.gz \
  --output /tmp/yuwp-english-parity/data/source/test-clean.tar.gz
printf '%s  %s\n' \
  39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23 \
  /tmp/yuwp-english-parity/data/source/test-clean.tar.gz | shasum -a 256 -c -
printf '%s  %s\n' 32fa31d27d2e1cad72775fee3f4849a9 test-clean.tar.gz \
  > /tmp/yuwp-english-parity/data/source/md5sum.txt
tar -xzf /tmp/yuwp-english-parity/data/source/test-clean.tar.gz \
  -C /tmp/yuwp-english-parity/data/source/extracted
)
```

The materializer reads the existing tracked AMI fixture at `benchmarks/fixtures/subtitle-long/ami-en2002b-d-0765-0945`, plus the LibriSpeech archive. It writes the three manifests and `materialization.json` in this fixture directory. **Run it only in a disposable checkout, not over the frozen report checkout.** Save and compare the frozen manifest hashes before accepting reconstructed data:

```bash
(
set -euo pipefail
if [[ -e /tmp/yuwp-english-parity/data/audio || -L /tmp/yuwp-english-parity/data/audio ]]; then
  printf '%s\n' 'Refusing to overwrite materialized audio; use a fresh disposable checkout/staging root.' >&2
  exit 1
fi
uv run --python 3.14 --no-project \
  benchmarks/fixtures/english-parity/reproduction/materialize.py
)
```

Expected dev/heldout/diagnostic hashes are in the committed `materialization.json`. Environment fields in a newly generated materialization receipt can differ; the audio/PCM/reference hashes and all three manifests must match. Generating heldout audio is not permission to infer on it. LibriSpeech and AMI are CC BY 4.0; attribution and original URLs are in `sources.json`.

## Pinned model files

From the repository root:

```bash
(
set -euo pipefail
if [[ -e /tmp/yuwp-english-parity/models || -L /tmp/yuwp-english-parity/models ]]; then
  printf '%s\n' 'Refusing to overwrite existing model staging.' >&2
  exit 1
fi
uv run --python 3.14 --no-project \
  benchmarks/fixtures/english-parity/reproduction/download-models.py --include-qwen
)
```

This explicitly downloads public assets selected by `model-sources.json` at its frozen revisions. It checks size and LFS SHA-256 or Git blob identity before renaming each partial file. It does not load models. The acquisition helper is the original download logic with an explicit option to acquire Qwen too; the actual run reused Qwen's verified local HF cache. Freshly acquired Qwen files stage under `/tmp/yuwp-english-parity/models/Qwen3-ASR-1.7B-bf16`. Record hashes and use that directory for **both** `--model` and `--batch-model`; do not substitute another size or precision.

## Native prototype, including the preserved resolver failure

Requires Swift 6, Apple Silicon/macOS, and Core ML. The recorded compiler was Apple Swift 6.3.3. Do not install system components or change active Xcode as part of this recipe.

```bash
(
set -euo pipefail
for destination in /tmp/yuwp-english-parity/FluidAudio /tmp/yuwp-english-parity/adapter; do
  if [[ -e "$destination" || -L "$destination" ]]; then
    printf 'Refusing to overwrite %s\n' "$destination" >&2
    exit 1
  fi
done
git clone https://github.com/FluidInference/FluidAudio.git /tmp/yuwp-english-parity/FluidAudio
git -C /tmp/yuwp-english-parity/FluidAudio checkout --detach \
  5c19d5e12320e22bbfb7a1877b089d2665a69add
cp -R benchmarks/fixtures/english-parity/reproduction/adapter \
  /tmp/yuwp-english-parity/adapter
bash /tmp/yuwp-english-parity/adapter/validate.sh
)
```

`validate.sh` builds and runs model-free tests. It checks the FluidAudio source pin and a clean tracked SDK diff. SwiftPM obtains the manifest-checksummed NemoTextProcessing v0.3.0 binary dependency; that dependency is not a system installation. Upstream warnings are not a warning-free build.

The original adapter intentionally preserves the first run's bad HF-basename guard. **Do not silently fix it and call that a replay of the original run.** To reconstruct the separately reviewed P1 readiness repair:

```bash
(
set -euo pipefail
for destination in /tmp/yuwp-english-parity/adapter-readiness-repair /tmp/yuwp-english-parity/repair-assets; do
  if [[ -e "$destination" || -L "$destination" ]]; then
    printf 'Refusing to overwrite %s\n' "$destination" >&2
    exit 1
  fi
done
mkdir /tmp/yuwp-english-parity/adapter-readiness-repair
cp -R benchmarks/fixtures/english-parity/reproduction/adapter/. \
  /tmp/yuwp-english-parity/adapter-readiness-repair/
patch -d /tmp/yuwp-english-parity/adapter-readiness-repair -p1 \
  < benchmarks/fixtures/english-parity/reproduction/parakeet-resolver-repair.patch
mkdir /tmp/yuwp-english-parity/repair-assets
ln -s /tmp/yuwp-english-parity/models/parakeet-tdt-0.6b-v2-coreml \
  /tmp/yuwp-english-parity/repair-assets/parakeet-tdt-0.6b-v2
bash /tmp/yuwp-english-parity/adapter-readiness-repair/validate.sh
)
```

The patch changes the local folder guard, its regression test/test dependency, and the validation script's isolated directory/inventory. No model setting changes. Native source is separate from the root Yuwp package and cannot enable a production backend.

## Baseline and execution boundary

Build `yuwp-asr` from Yuwp source `45cfcbcb8089f02aa3c2cb4c2a2709c32e57c022` with its locked dependencies, not a later default. Follow the repository's release-build instructions. This run could not compile Metal because the component was absent; the parent authorized reuse of a cache whose complete 39-file Metal source hash matched `591d42890ef3702198977a5aff602183` at mlx-swift `3b11207d4870fc2b703fc6c7931741aa196ec914`. The recorded shader SHA-256 is in the receipt. A new reconstruction needs a working compiler or independently verified source-matched cache; an arbitrary existing metallib is not acceptable.

The committed receipts identify this run's exact commands, source/binary/asset hashes and runtime outcomes. Their absolute host paths are evidence, not portable commands. Adapt only staging paths for a new run, verify identical input bytes, and freeze new receipts before inference. Read the protocol, scope amendments and report before using `asr english-parity`.

`parakeet-retry.py` is the exact parent-reviewed thin supplemental driver, not a general retry command. It requires the original completed-run receipt, successful repaired readiness receipt and matching pre-run hashes. The scoring scripts preserve original versus supplemental evidence and stratify LibriSpeech, AMI and synthetic diagnostics. They consume recorded outputs; they do not run models. Raw event artifacts are linked from the report; compact per-trial metrics/transcripts and their hashes remain in the checkout so the reported counts and quantiles do not depend only on temporary files.
