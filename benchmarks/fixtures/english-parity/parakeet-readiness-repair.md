# Parakeet local-path readiness repair exception

The parent authorized this exception after the original dev run began, when Parakeet failed before receiving audio. This is not a quality-tuning change or permission to rerun the frozen experiment.

## Original result is preserved

Original adapter `/tmp/yuwp-english-parity/adapter`, commands, model paths, source archive and `dev-run.jsonl` remain unchanged. Every original P1 trial still uses that adapter and must retain its failure. The error is a resolver mismatch: the adapter requires HF basename `parakeet-tdt-0.6b-v2-coreml`; pinned FluidAudio resolves assets under `Repo.parakeetV2.folderName`, `parakeet-tdt-0.6b-v2`. The verified asset files exist, but the SDK looks in a different sibling directory.

## Narrow authority

Create a separate repaired adapter and staging path. Use the SDK's public folderName rather than duplicating the HF repo basename. Stage the same hash-verified assets under that local name without changing bytes. No weight, precision, decoder, compute-unit, preview cadence, final-pass or transcript-policy change is permitted.

Code-only repair and model-free validation may proceed during the original run. **No repaired model load, health probe, or inference until that run finishes.** Independently review the exact repair and validate offline readiness only. Preserve original and repaired source/binary/path hashes and logs.

A P1 diagnostic retry needs a separate pre-results freeze and explicit parent approval. This exception grants no retry, heldout inference, acceptance claim, production support or landing. Do not call the pre-audio resolver failure evidence about Parakeet recognition quality.
