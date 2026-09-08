# Separately authorized P1 resolver-repaired development pass

Parent approval was received after the original 81-trial run finished and the corrected adapter passed offline readiness. No repaired Parakeet speech output existed when this scope was frozen.

One P1-only pass: the same 12 LibriSpeech dev clips, nine AMI dev segments and six synthetic dev diagnostics, in the original case order. Maximum 27 attempted trials and one hour including startup, warm-up and cleanup. Use the reviewed evaluator's `run_trial`, manifest loader, ordering and scoring unchanged. Each trial uses a fresh process, the original first-dev-short warm-up, then a clean measured session. Same 100ms packet-end clock, 1s preview cadence, 45s cap, final pass, precision, compute units, decoder and assets. Only the independently reviewed SDK local-directory resolver correction differs.

The thin driver invokes only Parakeet. No Qwen/Nemotron rerun, measured retry, tuning, third experiment, heldout execution or production integration is authorized. Keep all errors, timeouts and missing trials. Do not modify the original evaluator or 81-trial artifact.

This is **NON-ACCEPTANCE**, `no_load_control=true`, separate from the earlier interleaved run. Its latency can be described but not used as a causal or parity comparison against earlier Qwen/Nemotron timings. Quality comparisons on identical dev references remain descriptive, not confirmation. Report LibriSpeech, AMI and synthetic strata separately. An empty speech transcript is a recognition error even if transport succeeds.

Before launch, hash this scope, driver, commands, reviewed repaired binary/source and model inventory. Parent reviews the thin driver before launch. The readiness receipt is `/tmp/yuwp-english-parity/parakeet-repaired-readiness.json`. After this pass, consolidate evidence and stop; no further experiment is part of this delivery.
