**APPROVE remains valid, now including both pre-run files.** No new blocker.

- Receipt and command JSON contain identical command objects.
- All **12 receipt hashes** match current files, including binaries, metallib and reviewed source.
- Independently verified sizes and SHA-256 for **43 candidate assets and nine Qwen assets**.
- Commands select the pinned Qwen checkpoint for streaming/final passes, stdio transport, N1 560ms and P1 v2. Qwen defaults preserve final accuracy/live VAD ON and automatic batch chunking.
- Full benchmark test log reports **51 tests passed**; inspected, not rerun.

```text
pre-run-receipt.json
5b39cada911f5c09fd8187b75ca4be6536c86e183169230965c19bf342e49a54

commands.json
08e8b18692d0dc3b59c59c4e50337f10f94109d34426bfb51b851850fe713808

benchmark-full-tests.log
9549b50684d97aa9702e4d6da363b1eaaf677edd46d99d7831b5fa08dce208ca
```

Approval remains **≤81 dev-only NON-ACCEPTANCE diagnostics**, with recorded load, explicit `diagnostic-under-load`, stratified reporting and all failures retained. Check actual Qwen stdio startup VAD evidence; HTTP readiness is not that evidence.

No candidate launch, inference, builds or writes performed.
