import Foundation
import NativeTTS
import YuwpHTTPServerSupport

func runAuKConvert(_ args: [String]) throws {
    var source: String?
    var thinker: String?
    var output: String?
    var bits: Int?
    var i = 0
    while i < args.count {
        let arg = args[i]
        func takeValue() -> String? {
            guard i + 1 < args.count else { return nil }
            i += 1
            return args[i]
        }
        switch arg {
        case "--src", "--source": source = takeValue()
        case "--thinker-src", "--thinker", "--qwen": thinker = takeValue()
        case "--out", "--output", "--dst": output = takeValue()
        case "--bits": bits = takeValue().flatMap(Int.init)
        case "--help", "-h":
            print("""
            Usage: yuwp-tts convert-auk --src <AuK-Flash-or-AuK-dir> --thinker-src <Qwen2.5-Omni-3B or auk-flash-mlx> --out <mlx-dir> [--bits 8]

            Converts official PyTorch AuK-Flash or AuK Base weights to MLX safetensors.
            Detects the variant from the source directory (auk_flash.safetensors vs auk_base.safetensors).
            --thinker-src may be Qwen2.5-Omni-3B or an already-converted mlx directory (reuses thinker).
            Runtime inference is pure Swift/MLX and does not call Python.
            """)
            return
        default:
            break
        }
        i += 1
    }
    guard let source, let thinker, let output else {
        throw AuKError.invalidInput("convert-auk requires --src, --thinker-src, and --out")
    }
    if let bits, bits != 4 && bits != 8 {
        throw AuKError.invalidInput("--bits must be 4 or 8")
    }
    let result = try AuKConvert.convertDirectory(
        source: URL(fileURLWithPath: source).standardizedFileURL,
        thinkerSource: URL(fileURLWithPath: thinker).standardizedFileURL,
        output: URL(fileURLWithPath: output).standardizedFileURL,
        bits: bits
    )
    let variant: String
    switch inspectAuKModelDirectory(result.outputDirectory) {
    case .converted(let detected), .pytorchSource(let detected):
        variant = detected.rawValue
    case .unknown:
        variant = "unknown"
    }
    fputs(
        "Converted AuK (\(variant)) to \(result.outputDirectory.path) (vae=\(result.vaeTensors) dit=\(result.ditTensors) thinker=\(result.thinkerTensors))\n",
        stderr
    )
}

func runAuKGenerate(options: TTSTestOptions) async throws {
    guard let modelPath = options.modelPath else {
        throw AuKError.invalidInput("--model <auk-mlx-dir> is required")
    }
    let plan = try aukPlan(from: options)
    let thinkerURL = options.thinkerPath.map { URL(fileURLWithPath: $0).standardizedFileURL }
    let engine = try await AuKEngine.load(
        modelDirectory: URL(fileURLWithPath: modelPath).standardizedFileURL,
        thinkerDirectory: thinkerURL,
        bits: options.bits
    )
    let refURL = options.referenceAudioPath.map { URL(fileURLWithPath: $0).standardizedFileURL }
    var chunkSamples: [[Float]] = []
    for (index, chunk) in plan.chunks.enumerated() {
        let (samples, _) = try engine.generate(
            instruction: chunk.instruction,
            referenceAudioURL: refURL,
            genSeconds: chunk.genSeconds,
            seed: options.seed,
            nfe: options.nfe,
            cfgStrength: options.cfg,
            sway: options.sway
        )
        chunkSamples.append(samples)
        if options.stream {
            fputs(
                "AuK \(engine.variant.rawValue) chunk \(index + 1)/\(plan.chunks.count): \(String(format: "%.2f", Double(samples.count) / Double(engine.sampleRate)))s\n",
                stderr
            )
        }
    }
    let samples = aukConcatenateTTSChunks(
        chunkSamples,
        sampleRate: engine.sampleRate,
        pauseSeconds: plan.pauseSeconds
    )
    try AudioUtils.writeWavFile(
        samples: samples,
        sampleRate: Double(engine.sampleRate),
        fileURL: URL(fileURLWithPath: options.outputPath)
    )
    let duration = Double(samples.count) / Double(engine.sampleRate)
    fputs(
        "\(engine.modelName) synthesized \(String(format: "%.2f", duration))s audio (\(plan.chunks.count) chunk(s)) -> \(options.outputPath)\n",
        stderr
    )
    if options.play {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        proc.arguments = [options.outputPath]
        try proc.run()
        proc.waitUntilExit()
    }
}

func runAuKServe(options: TTSTestOptions) async throws {
    guard let modelPath = options.modelPath else {
        throw AuKError.invalidInput("serve requires --model <auk-mlx-dir>")
    }
    let loadStart = Date()
    let thinkerURL = options.thinkerPath.map { URL(fileURLWithPath: $0).standardizedFileURL }
    let engine = try await AuKEngine.load(
        modelDirectory: URL(fileURLWithPath: modelPath).standardizedFileURL,
        thinkerDirectory: thinkerURL,
        bits: options.bits
    )
    let loadSeconds = Date().timeIntervalSince(loadStart)
    let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
    let gate = AsyncGate()
    let sampling = engine.defaultSampling

    if options.transport == "http" {
        startHTTPServer(
            config: HTTPServerConfig(host: options.host, port: options.port),
            handler: { request in
                let path = request.path.split(separator: "?").first.map(String.init) ?? request.path
                if path == "/v1/info" {
                    guard request.method == "GET" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
                    var info: [String: Any] = [
                        "status": "ready",
                        "service": "yuwp-tts",
                        "model": modelName,
                        "backend": engine.backend,
                        "variant": engine.variant.rawValue,
                        "sample_rate": engine.sampleRate,
                        "load_seconds": loadSeconds,
                        "nfe": sampling.nfe,
                        "cfg": sampling.cfgStrength,
                    ]
                    if engine.variant == .base {
                        info["sway"] = sampling.sway as Any
                    }
                    return jsonResponse(status: 200, info)
                }
                guard path == "/v1/audio/speech" || path == "/v1/audio/speech/stream" else {
                    return jsonResponse(status: 404, ["error": "unknown endpoint: \(request.method) \(path)"])
                }
                guard request.method == "POST" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
                let speechRequest: OpenAISpeechRequest
                do {
                    speechRequest = try JSONDecoder().decode(OpenAISpeechRequest.self, from: request.body)
                } catch {
                    return jsonResponse(status: 400, ["error": "invalid JSON request: \(error.localizedDescription)"])
                }
                switch aukPlanResult(from: speechRequest, options: options) {
                case .rejected(let rejection):
                    return jsonResponse(status: 400, rejection.jsonObject)
                case .plan:
                    break
                }
                if path == "/v1/audio/speech/stream" || speechRequest.stream == true {
                    return streamingResponse(status: 200, contentType: "application/x-ndjson; charset=utf-8") { writer in
                        let semaphore = DispatchSemaphore(value: 0)
                        Task {
                            await writeAuKSpeechStream(
                                engine: engine,
                                request: speechRequest,
                                options: options,
                                gate: gate,
                                writer: writer
                            )
                            semaphore.signal()
                        }
                        semaphore.wait()
                    }
                }
                let semaphore = DispatchSemaphore(value: 0)
                final class Box: @unchecked Sendable { var response: HTTPResponse? }
                let box = Box()
                Task {
                    box.response = await aukSpeechResponse(
                        engine: engine,
                        request: speechRequest,
                        options: options,
                        gate: gate
                    )
                    semaphore.signal()
                }
                semaphore.wait()
                return box.response ?? jsonResponse(status: 500, ["error": "speech synthesis failed"])
            }
        )
    }

    emitJSON([
        "event": "ready",
        "sampleRate": engine.sampleRate,
        "loadSeconds": loadSeconds,
        "backend": engine.backend,
        "variant": engine.variant.rawValue,
        "nfe": sampling.nfe,
        "cfg": sampling.cfgStrength,
    ])

    let decoder = JSONDecoder()
    while let line = readLine() {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { continue }
        if trimmed == "shutdown" { break }
        let request: TTSServeRequest
        do {
            request = try decoder.decode(TTSServeRequest.self, from: Data(trimmed.utf8))
        } catch {
            emitJSON(["event": "error", "error": "Invalid request: \(error.localizedDescription)"])
            continue
        }
        let requestID = request.id ?? UUID().uuidString
        let outputPath = request.out ?? "/tmp/yuwp-tts-serve-\(requestID).wav"
        switch aukPlanResult(from: request, options: options) {
        case .rejected(let rejection):
            emitJSON([
                "event": "error",
                "id": requestID,
                "error": rejection.message,
                "code": rejection.code,
                "options": rejection.options,
            ])
            continue
        case .plan(let plan):
            let refPath = request.refAudio ?? options.referenceAudioPath
            await gate.enter()
            do {
                var chunkSamples: [[Float]] = []
                var chunkCount = 0
                let synthStart = Date()
                var firstAudio: TimeInterval?
                for chunk in plan.chunks {
                    let (samples, sampleRate) = try engine.generate(
                        instruction: chunk.instruction,
                        referenceAudioURL: refPath.map { URL(fileURLWithPath: $0) },
                        genSeconds: chunk.genSeconds,
                        seed: options.seed,
                        nfe: request.nfe ?? options.nfe,
                        cfgStrength: request.cfgStrength ?? request.cfg ?? options.cfg,
                        sway: request.sway ?? options.sway
                    )
                    if firstAudio == nil {
                        firstAudio = Date().timeIntervalSince(synthStart)
                    }
                    chunkCount += 1
                    chunkSamples.append(samples)
                    if request.emitChunks == true {
                        emitJSON([
                            "event": "chunk",
                            "id": requestID,
                            "chunk": chunkCount,
                            "samples": samples.count,
                            "seconds": Double(samples.count) / Double(sampleRate),
                            "elapsedSeconds": Date().timeIntervalSince(synthStart),
                        ])
                    }
                }
                let samples = aukConcatenateTTSChunks(
                    chunkSamples,
                    sampleRate: engine.sampleRate,
                    pauseSeconds: plan.pauseSeconds
                )
                try AudioUtils.writeWavFile(
                    samples: samples,
                    sampleRate: Double(engine.sampleRate),
                    fileURL: URL(fileURLWithPath: outputPath)
                )
                await gate.leave()
                emitJSON([
                    "event": "done",
                    "id": requestID,
                    "out": outputPath,
                    "firstAudioSeconds": firstAudio ?? -1,
                    "audioDurationSeconds": Double(samples.count) / Double(engine.sampleRate),
                    "wallSeconds": Date().timeIntervalSince(synthStart),
                    "chunks": chunkCount,
                ])
            } catch {
                await gate.leave()
                emitJSON(["event": "error", "id": requestID, "error": error.localizedDescription])
            }
        }
    }
}

private func aukPlan(from options: TTSTestOptions) throws -> AuKSpeechPlan {
    switch aukPlanResult(from: options) {
    case .plan(let plan):
        return plan
    case .rejected(let rejection):
        throw AuKError.invalidInput(rejection.message)
    }
}

private func aukPlanResult(from options: TTSTestOptions) -> AuKSpeechPlanResult {
    AuKSpeechPlanner.plan(
        AuKSpeechPlanRequest(
            task: options.task,
            input: options.textSet || options.instruction == nil ? options.text : nil,
            instruction: options.instruction,
            hasSourceAudio: options.referenceAudioPath != nil,
            autoChunk: options.autoChunk,
            interChunkPauseSeconds: options.chunkPauseMs.map { $0 / 1000 },
            genSeconds: options.genSeconds
        )
    )
}

private func aukPlanResult(from request: OpenAISpeechRequest, options: TTSTestOptions) -> AuKSpeechPlanResult {
    AuKSpeechPlanner.plan(
        AuKSpeechPlanRequest(
            task: request.task,
            mode: request.mode,
            input: request.input,
            instruction: request.instruction,
            instructions: request.instructions,
            hasSourceAudio: (request.refAudio ?? options.referenceAudioPath) != nil,
            autoChunk: request.autoChunk,
            chunkTargetCharacters: request.chunkTargetCharacters,
            chunkHardCharacterLimit: request.chunkHardCharacterLimit,
            interChunkPauseSeconds: request.resolvedPauseSeconds ?? options.chunkPauseMs.map { $0 / 1000 },
            genSeconds: request.genSeconds ?? options.genSeconds
        )
    )
}

private func aukPlanResult(from request: TTSServeRequest, options: TTSTestOptions) -> AuKSpeechPlanResult {
    AuKSpeechPlanner.plan(
        AuKSpeechPlanRequest(
            task: request.task ?? options.task,
            mode: request.mode,
            input: request.text,
            instruction: request.instruction,
            hasSourceAudio: (request.refAudio ?? options.referenceAudioPath) != nil,
            autoChunk: request.autoChunk ?? options.autoChunk,
            chunkTargetCharacters: request.chunkTargetCharacters,
            chunkHardCharacterLimit: request.chunkHardCharacterLimit,
            interChunkPauseSeconds: request.interChunkPauseSeconds
                ?? request.chunkPauseMs.map { $0 / 1000 }
                ?? options.chunkPauseMs.map { $0 / 1000 },
            genSeconds: request.genSeconds ?? options.genSeconds
        )
    )
}

private func aukHTTPError(_ error: Error) -> HTTPResponse {
    let status: Int
    if let aukError = error as? AuKError {
        switch aukError {
        case .invalidInput:
            status = 400
        default:
            status = 500
        }
    } else {
        status = 500
    }
    return jsonResponse(status: status, ["error": error.localizedDescription])
}

private func aukSpeechResponse(
    engine: AuKEngine,
    request: OpenAISpeechRequest,
    options: TTSTestOptions,
    gate: AsyncGate
) async -> HTTPResponse {
    let responseFormat = request.responseFormat?.lowercased() ?? "wav"
    guard responseFormat == "wav" else {
        return jsonResponse(status: 400, ["error": "unsupported response_format '\(responseFormat)'; only wav is currently supported"])
    }
    let plan: AuKSpeechPlan
    switch aukPlanResult(from: request, options: options) {
    case .rejected(let rejection):
        return jsonResponse(status: 400, rejection.jsonObject)
    case .plan(let prepared):
        plan = prepared
    }
    let refPath = request.refAudio ?? options.referenceAudioPath
    await gate.enter()
    do {
        var chunkSamples: [[Float]] = []
        for chunk in plan.chunks {
            let (samples, _) = try engine.generate(
                instruction: chunk.instruction,
                referenceAudioURL: refPath.map { URL(fileURLWithPath: $0) },
                genSeconds: chunk.genSeconds,
                seed: nil,
                nfe: request.nfe ?? options.nfe,
                cfgStrength: request.resolvedCFG ?? options.cfg,
                sway: request.sway ?? options.sway
            )
            chunkSamples.append(samples)
        }
        let samples = aukConcatenateTTSChunks(
            chunkSamples,
            sampleRate: engine.sampleRate,
            pauseSeconds: plan.pauseSeconds
        )
        let outputPath = "/tmp/yuwp-tts-http-\(UUID().uuidString).wav"
        try AudioUtils.writeWavFile(
            samples: samples,
            sampleRate: Double(engine.sampleRate),
            fileURL: URL(fileURLWithPath: outputPath)
        )
        let data = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        try? FileManager.default.removeItem(atPath: outputPath)
        await gate.leave()
        return binaryResponse(status: 200, body: data, contentType: "audio/wav")
    } catch {
        await gate.leave()
        return aukHTTPError(error)
    }
}

private func writeAuKSpeechStream(
    engine: AuKEngine,
    request: OpenAISpeechRequest,
    options: TTSTestOptions,
    gate: AsyncGate,
    writer: HTTPStreamWriter
) async {
    func emit(_ object: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        else { return }
        var payload = data
        payload.append(0x0A)
        writer.write(payload)
    }

    let plan: AuKSpeechPlan
    switch aukPlanResult(from: request, options: options) {
    case .rejected(let rejection):
        emit(rejection.jsonObject.merging(["event": "error"]) { _, new in new })
        return
    case .plan(let prepared):
        plan = prepared
    }
    let refPath = request.refAudio ?? options.referenceAudioPath
    await gate.enter()
    do {
        try AuKStreamSession.run(
            sampleRate: engine.sampleRate,
            backend: engine.backend,
            variant: engine.variant.rawValue,
            plan: plan,
            generate: { chunk in
                let (samples, _) = try engine.generate(
                    instruction: chunk.instruction,
                    referenceAudioURL: refPath.map { URL(fileURLWithPath: $0) },
                    genSeconds: chunk.genSeconds,
                    seed: nil,
                    nfe: request.nfe ?? options.nfe,
                    cfgStrength: request.resolvedCFG ?? options.cfg,
                    sway: request.sway ?? options.sway
                )
                return samples
            },
            emit: { event in
                switch event {
                case .audio(let audio):
                    emit(aukSpeechStreamJSON(event, audioBase64: pcm16Base64(samples: audio.pcm)))
                default:
                    emit(aukSpeechStreamJSON(event))
                }
            }
        )
        await gate.leave()
    } catch {
        await gate.leave()
        emit(["event": "error", "error": error.localizedDescription])
    }
}

private func pcm16Base64(samples: [Float]) -> String {
    var data = Data(capacity: samples.count * 2)
    for sample in samples {
        let clamped = max(-1.0, min(1.0, sample))
        var pcm = Int16((clamped * Float(Int16.max)).rounded()).littleEndian
        withUnsafeBytes(of: &pcm) { data.append(contentsOf: $0) }
    }
    return data.base64EncodedString()
}
