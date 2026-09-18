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
            Usage: yuwp-tts convert-auk --src <AuK-Flash-dir> --thinker-src <Qwen2.5-Omni-3B> --out <mlx-dir> [--bits 8]

            Converts official PyTorch AuK-Flash + Qwen2.5-Omni Thinker weights to MLX safetensors.
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
    fputs(
        "Converted AuK-Flash to \(result.outputDirectory.path) (vae=\(result.vaeTensors) dit=\(result.ditTensors) thinker=\(result.thinkerTensors))\n",
        stderr
    )
}

func runAuKGenerate(options: TTSTestOptions) async throws {
    guard let modelPath = options.modelPath else {
        throw AuKError.invalidInput("--model <auk-mlx-dir> is required")
    }
    guard let instruction = aukResolveInstruction(
        instruction: options.instruction,
        instructions: nil,
        input: options.text
    ) else {
        throw AuKError.invalidInput("AuK-Flash requires --instruction (or --text)")
    }
    _ = try aukResolveGenSeconds(options.genSeconds, hasReferenceAudio: options.referenceAudioPath != nil)
    let thinkerURL = options.thinkerPath.map { URL(fileURLWithPath: $0).standardizedFileURL }
    let engine = try await AuKEngine.load(
        modelDirectory: URL(fileURLWithPath: modelPath).standardizedFileURL,
        thinkerDirectory: thinkerURL,
        bits: options.bits
    )
    let refURL = options.referenceAudioPath.map { URL(fileURLWithPath: $0).standardizedFileURL }
    let (samples, sampleRate) = try engine.generate(
        instruction: instruction,
        referenceAudioURL: refURL,
        genSeconds: options.genSeconds,
        seed: options.seed
    )
    try AudioUtils.writeWavFile(
        samples: samples,
        sampleRate: Double(sampleRate),
        fileURL: URL(fileURLWithPath: options.outputPath)
    )
    let duration = Double(samples.count) / Double(sampleRate)
    fputs(
        "AuK-Flash synthesized \(String(format: "%.2f", duration))s audio -> \(options.outputPath)\n",
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

    if options.transport == "http" {
        startHTTPServer(
            config: HTTPServerConfig(host: options.host, port: options.port),
            handler: { request in
                let path = request.path.split(separator: "?").first.map(String.init) ?? request.path
                if path == "/v1/info" {
                    guard request.method == "GET" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
                    return jsonResponse(status: 200, [
                        "status": "ready",
                        "service": "yuwp-tts",
                        "model": modelName,
                        "backend": "auk-flash",
                        "sample_rate": engine.sampleRate,
                        "load_seconds": loadSeconds,
                        "nfe": 4,
                        "cfg": 0,
                    ])
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
                if path == "/v1/audio/speech/stream" || speechRequest.stream == true {
                    return streamingResponse(status: 200, contentType: "application/x-ndjson; charset=utf-8") { writer in
                        let semaphore = DispatchSemaphore(value: 0)
                        Task {
                            await writeAuKSpeechStream(
                                engine: engine,
                                request: speechRequest,
                                defaultRef: options.referenceAudioPath,
                                defaultSeconds: options.genSeconds,
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
                        defaultRef: options.referenceAudioPath,
                        defaultSeconds: options.genSeconds,
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
        "backend": "auk-flash",
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
        let instruction = aukResolveInstruction(
            instruction: request.instruction,
            instructions: nil,
            input: request.text
        )
        guard let instruction else {
            emitJSON(["event": "error", "id": requestID, "error": "instruction/text is required"])
            continue
        }
        let refPath = request.refAudio ?? options.referenceAudioPath
        let genSeconds: Double?
        do {
            genSeconds = try aukResolveGenSeconds(request.genSeconds ?? options.genSeconds, hasReferenceAudio: refPath != nil)
        } catch {
            emitJSON(["event": "error", "id": requestID, "error": error.localizedDescription])
            continue
        }
        await gate.enter()
        do {
            let (samples, sampleRate) = try engine.generate(
                instruction: instruction,
                referenceAudioURL: refPath.map { URL(fileURLWithPath: $0) },
                genSeconds: genSeconds,
                seed: options.seed
            )
            try AudioUtils.writeWavFile(
                samples: samples,
                sampleRate: Double(sampleRate),
                fileURL: URL(fileURLWithPath: outputPath)
            )
            await gate.leave()
            emitJSON([
                "event": "done",
                "id": requestID,
                "out": outputPath,
                "audioDurationSeconds": Double(samples.count) / Double(sampleRate),
            ])
        } catch {
            await gate.leave()
            emitJSON(["event": "error", "id": requestID, "error": error.localizedDescription])
        }
    }
}

private func aukInstruction(from request: OpenAISpeechRequest) -> String? {
    aukResolveInstruction(
        instruction: request.instruction,
        instructions: request.instructions,
        input: request.input
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
    defaultRef: String?,
    defaultSeconds: Double?,
    gate: AsyncGate
) async -> HTTPResponse {
    let responseFormat = request.responseFormat?.lowercased() ?? "wav"
    guard responseFormat == "wav" else {
        return jsonResponse(status: 400, ["error": "unsupported response_format '\(responseFormat)'; only wav is currently supported"])
    }
    guard let instruction = aukInstruction(from: request) else {
        return jsonResponse(status: 400, ["error": "instruction/input is required"])
    }
    let refPath = request.refAudio ?? defaultRef
    let genSeconds: Double?
    do {
        genSeconds = try aukResolveGenSeconds(request.genSeconds ?? defaultSeconds, hasReferenceAudio: refPath != nil)
    } catch {
        return jsonResponse(status: 400, ["error": error.localizedDescription])
    }
    await gate.enter()
    do {
        let (samples, sampleRate) = try engine.generate(
            instruction: instruction,
            referenceAudioURL: refPath.map { URL(fileURLWithPath: $0) },
            genSeconds: genSeconds,
            seed: nil
        )
        let outputPath = "/tmp/yuwp-tts-http-\(UUID().uuidString).wav"
        try AudioUtils.writeWavFile(
            samples: samples,
            sampleRate: Double(sampleRate),
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
    defaultRef: String?,
    defaultSeconds: Double?,
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

    guard let instruction = aukInstruction(from: request) else {
        emit(["event": "error", "error": "instruction/input is required"])
        return
    }
    let refPath = request.refAudio ?? defaultRef
    let genSeconds: Double?
    do {
        genSeconds = try aukResolveGenSeconds(request.genSeconds ?? defaultSeconds, hasReferenceAudio: refPath != nil)
    } catch {
        emit(["event": "error", "error": error.localizedDescription])
        return
    }
    emit([
        "event": "metadata",
        "format": "pcm_s16le",
        "sample_rate": engine.sampleRate,
        "channels": 1,
        "backend": "auk-flash",
    ])
    await gate.enter()
    do {
        let (samples, sampleRate) = try engine.generate(
            instruction: instruction,
            referenceAudioURL: refPath.map { URL(fileURLWithPath: $0) },
            genSeconds: genSeconds,
            seed: nil
        )
        await gate.leave()
        emit([
            "event": "audio",
            "audio": pcm16Base64(samples: samples),
            "samples": samples.count,
            "sample_rate": sampleRate,
        ])
        emit(["event": "done"])
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
