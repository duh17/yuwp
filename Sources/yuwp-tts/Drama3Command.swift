import Foundation
@preconcurrency import MLX
import NativeTTS
import YuwpHTTPServerSupport

func runDrama3Convert(_ args: [String]) throws {
    var source: String?
    var output: String?
    var bits: Int?
    var i = 0
    while i < args.count {
        func takeValue() -> String? {
            guard i + 1 < args.count else { return nil }
            i += 1
            return args[i]
        }
        switch args[i] {
        case "--src", "--source": source = takeValue()
        case "--out", "--output", "--dst": output = takeValue()
        case "--bits": bits = takeValue().flatMap(Int.init)
        case "--help", "-h":
            print("""
            Usage: yuwp-tts convert-drama3 --src <fishaudio/drama-3 dir> --out <mlx-dir> [--bits 8|4]

            Converts the official Fish Audio Drama 3 checkpoint (block-FP8 MoE experts, PyTorch codec.pth)
            to MLX safetensors. Without --bits the backbone and tower stay bf16 (~70 GB); --bits 8 or 4
            quantizes linear and embedding weights (MoE routers stay 8-bit). Runtime inference is Swift/MLX.
            """)
            return
        default:
            break
        }
        i += 1
    }
    guard let source, let output else {
        throw Drama3Error.invalidInput("convert-drama3 requires --src and --out")
    }
    let start = Date()
    let result = try Drama3Convert.convertDirectory(
        source: URL(fileURLWithPath: source).standardizedFileURL,
        output: URL(fileURLWithPath: output).standardizedFileURL,
        bits: bits
    )
    fputs(
        "Converted Drama 3 to \(result.outputDirectory.path) (tensors=\(result.tensorCount) codec=\(result.codecTensorCount)) in \(String(format: "%.0f", Date().timeIntervalSince(start)))s\n",
        stderr
    )
}

func runDrama3Generate(options: TTSTestOptions) async throws {
    guard let modelPath = options.modelPath else {
        throw Drama3Error.invalidInput("--model <drama3-mlx-dir> is required")
    }
    let loadStart = Date()
    let engine = try await Drama3Engine.load(directory: URL(fileURLWithPath: modelPath).standardizedFileURL)
    let loadSeconds = Date().timeIntervalSince(loadStart)

    guard options.referenceAudioPaths.count == options.referenceTexts.count else {
        throw Drama3Error.invalidInput("each --ref-audio needs a matching --ref-text transcript")
    }
    // Reference i voices <|speaker:i|> unless its transcript already starts with a speaker tag.
    var references: [Drama3Reference] = []
    for (index, (path, transcript)) in zip(options.referenceAudioPaths, options.referenceTexts).enumerated() {
        let (_, audio) = try loadAudioArray(from: URL(fileURLWithPath: path), sampleRate: engine.sampleRate)
        let trimmed = transcript.drop(while: { $0.isWhitespace })
        let tagged = trimmed.hasPrefix("<|speaker:") ? transcript : "<|speaker:\(index)|> \(transcript)"
        references.append(Drama3Reference(transcript: tagged, samples: audio.asArray(Float.self)))
    }

    var sampling = Drama3Sampling()
    if let temperature = options.temperature { sampling.temperature = temperature }
    if let topP = options.topP { sampling.topP = topP }
    if let topK = options.topK { sampling.topK = topK }
    if let maxTokens = options.maxTokens { sampling.maxFrames = maxTokens }
    sampling.seed = options.seed

    let synthStart = Date()
    if options.stream {
        let outputURL = URL(fileURLWithPath: options.outputPath)
        let writer = try StreamingWAVWriter(url: outputURL, sampleRate: Double(engine.sampleRate))
        let player = options.play ? try StreamingAudioPlayer(sampleRate: engine.sampleRate) : nil
        var chunkCount = 0
        var writeError: Error?
        var streamOptions = Drama3StreamOptions()
        streamOptions.intervalSeconds = options.streamingInterval
        let result = try engine.synthesize(
            text: options.text,
            references: references,
            sampling: sampling,
            stream: streamOptions,
            onAudio: { samples in
                do {
                    try writer.writeChunk(samples)
                    try player?.schedule(samples: samples)
                } catch {
                    writeError = error
                    return
                }
                chunkCount += 1
                let chunkDuration = Double(samples.count) / Double(engine.sampleRate)
                fputs(
                    "stream chunk \(chunkCount): \(String(format: "%.2f", chunkDuration))s audio at +\(String(format: "%.2f", Date().timeIntervalSince(synthStart)))s\n",
                    stderr
                )
            }
        )
        if let writeError { throw writeError }
        _ = writer.finalize()
        if let player {
            await player.waitUntilComplete()
            player.stop()
        }
        let synthSeconds = Date().timeIntervalSince(synthStart)
        let duration = Double(result.samples.count) / Double(engine.sampleRate)
        let framesPerSecond = Double(result.frameCount) / max(result.generationSeconds, 1e-6)
        let firstAudio = result.firstAudioSeconds.map { String(format: "%.2f", $0) } ?? "n/a"
        fputs(
            "\(engine.modelName): loaded in \(String(format: "%.1f", loadSeconds))s; \(result.frameCount) frames (\(String(format: "%.1f", framesPerSecond)) frames/s, prompt \(result.promptRowCount) rows) -> \(String(format: "%.2f", duration))s audio in \(String(format: "%.1f", synthSeconds))s; first audio +\(firstAudio)s; chunks=\(chunkCount)\(result.finishedWithEndOfSpeech ? "" : " [hit --max-tokens]") -> \(options.outputPath)\n",
            stderr
        )
        return
    }

    let result = try engine.synthesize(text: options.text, references: references, sampling: sampling)
    let synthSeconds = Date().timeIntervalSince(synthStart)
    try AudioUtils.writeWavFile(
        samples: result.samples,
        sampleRate: Double(engine.sampleRate),
        fileURL: URL(fileURLWithPath: options.outputPath)
    )
    let duration = Double(result.samples.count) / Double(engine.sampleRate)
    let framesPerSecond = Double(result.frameCount) / max(result.generationSeconds, 1e-6)
    fputs(
        "\(engine.modelName): loaded in \(String(format: "%.1f", loadSeconds))s; \(result.frameCount) frames (\(String(format: "%.1f", framesPerSecond)) frames/s, prompt \(result.promptRowCount) rows) -> \(String(format: "%.2f", duration))s audio in \(String(format: "%.1f", synthSeconds))s\(result.finishedWithEndOfSpeech ? "" : " [hit --max-tokens]") -> \(options.outputPath)\n",
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

// MARK: - serve

/// HTTP server for Drama 3. Voices are saved Yuwp voices (the same `/v1/voices` library the Qwen
/// server writes) that carry reference audio and its transcript; voice i of a request speaks
/// `<|speaker:i|>`. Long scripts are generated chunk by chunk with the same references.
func runDrama3Serve(options: TTSTestOptions) async throws {
    guard let modelPath = options.modelPath else {
        throw Drama3Error.invalidInput("serve requires --model <drama3-mlx-dir>")
    }
    guard options.transport == "http" else {
        throw Drama3Error.invalidInput("Drama 3 serve supports --transport http only")
    }
    let loadStart = Date()
    let engine = try await Drama3Engine.load(directory: URL(fileURLWithPath: modelPath).standardizedFileURL)
    var sampling = Drama3Sampling()
    if let temperature = options.temperature { sampling.temperature = temperature }
    if let topP = options.topP { sampling.topP = topP }
    if let topK = options.topK { sampling.topK = topK }
    if let maxTokens = options.maxTokens { sampling.maxFrames = maxTokens }
    let state = Drama3HTTPState(
        engine: engine,
        modelName: URL(fileURLWithPath: modelPath).lastPathComponent,
        loadSeconds: Date().timeIntervalSince(loadStart),
        voiceLibrary: try VoiceLibrary(),
        defaultVoice: options.voice,
        defaultSampling: sampling,
        defaultMaxFramesIsExplicit: options.maxTokens != nil,
        defaultStreamingInterval: options.streamingInterval
    )
    fputs("\(engine.modelName) ready on \(options.host):\(options.port) (loaded in \(String(format: "%.1f", state.loadSeconds))s)\n", stderr)
    startHTTPServer(
        config: HTTPServerConfig(host: options.host, port: options.port),
        handler: { request in state.handle(request) }
    )
}

final class Drama3HTTPState: @unchecked Sendable {
    let engine: Drama3Engine
    let modelName: String
    let loadSeconds: TimeInterval
    let voiceLibrary: VoiceLibrary
    let defaultVoice: String?
    let defaultSampling: Drama3Sampling
    /// `--max-tokens` was given, so per-script frame budgets do not apply.
    let defaultMaxFramesIsExplicit: Bool
    let defaultStreamingInterval: Double
    private let gate = AsyncGate()
    private let cacheLock = NSLock()
    /// Encoded reference clips by voice ID, keyed to the record's `updatedAt` because another
    /// process (the Qwen server) can rewrite the shared library.
    private var referenceCodes: [String: (updatedAt: String, codes: MLXArray)] = [:]

    init(
        engine: Drama3Engine,
        modelName: String,
        loadSeconds: TimeInterval,
        voiceLibrary: VoiceLibrary,
        defaultVoice: String?,
        defaultSampling: Drama3Sampling,
        defaultMaxFramesIsExplicit: Bool,
        defaultStreamingInterval: Double
    ) {
        self.engine = engine
        self.modelName = modelName
        self.loadSeconds = loadSeconds
        self.voiceLibrary = voiceLibrary
        self.defaultVoice = defaultVoice
        self.defaultSampling = defaultSampling
        self.defaultMaxFramesIsExplicit = defaultMaxFramesIsExplicit
        self.defaultStreamingInterval = defaultStreamingInterval
    }

    func handle(_ request: HTTPRequest) -> HTTPResponse {
        let path = request.path.split(separator: "?").first.map(String.init) ?? request.path
        if path == "/v1/info" {
            guard request.method == "GET" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
            var info: [String: Any] = [
                "status": "ready",
                "service": "yuwp-tts",
                "backend": "drama3",
                "model": modelName,
                "sample_rate": engine.sampleRate,
                "load_seconds": loadSeconds,
                "max_speakers": 16,
            ]
            if let defaultVoice { info["default_voice"] = defaultVoice }
            return jsonResponse(status: 200, info)
        }
        let pathComponents = path.split(separator: "/").map(String.init)
        if pathComponents.count >= 2, pathComponents[0] == "v1", pathComponents[1] == "voices" {
            return voiceLibraryResponse(voiceLibrary, request: request, pathComponents: pathComponents) { voiceID in
                cacheLock.lock()
                referenceCodes.removeValue(forKey: voiceID)
                cacheLock.unlock()
            }
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
                    await self.writeSpeechStream(speechRequest, to: writer)
                    semaphore.signal()
                }
                semaphore.wait()
            }
        }
        let responseFormat = speechRequest.responseFormat?.lowercased() ?? "wav"
        guard responseFormat == "wav" else {
            return jsonResponse(status: 400, ["error": "unsupported response_format '\(responseFormat)'; only wav is currently supported"])
        }
        let semaphore = DispatchSemaphore(value: 0)
        final class Box: @unchecked Sendable { var response: HTTPResponse? }
        let box = Box()
        Task {
            box.response = await self.speechResponse(speechRequest)
            semaphore.signal()
        }
        semaphore.wait()
        return box.response ?? jsonResponse(status: 500, ["error": "speech synthesis failed"])
    }

    private struct Job {
        var chunks: [String]
        var references: [Drama3EncodedReference]
        var sampling: Drama3Sampling
        var maxFramesIsExplicit: Bool

        /// Without an explicit cap, each chunk gets a length-based budget so a missed
        /// end-of-speech stops within seconds instead of running to `maxFrames`.
        func sampling(for chunk: String) -> Drama3Sampling {
            guard !maxFramesIsExplicit else { return sampling }
            var capped = sampling
            capped.maxFrames = min(sampling.maxFrames, Drama3Sampling.frameBudget(forScript: chunk))
            return capped
        }
    }

    /// Runs inside `gate`: reference encoding uses the GPU.
    private func prepare(_ request: OpenAISpeechRequest) throws -> Job {
        let script = request.input?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !script.isEmpty else { throw Drama3Error.invalidInput("input must not be empty") }
        var sampling = defaultSampling
        if let temperature = request.temperature { sampling.temperature = temperature }
        if let topP = request.topP { sampling.topP = topP }
        if let topK = request.topK { sampling.topK = topK }
        if let maxTokens = request.maxTokens { sampling.maxFrames = maxTokens }
        let voiceIDs = request.voices ?? [request.voiceId ?? request.voice ?? defaultVoice].compactMap { $0 }
        guard voiceIDs.count <= 16 else { throw Drama3Error.invalidInput("at most 16 voices (speakers 0...15)") }
        let references = try voiceIDs.enumerated().map { index, voiceID in
            try reference(voiceID: voiceID, speaker: index)
        }
        return Job(
            chunks: Drama3ScriptChunker.chunk(script),
            references: references,
            sampling: sampling,
            maxFramesIsExplicit: defaultMaxFramesIsExplicit || request.maxTokens != nil
        )
    }

    private func reference(voiceID: String, speaker: Int) throws -> Drama3EncodedReference {
        guard VoiceLibrary.isValidID(voiceID), let record = try voiceLibrary.get(voiceID) else {
            throw Drama3Error.invalidInput("unknown voice '\(voiceID)'; Drama 3 speaks saved Yuwp voices that have reference audio")
        }
        guard let url = voiceLibrary.referenceAudioURL(for: record),
              FileManager.default.fileExists(atPath: url.path),
              let transcript = record.referenceText?.trimmingCharacters(in: .whitespacesAndNewlines),
              !transcript.isEmpty
        else {
            throw Drama3Error.invalidInput("voice '\(voiceID)' has no reference clip and transcript to clone")
        }
        cacheLock.lock()
        let cached = referenceCodes[voiceID]
        cacheLock.unlock()
        let codes: MLXArray
        if let cached, cached.updatedAt == record.updatedAt {
            codes = cached.codes
        } else {
            let (_, audio) = try loadAudioArray(from: url, sampleRate: engine.sampleRate)
            codes = engine.encodeReference(Drama3Reference(transcript: transcript, samples: audio.asArray(Float.self)))
            cacheLock.lock()
            referenceCodes[voiceID] = (record.updatedAt, codes)
            cacheLock.unlock()
        }
        let tagged = transcript.hasPrefix("<|speaker:") ? transcript : "<|speaker:\(speaker)|> \(transcript)"
        return Drama3EncodedReference(transcript: tagged, codes: codes)
    }

    private func speechResponse(_ request: OpenAISpeechRequest) async -> HTTPResponse {
        await gate.enter()
        do {
            let job = try prepare(request)
            var samples: [Float] = []
            for chunk in job.chunks {
                let result = try engine.synthesize(text: chunk, encodedReferences: job.references, sampling: job.sampling(for: chunk))
                samples.append(contentsOf: result.samples)
            }
            let outputPath = "/tmp/yuwp-tts-drama3-\(UUID().uuidString).wav"
            try AudioUtils.writeWavFile(samples: samples, sampleRate: Double(engine.sampleRate), fileURL: URL(fileURLWithPath: outputPath))
            let data = try Data(contentsOf: URL(fileURLWithPath: outputPath))
            try? FileManager.default.removeItem(atPath: outputPath)
            await gate.leave()
            return binaryResponse(status: 200, body: data, contentType: "audio/wav")
        } catch {
            await gate.leave()
            return jsonResponse(status: Self.httpStatus(for: error), ["error": error.localizedDescription])
        }
    }

    private func writeSpeechStream(_ request: OpenAISpeechRequest, to writer: HTTPStreamWriter) async {
        func emit(_ object: [String: Any]) {
            guard JSONSerialization.isValidJSONObject(object),
                  var data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            else { return }
            data.append(0x0A)
            writer.write(data)
        }
        let start = Date()
        emit([
            "event": "metadata",
            "format": "pcm_s16le",
            "sample_rate": engine.sampleRate,
            "channels": 1,
            "encoding": "base64",
            "backend": "drama3",
        ])
        await gate.enter()
        do {
            let job = try prepare(request)
            var streamOptions = Drama3StreamOptions()
            streamOptions.intervalSeconds = request.streamingInterval ?? defaultStreamingInterval
            var firstAudio: TimeInterval?
            var chunkCount = 0
            var totalSamples = 0
            var frames = 0
            var hitMaxFrames = false
            let sampleRate = Double(engine.sampleRate)
            for text in job.chunks {
                let result = try engine.synthesize(
                    text: text,
                    encodedReferences: job.references,
                    sampling: job.sampling(for: text),
                    stream: streamOptions,
                    onAudio: { samples in
                        if firstAudio == nil { firstAudio = Date().timeIntervalSince(start) }
                        chunkCount += 1
                        totalSamples += samples.count
                        emit([
                            "event": "audio",
                            "chunk": chunkCount,
                            "samples": samples.count,
                            "seconds": Double(samples.count) / sampleRate,
                            "elapsed_seconds": Date().timeIntervalSince(start),
                            "audio": pcm16Base64(samples: samples),
                        ])
                    }
                )
                frames += result.frameCount
                if !result.finishedWithEndOfSpeech { hitMaxFrames = true }
            }
            await gate.leave()
            emit([
                "event": "done",
                "first_audio_seconds": firstAudio ?? -1,
                "audio_duration_seconds": Double(totalSamples) / sampleRate,
                "wall_seconds": Date().timeIntervalSince(start),
                "chunks": chunkCount,
                "frames": frames,
                "script_chunks": job.chunks.count,
                "hit_max_frames": hitMaxFrames,
            ])
        } catch {
            await gate.leave()
            emit(["event": "error", "error": error.localizedDescription])
        }
    }

    static func httpStatus(for error: Error) -> Int {
        if case Drama3Error.invalidInput = error { return 400 }
        if error is VoiceLibraryError { return 400 }
        return 500
    }
}
