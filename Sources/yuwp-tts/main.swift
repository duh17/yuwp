import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXLMCommon
import NativeTTS
import YuwpHTTPServerSupport

struct TTSTestOptions {
    var modelPath: String?
    var text = "Hello from Yuwp TTS."
    var outputPath = "tts-output.wav"
    var voice: String?
    var referenceAudioPath: String?
    var referenceText: String?
    var language: String? = "English"
    var maxTokens: Int?
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    var minP: Float?
    var repetitionPenalty: Float?
    var stream = false
    var streamingInterval: Double = 0.2
    var play = false
    var transport = "stdio"
    var host = "127.0.0.1"
    var port: UInt16 = 7937
}

func parseOptions(_ args: [String]) -> TTSTestOptions {
    var options = TTSTestOptions()
    var i = 0
    while i < args.count {
        let arg = args[i]
        func takeValue() -> String? {
            guard i + 1 < args.count else { return nil }
            i += 1
            return args[i]
        }
        switch arg {
        case "--model": options.modelPath = takeValue()
        case "--text": options.text = takeValue() ?? options.text
        case "--out", "--output": options.outputPath = takeValue() ?? options.outputPath
        case "--voice": options.voice = takeValue()
        case "--ref-audio": options.referenceAudioPath = takeValue()
        case "--ref-text": options.referenceText = takeValue()
        case "--language": options.language = takeValue()
        case "--max-tokens": options.maxTokens = takeValue().flatMap(Int.init)
        case "--temperature": options.temperature = takeValue().flatMap(Float.init)
        case "--top-p": options.topP = takeValue().flatMap(Float.init)
        case "--top-k": options.topK = takeValue().flatMap(Int.init)
        case "--min-p": options.minP = takeValue().flatMap(Float.init)
        case "--repetition-penalty": options.repetitionPenalty = takeValue().flatMap(Float.init)
        case "--stream": options.stream = true
        case "--streaming-interval": options.streamingInterval = takeValue().flatMap(Double.init) ?? options.streamingInterval
        case "--play": options.play = true
        case "--help", "-h":
            print("""
            Usage: yuwp-tts --model <model-dir> [options]
                   yuwp-tts serve --transport http --model <model-dir> [--host 127.0.0.1] [--port 7937]

            Options:
              --text <text>                Text to synthesize
              --out <path>                 Output WAV path (default: tts-output.wav)
              --voice <description>        VoiceDesign description or CustomVoice speaker/style
              --ref-audio <path>           Reference WAV/M4A for cloning
              --ref-text <text>            Transcript of reference audio
              --language <language>        Language, default English
              --max-tokens <n>             Maximum codec tokens to generate
              --temperature <float>        Sampling temperature; 0 uses greedy decoding
              --top-p <float>              Top-p sampling threshold
              --top-k <n>                  Top-k sampling; 0 disables
              --min-p <float>              Min-p sampling threshold
              --repetition-penalty <float> Repetition penalty
              --stream                     Stream decode/write chunks instead of waiting for full decode
              --streaming-interval <sec>   Target streaming chunk interval, default 0.2
              --play                       Play the generated WAV, or play chunks live with --stream
            """)
            exit(0)
        default:
            break
        }
        i += 1
    }
    return options
}

final class StreamingAudioPlayer: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let lock = NSLock()
    private var scheduledChunks = 0
    private var completedChunks = 0

    init(sampleRate: Int) throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "tts-test", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio playback format"])
        }
        self.format = format
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        try engine.start()
        player.play()
    }

    func schedule(samples: [Float]) throws {
        guard !samples.isEmpty else { return }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw NSError(domain: "tts-test", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio playback buffer"])
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard let channel = buffer.floatChannelData?[0] else {
            throw NSError(domain: "tts-test", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to access audio playback buffer"])
        }
        samples.withUnsafeBufferPointer { source in
            if let baseAddress = source.baseAddress {
                channel.update(from: baseAddress, count: samples.count)
            }
        }

        lock.lock()
        scheduledChunks += 1
        lock.unlock()

        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self else { return }
            lock.lock()
            completedChunks += 1
            lock.unlock()
        }
    }

    private func isComplete() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return scheduledChunks > 0 && completedChunks >= scheduledChunks
    }

    func waitUntilComplete() async {
        while true {
            if isComplete() { break }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    func stop() {
        player.stop()
        engine.stop()
    }
}

struct TTSServeRequest: Decodable {
    var id: String?
    var text: String
    var out: String?
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    var minP: Float?
    var repetitionPenalty: Float?
    var maxTokens: Int?
    var streamingInterval: Double?
    var emitChunks: Bool?
}

struct OpenAISpeechRequest: Decodable, Sendable {
    var model: String?
    var input: String
    var voice: String?
    var voiceId: String?
    var responseFormat: String?
    var speed: Double?
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    var minP: Float?
    var repetitionPenalty: Float?
    var maxTokens: Int?
    var stream: Bool?
    var streamingInterval: Double?

    enum CodingKeys: String, CodingKey {
        case model
        case input
        case voice
        case voiceId = "voice_id"
        case responseFormat = "response_format"
        case speed
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case maxTokens = "max_tokens"
        case stream
        case streamingInterval = "streaming_interval"
    }
}

struct VoicePreviewRequest: Decodable, Sendable {
    var input: String?
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    var minP: Float?
    var repetitionPenalty: Float?
    var maxTokens: Int?
    var streamingInterval: Double?

    enum CodingKeys: String, CodingKey {
        case input
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case maxTokens = "max_tokens"
        case streamingInterval = "streaming_interval"
    }
}

func codableJSONResponse<T: Encodable>(status: Int, _ value: T) -> HTTPResponse {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    guard let data = try? encoder.encode(value) else {
        return jsonResponse(status: 500, ["error": "failed to encode response"])
    }
    return HTTPResponse(status: status, contentType: "application/json", body: data)
}

func codableJSONObject<T: Encodable>(_ value: T) -> Any? {
    guard let data = try? JSONEncoder().encode(value) else { return nil }
    return try? JSONSerialization.jsonObject(with: data)
}

actor AsyncGate {
    private var isOccupied = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func enter() async {
        if !isOccupied {
            isOccupied = true
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func leave() {
        if waiters.isEmpty {
            isOccupied = false
        } else {
            let continuation = waiters.removeFirst()
            continuation.resume()
        }
    }
}

final class TTSHTTPState: @unchecked Sendable {
    let model: Qwen3TTSModel
    let modelName: String
    let loadSeconds: TimeInterval
    let refAudio: MLXArray?
    let refText: String?
    let language: String?
    let defaultVoice: String?
    let conditioning: Qwen3TTSModel.Qwen3TTSReferenceConditioning?
    let voiceLibrary: VoiceLibrary
    let defaultGenerationParameters: GenerateParameters
    let defaultStreamingInterval: Double
    private let synthesisGate = AsyncGate()
    private var voiceConditioningCache: [String: Qwen3TTSModel.Qwen3TTSReferenceConditioning] = [:]

    init(
        model: Qwen3TTSModel,
        modelName: String,
        loadSeconds: TimeInterval,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        defaultVoice: String?,
        conditioning: Qwen3TTSModel.Qwen3TTSReferenceConditioning?,
        voiceLibrary: VoiceLibrary,
        defaultGenerationParameters: GenerateParameters,
        defaultStreamingInterval: Double
    ) {
        self.model = model
        self.modelName = modelName
        self.loadSeconds = loadSeconds
        self.refAudio = refAudio
        self.refText = refText
        self.language = language
        self.defaultVoice = defaultVoice
        self.conditioning = conditioning
        self.voiceLibrary = voiceLibrary
        self.defaultGenerationParameters = defaultGenerationParameters
        self.defaultStreamingInterval = defaultStreamingInterval
    }

    func infoResponse() -> HTTPResponse {
        jsonResponse(status: 200, [
            "status": "ready",
            "service": "yuwp-tts",
            "model": modelName,
            "sample_rate": model.sampleRate,
            "has_reference_conditioning": conditioning != nil,
            "load_seconds": loadSeconds,
        ])
    }

    func speechResponse(_ speechRequest: OpenAISpeechRequest) async -> HTTPResponse {
        let responseFormat = speechRequest.responseFormat?.lowercased() ?? "wav"
        guard responseFormat == "wav" else {
            return jsonResponse(status: 400, ["error": "unsupported response_format '\(responseFormat)'; only wav is currently supported"])
        }

        await synthesisGate.enter()

        do {
            let voiceContext = try resolveVoiceContext(for: speechRequest)
            let generationParameters = generationParameters(for: speechRequest, voice: voiceContext.record)
            let audio = if let conditioning = voiceContext.conditioning {
                try await model.generate(
                    text: speechRequest.input,
                    conditioning: conditioning,
                    generationParameters: generationParameters
                )
            } else {
                try await model.generate(
                    text: speechRequest.input,
                    voice: voiceContext.voice,
                    refAudio: voiceContext.refAudio,
                    refText: voiceContext.refText,
                    language: voiceContext.language,
                    generationParameters: generationParameters
                )
            }
            MLX.eval(audio)
            let samples = audio.asArray(Float.self)
            let outputPath = "/tmp/yuwp-tts-http-\(UUID().uuidString).wav"
            try AudioUtils.writeWavFile(
                samples: samples,
                sampleRate: Double(model.sampleRate),
                fileURL: URL(fileURLWithPath: outputPath)
            )
            let data = try Data(contentsOf: URL(fileURLWithPath: outputPath))
            try? FileManager.default.removeItem(atPath: outputPath)
            await synthesisGate.leave()
            return binaryResponse(status: 200, body: data, contentType: "audio/wav")
        } catch {
            await synthesisGate.leave()
            return jsonResponse(status: 500, ["error": error.localizedDescription])
        }
    }

    func streamingSpeechResponse(_ speechRequest: OpenAISpeechRequest) -> HTTPResponse {
        streamingResponse(status: 200, contentType: "application/x-ndjson; charset=utf-8") { writer in
            let semaphore = DispatchSemaphore(value: 0)
            Task {
                await self.writeSpeechStream(speechRequest, to: writer)
                semaphore.signal()
            }
            semaphore.wait()
        }
    }

    private struct VoiceContext {
        var record: VoiceRecord?
        var voice: String?
        var refAudio: MLXArray?
        var refText: String?
        var language: String?
        var conditioning: Qwen3TTSModel.Qwen3TTSReferenceConditioning?
    }

    private func generationParameters(for speechRequest: OpenAISpeechRequest, voice: VoiceRecord? = nil) -> GenerateParameters {
        var generationParameters = defaultGenerationParameters
        if let defaults = voice?.defaults {
            if let maxTokens = defaults.maxTokens { generationParameters.maxTokens = maxTokens }
            if let temperature = defaults.temperature { generationParameters.temperature = temperature }
            if let topP = defaults.topP { generationParameters.topP = topP }
            if let topK = defaults.topK { generationParameters.topK = topK }
            if let minP = defaults.minP { generationParameters.minP = minP }
            if let repetitionPenalty = defaults.repetitionPenalty { generationParameters.repetitionPenalty = repetitionPenalty }
        }
        if let maxTokens = speechRequest.maxTokens { generationParameters.maxTokens = maxTokens }
        if let temperature = speechRequest.temperature { generationParameters.temperature = temperature }
        if let topP = speechRequest.topP { generationParameters.topP = topP }
        if let topK = speechRequest.topK { generationParameters.topK = topK }
        if let minP = speechRequest.minP { generationParameters.minP = minP }
        if let repetitionPenalty = speechRequest.repetitionPenalty { generationParameters.repetitionPenalty = repetitionPenalty }
        return generationParameters
    }

    private func resolveVoiceContext(for speechRequest: OpenAISpeechRequest) throws -> VoiceContext {
        if let explicitVoiceID = speechRequest.voiceId {
            guard let record = try voiceLibrary.get(explicitVoiceID) else { throw VoiceLibraryError.notFound }
            return try resolveVoiceContext(for: record)
        }

        if let voice = speechRequest.voice,
           VoiceLibrary.isValidID(voice),
           let record = try voiceLibrary.get(voice) {
            return try resolveVoiceContext(for: record)
        }

        return VoiceContext(
            record: nil,
            voice: speechRequest.voice ?? defaultVoice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            conditioning: conditioning
        )
    }

    private func resolveVoiceContext(for record: VoiceRecord) throws -> VoiceContext {
        let resolvedVoice = record.prompt ?? record.voice ?? defaultVoice
        let resolvedLanguage = record.language ?? language
        guard let referenceURL = voiceLibrary.referenceAudioURL(for: record) else {
            return VoiceContext(
                record: record,
                voice: resolvedVoice,
                refAudio: nil,
                refText: nil,
                language: resolvedLanguage,
                conditioning: nil
            )
        }

        let (_, audio) = try loadAudioArray(from: referenceURL, sampleRate: model.sampleRate)
        if let referenceText = record.referenceText {
            if let cached = voiceConditioningCache[record.id] {
                return VoiceContext(
                    record: record,
                    voice: resolvedVoice,
                    refAudio: audio,
                    refText: referenceText,
                    language: resolvedLanguage,
                    conditioning: cached
                )
            }
            let prepared = try model.prepareReferenceConditioning(
                refAudio: audio,
                refText: referenceText,
                language: resolvedLanguage
            )
            voiceConditioningCache[record.id] = prepared
            return VoiceContext(
                record: record,
                voice: resolvedVoice,
                refAudio: audio,
                refText: referenceText,
                language: resolvedLanguage,
                conditioning: prepared
            )
        }

        return VoiceContext(
            record: record,
            voice: resolvedVoice,
            refAudio: audio,
            refText: nil,
            language: resolvedLanguage,
            conditioning: nil
        )
    }

    private func writeSpeechStream(_ speechRequest: OpenAISpeechRequest, to writer: HTTPStreamWriter) async {
        let streamingInterval = speechRequest.streamingInterval ?? defaultStreamingInterval
        let synthStart = Date()

        writeJSONLine([
            "event": "metadata",
            "format": "pcm_s16le",
            "sample_rate": model.sampleRate,
            "channels": 1,
            "encoding": "base64",
        ], to: writer)

        await synthesisGate.enter()

        do {
            var firstAudioTime: TimeInterval?
            var chunkCount = 0
            var totalSamples = 0
            var tokenCount = 0
            var generationInfo: AudioGenerationInfo?
            let voiceContext = try resolveVoiceContext(for: speechRequest)
            let generationParameters = generationParameters(for: speechRequest, voice: voiceContext.record)

            let stream = if let conditioning = voiceContext.conditioning {
                model.generateStream(
                    text: speechRequest.input,
                    conditioning: conditioning,
                    generationParameters: generationParameters,
                    streamingInterval: streamingInterval
                )
            } else {
                model.generateStream(
                    text: speechRequest.input,
                    voice: voiceContext.voice,
                    refAudio: voiceContext.refAudio,
                    refText: voiceContext.refText,
                    language: voiceContext.language,
                    generationParameters: generationParameters,
                    streamingInterval: streamingInterval
                )
            }

            for try await event in stream {
                switch event {
                case .token:
                    tokenCount += 1
                case .info(let info):
                    generationInfo = info
                case .audio(let audioChunk):
                    MLX.eval(audioChunk)
                    let samples = audioChunk.asArray(Float.self)
                    guard !samples.isEmpty else { continue }
                    if firstAudioTime == nil {
                        firstAudioTime = Date().timeIntervalSince(synthStart)
                    }
                    chunkCount += 1
                    totalSamples += samples.count
                    writeJSONLine([
                        "event": "audio",
                        "chunk": chunkCount,
                        "samples": samples.count,
                        "seconds": Double(samples.count) / Double(model.sampleRate),
                        "elapsed_seconds": Date().timeIntervalSince(synthStart),
                        "audio": pcm16Base64(samples: samples),
                    ], to: writer)
                }
            }

            writeJSONLine([
                "event": "done",
                "first_audio_seconds": firstAudioTime ?? -1,
                "audio_duration_seconds": Double(totalSamples) / Double(model.sampleRate),
                "wall_seconds": Date().timeIntervalSince(synthStart),
                "chunks": chunkCount,
                "tokens": tokenCount,
                "tokens_per_second": generationInfo?.tokensPerSecond ?? 0,
                "peak_memory_gb": generationInfo?.peakMemoryUsage ?? 0,
            ], to: writer)
            await synthesisGate.leave()
        } catch {
            await synthesisGate.leave()
            writeJSONLine(["event": "error", "error": error.localizedDescription], to: writer)
        }
    }

    private func writeJSONLine(_ object: [String: Any], to writer: HTTPStreamWriter) {
        guard JSONSerialization.isValidJSONObject(object),
              var data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        else { return }
        data.append(0x0A)
        writer.write(data)
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

    func voicesResponse(request: HTTPRequest, pathComponents: [String]) -> HTTPResponse {
        do {
            if pathComponents.count == 2 {
                switch request.method {
                case "GET":
                    return jsonResponse(status: 200, [
                        "object": "list",
                        "data": try voiceLibrary.list().compactMap(codableJSONObject),
                    ])
                case "POST":
                    let create = try JSONDecoder().decode(VoiceCreateRequest.self, from: request.body)
                    let record = try voiceLibrary.create(create)
                    return codableJSONResponse(status: 201, record)
                default:
                    return jsonResponse(status: 405, ["error": "method not allowed"])
                }
            }

            guard pathComponents.count >= 3 else { return jsonResponse(status: 404, ["error": "voice endpoint not found"]) }
            let voiceID = pathComponents[2]

            if pathComponents.count == 4, pathComponents[3] == "preview.wav" {
                guard request.method == "GET" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
                let previewURL = try voiceLibrary.previewURL(for: voiceID)
                guard FileManager.default.fileExists(atPath: previewURL.path) else {
                    return jsonResponse(status: 404, ["error": "preview not found"])
                }
                return binaryResponse(status: 200, body: try Data(contentsOf: previewURL), contentType: "audio/wav")
            }

            guard pathComponents.count == 3 else { return jsonResponse(status: 404, ["error": "voice endpoint not found"]) }
            switch request.method {
            case "GET":
                guard let record = try voiceLibrary.get(voiceID) else { return jsonResponse(status: 404, ["error": "voice not found"]) }
                return codableJSONResponse(status: 200, record)
            case "PATCH":
                let update = try JSONDecoder().decode(VoiceUpdateRequest.self, from: request.body)
                let record = try voiceLibrary.update(id: voiceID, request: update)
                voiceConditioningCache.removeValue(forKey: voiceID)
                return codableJSONResponse(status: 200, record)
            case "DELETE":
                try voiceLibrary.delete(id: voiceID)
                voiceConditioningCache.removeValue(forKey: voiceID)
                return jsonResponse(status: 200, ["deleted": true, "id": voiceID])
            default:
                return jsonResponse(status: 405, ["error": "method not allowed"])
            }
        } catch VoiceLibraryError.notFound {
            return jsonResponse(status: 404, ["error": "voice not found"])
        } catch VoiceLibraryError.conflict(let message) {
            return jsonResponse(status: 409, ["error": message])
        } catch {
            return jsonResponse(status: 400, ["error": error.localizedDescription])
        }
    }

    func previewResponse(voiceID: String, body: Data) async -> HTTPResponse {
        do {
            let request = if body.isEmpty {
                VoicePreviewRequest(
                    input: nil,
                    temperature: nil,
                    topP: nil,
                    topK: nil,
                    minP: nil,
                    repetitionPenalty: nil,
                    maxTokens: nil,
                    streamingInterval: nil
                )
            } else {
                try JSONDecoder().decode(VoicePreviewRequest.self, from: body)
            }
            let speechRequest = OpenAISpeechRequest(
                model: "yuwp-tts",
                input: request.input ?? "Hello. This is a short preview of this Yuwp voice.",
                voice: nil,
                voiceId: voiceID,
                responseFormat: "wav",
                speed: nil,
                temperature: request.temperature,
                topP: request.topP,
                topK: request.topK,
                minP: request.minP,
                repetitionPenalty: request.repetitionPenalty,
                maxTokens: request.maxTokens,
                stream: false,
                streamingInterval: request.streamingInterval
            )
            let response = await speechResponse(speechRequest)
            guard response.status == 200 else { return response }
            let previewURL = try voiceLibrary.previewURL(for: voiceID)
            try response.body.write(to: previewURL, options: [.atomic])
            _ = try voiceLibrary.markPreview(id: voiceID)
            return response
        } catch {
            return jsonResponse(status: 400, ["error": error.localizedDescription])
        }
    }
}

func emitJSON(_ object: [String: Any]) {
    guard JSONSerialization.isValidJSONObject(object),
          let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    else {
        return
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
}

func parseServeOptions(_ args: [String]) -> TTSTestOptions {
    var options = TTSTestOptions()
    var i = 0
    while i < args.count {
        let arg = args[i]
        func takeValue() -> String? {
            guard i + 1 < args.count else { return nil }
            i += 1
            return args[i]
        }
        switch arg {
        case "--model": options.modelPath = takeValue()
        case "--voice": options.voice = takeValue()
        case "--ref-audio": options.referenceAudioPath = takeValue()
        case "--ref-text": options.referenceText = takeValue()
        case "--language": options.language = takeValue()
        case "--temperature": options.temperature = takeValue().flatMap(Float.init)
        case "--top-p": options.topP = takeValue().flatMap(Float.init)
        case "--top-k": options.topK = takeValue().flatMap(Int.init)
        case "--min-p": options.minP = takeValue().flatMap(Float.init)
        case "--repetition-penalty": options.repetitionPenalty = takeValue().flatMap(Float.init)
        case "--max-tokens": options.maxTokens = takeValue().flatMap(Int.init)
        case "--streaming-interval": options.streamingInterval = takeValue().flatMap(Double.init) ?? options.streamingInterval
        case "--transport": options.transport = takeValue() ?? options.transport
        case "--host": options.host = takeValue() ?? options.host
        case "--port": options.port = takeValue().flatMap(UInt16.init) ?? options.port
        default: break
        }
        i += 1
    }
    return options
}

func runServe(_ args: [String]) async throws {
    let options = parseServeOptions(args)
    guard let modelPath = options.modelPath else {
        throw AudioGenerationError.invalidInput("serve requires --model <model-dir>")
    }

    let loadStart = Date()
    let model = try await Qwen3TTSModel.fromModelDirectory(URL(fileURLWithPath: modelPath).standardizedFileURL)

    let refAudio: MLXArray?
    if let referenceAudioPath = options.referenceAudioPath {
        let (_, audio) = try loadAudioArray(
            from: URL(fileURLWithPath: referenceAudioPath),
            sampleRate: model.sampleRate
        )
        refAudio = audio
    } else {
        refAudio = nil
    }

    let conditioning = if let refAudio, let refText = options.referenceText {
        try model.prepareReferenceConditioning(
            refAudio: refAudio,
            refText: refText,
            language: options.language
        )
    } else {
        nil as Qwen3TTSModel.Qwen3TTSReferenceConditioning?
    }

    var baseGenerationParameters = model.defaultGenerationParameters
    if let maxTokens = options.maxTokens { baseGenerationParameters.maxTokens = maxTokens }
    if let temperature = options.temperature { baseGenerationParameters.temperature = temperature }
    if let topP = options.topP { baseGenerationParameters.topP = topP }
    if let topK = options.topK { baseGenerationParameters.topK = topK }
    if let minP = options.minP { baseGenerationParameters.minP = minP }
    if let repetitionPenalty = options.repetitionPenalty { baseGenerationParameters.repetitionPenalty = repetitionPenalty }

    let defaultGenerationParameters = baseGenerationParameters

    if options.transport == "http" {
        let voiceLibrary = try VoiceLibrary()
        let state = TTSHTTPState(
            model: model,
            modelName: URL(fileURLWithPath: modelPath).lastPathComponent,
            loadSeconds: Date().timeIntervalSince(loadStart),
            refAudio: refAudio,
            refText: options.referenceText,
            language: options.language,
            defaultVoice: options.voice,
            conditioning: conditioning,
            voiceLibrary: voiceLibrary,
            defaultGenerationParameters: defaultGenerationParameters,
            defaultStreamingInterval: options.streamingInterval
        )
        startHTTPServer(
            config: HTTPServerConfig(host: options.host, port: options.port),
            handler: { request in
                let path = request.path.split(separator: "?").first.map(String.init) ?? request.path

                if path == "/v1/info" {
                    guard request.method == "GET" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
                    return state.infoResponse()
                }

                let pathComponents = path.split(separator: "/").map(String.init)
                if pathComponents.count == 4,
                   pathComponents[0] == "v1",
                   pathComponents[1] == "voices",
                   pathComponents[3] == "preview" {
                    guard request.method == "POST" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
                    let semaphore = DispatchSemaphore(value: 0)
                    final class PreviewBox: @unchecked Sendable { var response: HTTPResponse? }
                    let box = PreviewBox()
                    Task {
                        box.response = await state.previewResponse(voiceID: pathComponents[2], body: request.body)
                        semaphore.signal()
                    }
                    semaphore.wait()
                    return box.response ?? jsonResponse(status: 500, ["error": "preview generation failed"])
                }
                if pathComponents.count >= 2,
                   pathComponents[0] == "v1",
                   pathComponents[1] == "voices" {
                    return state.voicesResponse(request: request, pathComponents: pathComponents)
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
                    return state.streamingSpeechResponse(speechRequest)
                }

                let semaphore = DispatchSemaphore(value: 0)
                final class Box: @unchecked Sendable { var response: HTTPResponse? }
                let box = Box()
                Task {
                    box.response = await state.speechResponse(speechRequest)
                    semaphore.signal()
                }
                semaphore.wait()
                return box.response ?? jsonResponse(status: 500, ["error": "speech synthesis failed"])
            }
        )
    }

    emitJSON([
        "event": "ready",
        "sampleRate": model.sampleRate,
        "loadSeconds": Date().timeIntervalSince(loadStart),
        "hasReferenceConditioning": conditioning != nil,
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
        var generationParameters = baseGenerationParameters
        if let maxTokens = request.maxTokens { generationParameters.maxTokens = maxTokens }
        if let temperature = request.temperature { generationParameters.temperature = temperature }
        if let topP = request.topP { generationParameters.topP = topP }
        if let topK = request.topK { generationParameters.topK = topK }
        if let minP = request.minP { generationParameters.minP = minP }
        if let repetitionPenalty = request.repetitionPenalty { generationParameters.repetitionPenalty = repetitionPenalty }
        let streamingInterval = request.streamingInterval ?? options.streamingInterval
        let emitChunkEvents = request.emitChunks ?? true

        let synthStart = Date()
        do {
            let writer = try StreamingWAVWriter(
                url: URL(fileURLWithPath: outputPath),
                sampleRate: Double(model.sampleRate)
            )
            var firstAudioTime: TimeInterval?
            var chunkCount = 0
            var totalSamples = 0
            var tokenCount = 0
            var generationInfo: AudioGenerationInfo?

            let stream = if let conditioning {
                model.generateStream(
                    text: request.text,
                    conditioning: conditioning,
                    generationParameters: generationParameters,
                    streamingInterval: streamingInterval
                )
            } else {
                model.generateStream(
                    text: request.text,
                    voice: options.voice,
                    refAudio: refAudio,
                    refText: options.referenceText,
                    language: options.language,
                    generationParameters: generationParameters,
                    streamingInterval: streamingInterval
                )
            }

            for try await event in stream {
                switch event {
                case .token:
                    tokenCount += 1
                case .info(let info):
                    generationInfo = info
                case .audio(let audioChunk):
                    MLX.eval(audioChunk)
                    let samples = audioChunk.asArray(Float.self)
                    guard !samples.isEmpty else { continue }
                    if firstAudioTime == nil {
                        firstAudioTime = Date().timeIntervalSince(synthStart)
                    }
                    chunkCount += 1
                    totalSamples += samples.count
                    try writer.writeChunk(samples)
                    if emitChunkEvents {
                        emitJSON([
                            "event": "chunk",
                            "id": requestID,
                            "chunk": chunkCount,
                            "samples": samples.count,
                            "seconds": Double(samples.count) / Double(model.sampleRate),
                            "elapsedSeconds": Date().timeIntervalSince(synthStart),
                        ])
                    }
                }
            }
            _ = writer.finalize()

            let totalTime = Date().timeIntervalSince(synthStart)
            let audioDuration = Double(totalSamples) / Double(model.sampleRate)
            emitJSON([
                "event": "done",
                "id": requestID,
                "out": outputPath,
                "firstAudioSeconds": firstAudioTime ?? -1,
                "audioDurationSeconds": audioDuration,
                "wallSeconds": totalTime,
                "chunks": chunkCount,
                "tokens": tokenCount,
                "tokensPerSecond": generationInfo?.tokensPerSecond ?? 0,
                "peakMemoryGB": generationInfo?.peakMemoryUsage ?? 0,
            ])
        } catch {
            emitJSON(["event": "error", "id": requestID, "error": error.localizedDescription])
        }
    }
}

let rawArgs = Array(CommandLine.arguments.dropFirst())
if rawArgs.first == "serve" {
    do {
        try await runServe(Array(rawArgs.dropFirst()))
        exit(0)
    } catch {
        fputs("Error: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
}

let options = parseOptions(rawArgs)

guard let modelPath = options.modelPath else {
    fputs("Error: --model <model-dir> is required\n", stderr)
    exit(2)
}

do {
    let start = Date()
    let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL
    let model = try await Qwen3TTSModel.fromModelDirectory(modelURL)
    let loadTime = Date().timeIntervalSince(start)

    let refAudio: MLXArray?
    if let referenceAudioPath = options.referenceAudioPath {
        let (_, audio) = try loadAudioArray(
            from: URL(fileURLWithPath: referenceAudioPath),
            sampleRate: model.sampleRate
        )
        refAudio = audio
    } else {
        refAudio = nil
    }

    var generationParameters = model.defaultGenerationParameters
    if let maxTokens = options.maxTokens { generationParameters.maxTokens = maxTokens }
    if let temperature = options.temperature { generationParameters.temperature = temperature }
    if let topP = options.topP { generationParameters.topP = topP }
    if let topK = options.topK { generationParameters.topK = topK }
    if let minP = options.minP { generationParameters.minP = minP }
    if let repetitionPenalty = options.repetitionPenalty { generationParameters.repetitionPenalty = repetitionPenalty }

    let synthStart = Date()
    if options.stream {
        let outputURL = URL(fileURLWithPath: options.outputPath)
        let writer = try StreamingWAVWriter(url: outputURL, sampleRate: Double(model.sampleRate))
        let player = options.play ? try StreamingAudioPlayer(sampleRate: model.sampleRate) : nil
        var firstAudioTime: TimeInterval?
        var chunkCount = 0
        var totalSamples = 0
        var tokenCount = 0
        var generationInfo: AudioGenerationInfo?

        let stream = model.generateStream(
            text: options.text,
            voice: options.voice,
            refAudio: refAudio,
            refText: options.referenceText,
            language: options.language,
            generationParameters: generationParameters,
            streamingInterval: options.streamingInterval
        )

        for try await event in stream {
            switch event {
            case .token:
                tokenCount += 1
            case .info(let info):
                generationInfo = info
            case .audio(let audioChunk):
                MLX.eval(audioChunk)
                let samples = audioChunk.asArray(Float.self)
                guard !samples.isEmpty else { continue }
                if firstAudioTime == nil {
                    firstAudioTime = Date().timeIntervalSince(synthStart)
                }
                chunkCount += 1
                totalSamples += samples.count
                try writer.writeChunk(samples)
                try player?.schedule(samples: samples)
                let chunkDuration = Double(samples.count) / Double(model.sampleRate)
                fputs("stream chunk \(chunkCount): \(String(format: "%.2f", chunkDuration))s audio at +\(String(format: "%.2f", Date().timeIntervalSince(synthStart)))s\n", stderr)
            }
        }
        _ = writer.finalize()
        if let player {
            await player.waitUntilComplete()
            player.stop()
        }

        let synthTime = Date().timeIntervalSince(synthStart)
        let duration = Double(totalSamples) / Double(model.sampleRate)
        let firstAudio = firstAudioTime.map { String(format: "%.2f", $0) } ?? "n/a"
        fputs("Loaded in \(String(format: "%.2f", loadTime))s; streamed \(String(format: "%.2f", duration))s audio in \(String(format: "%.2f", synthTime))s; first audio +\(firstAudio)s; chunks=\(chunkCount); tokens=\(tokenCount) -> \(options.outputPath)\n", stderr)
        if let generationInfo {
            fputs(generationInfo.summary + "\n", stderr)
        }
    } else {
        let audio = try await model.generate(
            text: options.text,
            voice: options.voice,
            refAudio: refAudio,
            refText: options.referenceText,
            language: options.language,
            generationParameters: generationParameters
        )
        MLX.eval(audio)
        let synthTime = Date().timeIntervalSince(synthStart)
        let samples = audio.asArray(Float.self)
        try AudioUtils.writeWavFile(
            samples: samples,
            sampleRate: Double(model.sampleRate),
            fileURL: URL(fileURLWithPath: options.outputPath)
        )
        let duration = Double(samples.count) / Double(model.sampleRate)
        fputs("Loaded in \(String(format: "%.2f", loadTime))s; synthesized \(String(format: "%.2f", duration))s audio in \(String(format: "%.2f", synthTime))s -> \(options.outputPath)\n", stderr)

        if options.play {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
            proc.arguments = [options.outputPath]
            try proc.run()
            proc.waitUntilExit()
        }
    }
} catch {
    fputs("Error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
