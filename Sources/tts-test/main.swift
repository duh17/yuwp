import AVFoundation
import Foundation
import MLX
import MLXLMCommon
import NativeTTS

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
    var streamingInterval: Double = 2.0
    var play = false
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
            Usage: tts-test --model <model-dir> [options]

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
              --streaming-interval <sec>   Target streaming chunk interval, default 2.0
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

let options = parseOptions(Array(CommandLine.arguments.dropFirst()))

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
