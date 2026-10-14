import Foundation
@preconcurrency import MLX
import NativeTTS

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
