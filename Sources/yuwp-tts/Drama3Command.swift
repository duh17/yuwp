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

    var references: [Drama3Reference] = []
    if let referenceAudioPath = options.referenceAudioPath {
        guard let transcript = options.referenceText else {
            throw Drama3Error.invalidInput("--ref-audio needs --ref-text with its transcript")
        }
        let (_, audio) = try loadAudioArray(from: URL(fileURLWithPath: referenceAudioPath), sampleRate: engine.sampleRate)
        references.append(Drama3Reference(transcript: transcript, samples: audio.asArray(Float.self)))
    }

    var sampling = Drama3Sampling()
    if let temperature = options.temperature { sampling.temperature = temperature }
    if let topP = options.topP { sampling.topP = topP }
    if let topK = options.topK { sampling.topK = topK }
    if let maxTokens = options.maxTokens { sampling.maxFrames = maxTokens }
    sampling.seed = options.seed

    let synthStart = Date()
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
