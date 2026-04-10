// asr-bench — Load-once benchmark for NativeASR batch transcription
// Usage: asr-bench <model-dir> <wav-dir> [--iterations N] [--goldens <path>]
//
// Loads the model once, warms up Metal shaders, then runs N iterations
// of transcribing all WAV files. Reports METRIC lines for autoresearch.

import Foundation
import MLX
import NativeASR

func printUsage() {
    fputs("""
    Usage: asr-bench <model-dir> <wav-dir> [options]

    Options:
      --iterations <n>    Number of benchmark iterations (default: 3)
      --device <cpu|gpu>  MLX device to use (default: gpu)
      --goldens <path>    Golden transcripts JSON for correctness check
      --json-output <path> Write full transcripts as JSON

    """, stderr)
}

// Normalization matching the golden files
func normalize(_ text: String) -> String {
    var t = text.lowercased()
    // Strip ASCII punctuation (keep CJK punctuation)
    t = t.unicodeScalars.map { scalar -> String in
        let v = scalar.value
        if v < 128 {
            let c = Character(scalar)
            if c.isPunctuation || "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".contains(c) {
                return ""
            }
        }
        return String(scalar)
    }.joined()
    // Collapse whitespace
    t = t.split(separator: " ").joined(separator: " ")
    return t.trimmingCharacters(in: .whitespaces)
}

struct GoldenCase: Decodable {
    let id: String
    let normalized_text: String
}

struct GoldenFile: Decodable {
    let cases: [GoldenCase]
}

func runBench() throws {
    var args = Array(CommandLine.arguments.dropFirst())

    guard args.count >= 2 else { printUsage(); exit(1) }
    let modelDir = args.removeFirst()
    let wavDir = args.removeFirst()
    var iterations = 3
    var device: Device = .gpu
    var goldensPath: String? = nil
    var jsonOutputPath: String? = nil

    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--iterations":
            guard !args.isEmpty, let n = Int(args.removeFirst()) else {
                fputs("--iterations requires an integer\n", stderr); exit(1)
            }
            iterations = n
        case "--device":
            guard !args.isEmpty else { fputs("--device requires cpu or gpu\n", stderr); exit(1) }
            let value = args.removeFirst().lowercased()
            switch value {
            case "cpu": device = .cpu
            case "gpu", "metal": device = .gpu
            default:
                fputs("--device must be cpu or gpu\n", stderr); exit(1)
            }
        case "--goldens":
            guard !args.isEmpty else { fputs("--goldens requires a path\n", stderr); exit(1) }
            goldensPath = args.removeFirst()
        case "--json-output":
            guard !args.isEmpty else { fputs("--json-output requires a path\n", stderr); exit(1) }
            jsonOutputPath = args.removeFirst()
        default:
            fputs("Unknown option: \(flag)\n", stderr); printUsage(); exit(1)
        }
    }

    let modelURL = URL(fileURLWithPath: modelDir)
    let wavDirURL = URL(fileURLWithPath: wavDir)

    try Device.withDefaultDevice(device) {
        fputs("[asr-bench] Device: \(device)\n", stderr)

        // Find WAV files
        let wavFiles = try FileManager.default.contentsOfDirectory(at: wavDirURL, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "wav" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !wavFiles.isEmpty else {
            fputs("No WAV files found in \(wavDir)\n", stderr); exit(1)
        }
        fputs("[asr-bench] Found \(wavFiles.count) WAV files\n", stderr)

        // Load model ONCE
        let loadStart = Date()
        let transcriber = try Qwen3ASRTranscriber.load(from: modelURL)
        let loadTime = Date().timeIntervalSince(loadStart)

        // Warmup shaders / kernels on the selected device
        try transcriber.warmup()

        // Load goldens
        var goldens: [String: String] = [:]
        if let gp = goldensPath {
            let data = try Data(contentsOf: URL(fileURLWithPath: gp))
            let gf = try JSONDecoder().decode(GoldenFile.self, from: data)
            for c in gf.cases {
                goldens[c.id] = c.normalized_text
            }
        }

        // Pre-load audio samples into memory (file I/O not part of inference timing)
        fputs("[asr-bench] Pre-loading audio files...\n", stderr)
        var audioSamples: [(url: URL, samples: [Float])] = []
        for wav in wavFiles {
            let samples = try loadAudioFile(wav)
            audioSamples.append((url: wav, samples: samples))
        }

        // Benchmark iterations
        var allTimes: [Double] = []
        var transcripts: [String: String] = [:]

        for it in 1...iterations {
            MLX.Memory.clearCache()
            var iterTime = 0.0
            for (wav, samples) in audioSamples {
                let result = try transcriber.transcribe(audio: samples)
                iterTime += result.processingTime
                transcripts[wav.deletingPathExtension().lastPathComponent] = result.text
            }
            allTimes.append(iterTime)
            fputs("[asr-bench] Iteration \(it)/\(iterations): \(String(format: "%.4f", iterTime))s\n", stderr)
        }

        // Correctness
        var correct = 0
        let total = wavFiles.count
        for wav in wavFiles {
            let fid = wav.deletingPathExtension().lastPathComponent
            if let expected = goldens[fid] {
                let got = normalize(transcripts[fid] ?? "")
                if got == expected {
                    correct += 1
                } else {
                    fputs("[asr-bench] MISMATCH \(fid):\n  got:  \(got)\n  want: \(expected)\n", stderr)
                }
            } else {
                correct += 1
            }
        }

        let meanTime = allTimes.reduce(0, +) / Double(allTimes.count)
        let minTime = allTimes.min()!
        print("METRIC native_inference_mean_s=\(String(format: "%.6f", meanTime))")
        print("METRIC native_inference_min_s=\(String(format: "%.6f", minTime))")
        print("METRIC native_model_load_s=\(String(format: "%.4f", loadTime))")
        print("METRIC native_correctness_exact_count=\(correct)/\(total)")

        for wav in wavFiles {
            let fid = wav.deletingPathExtension().lastPathComponent
            let txt = transcripts[fid] ?? "N/A"
            let preview = String(txt.prefix(80))
            fputs("[asr-bench] \(fid): \(preview)\n", stderr)
        }

        if let jsonPath = jsonOutputPath {
            var entries: [[String: Any]] = []
            for wav in wavFiles {
                let fid = wav.deletingPathExtension().lastPathComponent
                entries.append([
                    "id": fid,
                    "text": transcripts[fid] ?? "",
                    "normalized": normalize(transcripts[fid] ?? "")
                ])
            }
            let jsonObj: [String: Any] = [
                "device": String(describing: device),
                "total": total,
                "inference_mean_s": meanTime,
                "model_load_s": loadTime,
                "results": entries
            ]
            let data = try JSONSerialization.data(withJSONObject: jsonObj, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: jsonPath))
            fputs("[asr-bench] JSON output written to \(jsonPath)\n", stderr)
        }
    }
}

do {
    try runBench()
    exit(0)
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
