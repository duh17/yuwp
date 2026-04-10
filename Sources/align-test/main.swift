// align-test — Verify ForcedAligner model loading and alignment
// Usage: align-test <aligner-model-dir> [wav-file] [transcript]

import Foundation
import MLX
import NativeASR

func loadAudioFile(_ path: String) throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)

    // Skip WAV header (44 bytes), read s16le PCM
    guard data.count > 44 else { throw NSError(domain: "Audio", code: 1, userInfo: [NSLocalizedDescriptionKey: "File too small"]) }
    let pcmData = data.subdata(in: 44 ..< data.count)
    let sampleCount = pcmData.count / 2
    var samples = [Float](repeating: 0, count: sampleCount)
    pcmData.withUnsafeBytes { raw in
        let int16s = raw.bindMemory(to: Int16.self)
        for i in 0 ..< sampleCount {
            samples[i] = Float(int16s[i]) / 32768.0
        }
    }
    return samples
}

func main() throws {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        fputs("Usage: align-test <aligner-model-dir> [wav-file] [transcript]\n", stderr)
        return
    }

    let modelDir = URL(fileURLWithPath: args[1])
    fputs("Loading ForcedAligner from \(modelDir.path)...\n", stderr)

    let t0 = CFAbsoluteTimeGetCurrent()
    let aligner = try ForcedAligner.load(from: modelDir)
    let loadTime = CFAbsoluteTimeGetCurrent() - t0
    fputs("Model loaded in \(String(format: "%.2f", loadTime))s\n", stderr)
    fputs("  classify_num: \(aligner.model.config.classifyNum)\n", stderr)
    fputs("  timestamp_segment_time: \(aligner.model.config.timestampSegmentTime)ms\n", stderr)

    // If audio file and transcript provided, run alignment
    if args.count >= 4 {
        let wavPath = args[2]
        let transcript = args[3]

        fputs("\nAligning: \"\(transcript)\"\n", stderr)
        fputs("Audio: \(wavPath)\n", stderr)

        let audio = try loadAudioFile(wavPath)
        fputs("Audio samples: \(audio.count) (\(String(format: "%.1f", Double(audio.count) / 16000.0))s)\n", stderr)

        let t1 = CFAbsoluteTimeGetCurrent()
        let items = aligner.align(audio: audio, text: transcript)
        let alignTime = CFAbsoluteTimeGetCurrent() - t1
        fputs("Alignment done in \(String(format: "%.2f", alignTime))s\n\n", stderr)

        // Print results
        for item in items {
            print(String(format: "%8.3f → %8.3f  %@", item.startTime, item.endTime, item.text))
        }
    } else {
        fputs("\nModel loaded successfully. Pass a WAV file and transcript to test alignment.\n", stderr)
    }
}

try main()
