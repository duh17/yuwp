import ASRServerSupport
import Foundation
import NativeASR

private func printUsage() {
    fputs("""
    Usage: yuwp-asr <command> [options]

    Commands:
      transcribe <audio-file> [options]   One-shot local transcription
      serve [options]                     Run the local ASR HTTP server

    Examples:
      yuwp-asr transcribe note.m4a --format json
      yuwp-asr transcribe note.m4a --model ~/models/Qwen3-ASR-0.6B-4bit
      yuwp-asr serve --model ~/models/Qwen3-ASR-1.7B-bf16 --port 9748

    """, stderr)
}

private func findServerBinary() -> String? {
    let executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    let executableDir = executableURL.deletingLastPathComponent()
    let candidates = [
        executableDir.appendingPathComponent("swift-mlx-asr-server").path,
        Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/swift-mlx-asr-server").path,
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(".build/arm64-apple-macosx/release/swift-mlx-asr-server").path,
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(".build/arm64-apple-macosx/debug/swift-mlx-asr-server").path,
    ]
    return candidates.first { FileManager.default.fileExists(atPath: $0) }
}

private func runServe(arguments: [String]) -> Int32 {
    if arguments.contains("--help") || arguments.contains("-h") {
        let usage = asrServerUsage.replacingOccurrences(of: "Usage: swift-mlx-asr-server", with: "Usage: yuwp-asr serve")
        fputs("\(usage)\n", stderr)
        return 0
    }

    guard let serverBinary = findServerBinary() else {
        fputs("Error: could not find swift-mlx-asr-server. Build it with `swift build --product swift-mlx-asr-server`.\n", stderr)
        return 1
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: serverBinary)
    process.arguments = arguments
    process.standardInput = FileHandle.standardInput
    process.standardOutput = FileHandle.standardOutput
    process.standardError = FileHandle.standardError

    do {
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    } catch {
        fputs("Error: failed to launch swift-mlx-asr-server: \(error.localizedDescription)\n", stderr)
        return 1
    }
}

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    printUsage()
    exit(1)
}

switch command {
case "transcribe":
    exit(Int32(YuwpTranscribeSupport.runCLI(arguments: Array(args.dropFirst()), programName: "yuwp-asr transcribe")))
case "serve":
    exit(runServe(arguments: Array(args.dropFirst())))
case "--help", "-h", "help":
    printUsage()
    exit(0)
default:
    fputs("Error: unknown command '\(command)'\n\n", stderr)
    printUsage()
    exit(1)
}
