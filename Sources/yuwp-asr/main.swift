import ASRServerSupport
import Foundation
import NativeASR

private struct CompanionCommand {
    let binaryName: String
}

private let companionCommands: [String: CompanionCommand] = [
    "restitch": .init(binaryName: "asr-stitch-debug"),
    "test": .init(binaryName: "asr-test"),
    "align": .init(binaryName: "align-test"),
]

private func printUsage() {
    fputs("""
    Usage: yuwp-asr <command> [options]

    Commands:
      transcribe <audio-file> [options]   One-shot local transcription
      serve [options]                     Run the local ASR HTTP server
      restitch <debug-json> [options]     Restitch subtitle output from debug JSON
      test ...                            Run internal batch smoke tooling
      align ...                           Run internal forced-aligner checks

    Examples:
      yuwp-asr transcribe note.m4a --format json
      yuwp-asr transcribe note.m4a --model ~/models/Qwen3-ASR-0.6B-4bit
      yuwp-asr serve --model ~/models/Qwen3-ASR-1.7B-bf16 --port 9748
      yuwp-asr restitch /tmp/debug.json --format srt

    """, stderr)
}

private func findCompanionBinary(named binaryName: String) -> String? {
    let executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    let executableDir = executableURL.deletingLastPathComponent()
    let candidates = [
        executableDir.appendingPathComponent(binaryName).path,
        Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/\(binaryName)").path,
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(".build/arm64-apple-macosx/release/\(binaryName)").path,
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(".build/arm64-apple-macosx/debug/\(binaryName)").path,
    ]
    return candidates.first { FileManager.default.fileExists(atPath: $0) }
}

private func runServe(arguments: [String]) -> Int32 {
    if arguments.contains("--help") || arguments.contains("-h") {
        let usage = asrServerUsage.replacingOccurrences(of: "Usage: swift-mlx-asr-server", with: "Usage: yuwp-asr serve")
        fputs("\(usage)\n", stderr)
        return 0
    }

    guard let serverBinary = findCompanionBinary(named: "swift-mlx-asr-server") else {
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

private func runCompanion(commandName: String, command: CompanionCommand, arguments: [String]) -> Int32 {
    guard let binary = findCompanionBinary(named: command.binaryName) else {
        fputs(
            "Error: could not find \(command.binaryName). Build it with `swift build --product \(command.binaryName)`.\n",
            stderr
        )
        return 1
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: binary)
    process.arguments = arguments
    process.standardInput = FileHandle.standardInput
    process.standardOutput = FileHandle.standardOutput
    process.standardError = FileHandle.standardError

    do {
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    } catch {
        fputs("Error: failed to launch \(commandName) (\(command.binaryName)): \(error.localizedDescription)\n", stderr)
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
case let name:
    if let companion = companionCommands[name] {
        exit(runCompanion(commandName: name, command: companion, arguments: Array(args.dropFirst())))
    }
    fputs("Error: unknown command '\(command)'\n\n", stderr)
    printUsage()
    exit(1)
}
