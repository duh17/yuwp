import Foundation

public struct ASRStreamRecordingConfiguration: Sendable, Equatable {
    public let enabled: Bool
    public let directory: URL
    public let transcriptionModel: String

    public init(enabled: Bool, directory: URL, transcriptionModel: String) {
        self.enabled = enabled
        self.directory = directory.standardizedFileURL
        self.transcriptionModel = transcriptionModel
    }

    public static func disabled(transcriptionModel: String) -> ASRStreamRecordingConfiguration {
        ASRStreamRecordingConfiguration(
            enabled: false,
            directory: Self.defaultRecordingsDirectory,
            transcriptionModel: transcriptionModel
        )
    }

    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment,
        transcriptionModel: String
    ) -> ASRStreamRecordingConfiguration {
        let enabled = parseBool(environment["YUWP_ASR_SAVE_RECORDINGS"])
        let directory = environment["YUWP_ASR_RECORDINGS_DIR"]
            .flatMap { value -> URL? in
                let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return nil }
                let expanded = NSString(string: trimmed).expandingTildeInPath
                return URL(fileURLWithPath: expanded, isDirectory: true)
            }
            ?? Self.defaultRecordingsDirectory
        return ASRStreamRecordingConfiguration(
            enabled: enabled,
            directory: directory,
            transcriptionModel: transcriptionModel
        )
    }

    public static var defaultRecordingsDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Yuwp/recordings", isDirectory: true)
    }

    private static func parseBool(_ raw: String?) -> Bool {
        guard let raw else { return false }
        return switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on": true
        default: false
        }
    }
}

struct ASRStreamRecordingHandle: Sendable, Equatable {
    let audioURL: URL
    let transcriptURL: URL
    let metadataURL: URL
    let durationSeconds: Double
    let createdAt: Date
}

struct ASRStreamRecordingContext: Sendable, Equatable {
    let sessionID: String
    let transcriptionModel: String
    let languageHint: String?
}

enum ASRStreamRecordingArtifactWriter {
    static func writeRecording(
        pcmData: Data,
        directory: URL,
        context: ASRStreamRecordingContext,
        date: Date = Date()
    ) throws -> ASRStreamRecordingHandle {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let baseName = uniqueBaseName(in: directory, preferredBaseName: "yuwp-asr-\(filenameTimestamp(for: date))")
        let audioURL = directory.appendingPathComponent("\(baseName).wav")
        let transcriptURL = directory.appendingPathComponent("\(baseName).txt")
        let metadataURL = directory.appendingPathComponent("\(baseName).json")
        let durationSeconds = Double(pcmData.count) / 32_000.0

        try writeWAV(pcmData, to: audioURL)

        let handle = ASRStreamRecordingHandle(
            audioURL: audioURL,
            transcriptURL: transcriptURL,
            metadataURL: metadataURL,
            durationSeconds: durationSeconds,
            createdAt: date
        )
        try writeMetadata(transcript: nil, for: handle, context: context, updatedAt: date)
        return handle
    }

    static func writeTranscript(
        _ transcript: String,
        for handle: ASRStreamRecordingHandle,
        context: ASRStreamRecordingContext,
        updatedAt: Date = Date()
    ) throws {
        let body = transcript.hasSuffix("\n") ? transcript : transcript + "\n"
        try Data(body.utf8).write(to: handle.transcriptURL, options: [.atomic])
        try writeMetadata(transcript: transcript, for: handle, context: context, updatedAt: updatedAt)
    }

    private static func writeWAV(_ pcmData: Data, to url: URL) throws {
        var data = Data()
        data.append("RIFF".data(using: .ascii)!)
        data.appendUInt32LE(UInt32(36 + pcmData.count))
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!)
        data.appendUInt32LE(16)
        data.appendUInt16LE(1)
        data.appendUInt16LE(1)
        data.appendUInt32LE(16_000)
        data.appendUInt32LE(32_000)
        data.appendUInt16LE(2)
        data.appendUInt16LE(16)
        data.append("data".data(using: .ascii)!)
        data.appendUInt32LE(UInt32(pcmData.count))
        data.append(pcmData)
        try data.write(to: url, options: [.atomic])
    }

    private static func writeMetadata(
        transcript: String?,
        for handle: ASRStreamRecordingHandle,
        context: ASRStreamRecordingContext,
        updatedAt: Date
    ) throws {
        let metadata = ASRStreamRecordingMetadata(
            schemaVersion: 1,
            source: "asr_stream",
            audioFile: handle.audioURL.lastPathComponent,
            transcriptFile: handle.transcriptURL.lastPathComponent,
            sessionID: context.sessionID,
            transcriptionModel: context.transcriptionModel,
            languageHint: context.languageHint,
            durationSeconds: handle.durationSeconds,
            createdAt: isoTimestamp(for: handle.createdAt),
            updatedAt: isoTimestamp(for: updatedAt),
            transcript: transcript
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: handle.metadataURL, options: [.atomic])
    }

    private static func uniqueBaseName(in directory: URL, preferredBaseName: String) -> String {
        var candidate = preferredBaseName
        var index = 2

        while artifactExists(in: directory, baseName: candidate) {
            candidate = "\(preferredBaseName)-\(index)"
            index += 1
        }

        return candidate
    }

    private static func artifactExists(in directory: URL, baseName: String) -> Bool {
        ["wav", "txt", "json"].contains { ext in
            FileManager.default.fileExists(atPath: directory.appendingPathComponent("\(baseName).\(ext)").path)
        }
    }

    private static func filenameTimestamp(for date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return formatter.string(from: date)
    }

    private static func isoTimestamp(for date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}

private struct ASRStreamRecordingMetadata: Encodable, Equatable {
    let schemaVersion: Int
    let source: String
    let audioFile: String
    let transcriptFile: String
    let sessionID: String
    let transcriptionModel: String
    let languageHint: String?
    let durationSeconds: Double
    let createdAt: String
    let updatedAt: String
    let transcript: String?
}

private extension Data {
    mutating func appendUInt16LE(_ value: UInt16) {
        var littleEndian = value.littleEndian
        let bytes = Swift.withUnsafeBytes(of: &littleEndian) { Data($0) }
        append(bytes)
    }

    mutating func appendUInt32LE(_ value: UInt32) {
        var littleEndian = value.littleEndian
        let bytes = Swift.withUnsafeBytes(of: &littleEndian) { Data($0) }
        append(bytes)
    }
}
