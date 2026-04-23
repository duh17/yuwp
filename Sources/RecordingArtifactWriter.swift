import Foundation

struct RecordingArtifactContext: Sendable, Equatable {
    let sessionID: String?
    let transcriptionModel: String
    let dictationLanguageMode: DictationLanguageMode
    let languageHint: String?
}

struct RecordingArtifactHandle: Sendable, Equatable {
    let audioURL: URL
    let transcriptURL: URL
    let metadataURL: URL
    let context: RecordingArtifactContext
    let durationSeconds: Double
    let createdAt: Date
}

enum RecordingArtifactWriter {
    static func writeRecording(
        pcmData: Data,
        directory: URL,
        context: RecordingArtifactContext,
        date: Date = Date()
    ) throws -> RecordingArtifactHandle {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let baseName = uniqueBaseName(in: directory, preferredBaseName: "yuwp-\(filenameTimestamp(for: date))")
        let audioURL = directory.appendingPathComponent("\(baseName).wav")
        let transcriptURL = directory.appendingPathComponent("\(baseName).txt")
        let metadataURL = directory.appendingPathComponent("\(baseName).json")
        let durationSeconds = Double(pcmData.count) / 32_000.0

        try WAVWriter.write(pcmData, to: audioURL)

        let handle = RecordingArtifactHandle(
            audioURL: audioURL,
            transcriptURL: transcriptURL,
            metadataURL: metadataURL,
            context: context,
            durationSeconds: durationSeconds,
            createdAt: date
        )
        try writeMetadata(transcript: nil, for: handle, updatedAt: date)
        return handle
    }

    static func writeTranscript(
        _ transcript: String,
        for handle: RecordingArtifactHandle,
        updatedAt: Date = Date()
    ) throws {
        let body = transcript.hasSuffix("\n") ? transcript : transcript + "\n"
        try Data(body.utf8).write(to: handle.transcriptURL, options: [.atomic])
        try writeMetadata(transcript: transcript, for: handle, updatedAt: updatedAt)
    }

    private static func writeMetadata(
        transcript: String?,
        for handle: RecordingArtifactHandle,
        updatedAt: Date
    ) throws {
        let metadata = RecordingArtifactMetadata(
            schemaVersion: 1,
            audioFile: handle.audioURL.lastPathComponent,
            transcriptFile: handle.transcriptURL.lastPathComponent,
            sessionID: handle.context.sessionID,
            transcriptionModel: handle.context.transcriptionModel,
            dictationLanguageMode: handle.context.dictationLanguageMode.rawValue,
            languageHint: handle.context.languageHint,
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
        let extensions = ["wav", "txt", "json"]
        return extensions.contains { ext in
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

private struct RecordingArtifactMetadata: Encodable, Equatable {
    let schemaVersion: Int
    let audioFile: String
    let transcriptFile: String
    let sessionID: String?
    let transcriptionModel: String
    let dictationLanguageMode: String
    let languageHint: String?
    let durationSeconds: Double
    let createdAt: String
    let updatedAt: String
    let transcript: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion
        case audioFile
        case transcriptFile
        case sessionID
        case transcriptionModel
        case dictationLanguageMode
        case languageHint
        case durationSeconds
        case createdAt
        case updatedAt
        case transcript
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(audioFile, forKey: .audioFile)
        try container.encode(transcriptFile, forKey: .transcriptFile)
        try encodeNullable(sessionID, forKey: .sessionID, into: &container)
        try container.encode(transcriptionModel, forKey: .transcriptionModel)
        try container.encode(dictationLanguageMode, forKey: .dictationLanguageMode)
        try encodeNullable(languageHint, forKey: .languageHint, into: &container)
        try container.encode(durationSeconds, forKey: .durationSeconds)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(updatedAt, forKey: .updatedAt)
        try encodeNullable(transcript, forKey: .transcript, into: &container)
    }

    private func encodeNullable(
        _ value: String?,
        forKey key: CodingKeys,
        into container: inout KeyedEncodingContainer<CodingKeys>
    ) throws {
        if let value {
            try container.encode(value, forKey: key)
        } else {
            try container.encodeNil(forKey: key)
        }
    }
}
