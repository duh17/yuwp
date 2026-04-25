import Foundation

struct VoiceGenerationDefaults: Codable, Sendable {
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    var minP: Float?
    var repetitionPenalty: Float?
    var maxTokens: Int?
    var streamingInterval: Double?

    enum CodingKeys: String, CodingKey {
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case maxTokens = "max_tokens"
        case streamingInterval = "streaming_interval"
    }
}

struct VoiceRecord: Codable, Sendable {
    var id: String
    var object = "voice"
    var name: String
    var kind: String
    var prompt: String?
    var voice: String?
    var language: String?
    var referenceText: String?
    var referenceAudioFilename: String?
    var previewFilename: String?
    var tags: [String]
    var defaults: VoiceGenerationDefaults
    var createdAt: String
    var updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case name
        case kind
        case prompt
        case voice
        case language
        case referenceText = "reference_text"
        case referenceAudioFilename = "reference_audio_filename"
        case previewFilename = "preview_filename"
        case tags
        case defaults
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

struct VoiceCreateRequest: Decodable, Sendable {
    var id: String?
    var name: String
    var kind: String?
    var prompt: String?
    var voice: String?
    var language: String?
    var referenceText: String?
    var referenceAudioBase64: String?
    var referenceAudioExtension: String?
    var tags: [String]?
    var defaults: VoiceGenerationDefaults?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case kind
        case prompt
        case voice
        case language
        case referenceText = "reference_text"
        case referenceAudioBase64 = "reference_audio_base64"
        case referenceAudioExtension = "reference_audio_extension"
        case tags
        case defaults
    }
}

struct VoiceUpdateRequest: Decodable, Sendable {
    var name: String?
    var kind: String?
    var prompt: String?
    var voice: String?
    var language: String?
    var referenceText: String?
    var referenceAudioBase64: String?
    var referenceAudioExtension: String?
    var tags: [String]?
    var defaults: VoiceGenerationDefaults?

    enum CodingKeys: String, CodingKey {
        case name
        case kind
        case prompt
        case voice
        case language
        case referenceText = "reference_text"
        case referenceAudioBase64 = "reference_audio_base64"
        case referenceAudioExtension = "reference_audio_extension"
        case tags
        case defaults
    }
}

final class VoiceLibrary: @unchecked Sendable {
    let rootURL: URL
    private let lock = NSLock()
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(rootURL: URL? = nil) throws {
        if let rootURL {
            self.rootURL = rootURL
        } else {
            let appSupport = try FileManager.default.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            self.rootURL = appSupport.appendingPathComponent("Yuwp/Voices", isDirectory: true)
        }
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(at: self.rootURL, withIntermediateDirectories: true)
    }

    func list() throws -> [VoiceRecord] {
        lock.lock()
        defer { lock.unlock() }
        let urls = try FileManager.default.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        let records = urls.compactMap { url -> VoiceRecord? in
            let jsonURL = url.appendingPathComponent("voice.json")
            guard let data = try? Data(contentsOf: jsonURL) else { return nil }
            return try? decoder.decode(VoiceRecord.self, from: data)
        }
        return records.sorted { $0.updatedAt > $1.updatedAt }
    }

    func get(_ id: String) throws -> VoiceRecord? {
        guard Self.isValidID(id) else { throw VoiceLibraryError.invalidID }
        lock.lock()
        defer { lock.unlock() }
        return try readRecordUnlocked(id)
    }

    func create(_ request: VoiceCreateRequest) throws -> VoiceRecord {
        let id = try Self.normalizedID(request.id ?? request.name)
        let now = Self.nowString()
        let kind = request.kind ?? (request.referenceAudioBase64 == nil ? "design" : "clone")
        var record = VoiceRecord(
            id: id,
            name: request.name,
            kind: kind,
            prompt: request.prompt,
            voice: request.voice,
            language: request.language,
            referenceText: request.referenceText,
            referenceAudioFilename: nil,
            previewFilename: nil,
            tags: request.tags ?? [],
            defaults: request.defaults ?? VoiceGenerationDefaults(),
            createdAt: now,
            updatedAt: now
        )

        lock.lock()
        defer { lock.unlock() }
        let dir = voiceDirectoryUnlocked(id)
        if FileManager.default.fileExists(atPath: dir.path) {
            throw VoiceLibraryError.conflict("voice already exists: \(id)")
        }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let encoded = request.referenceAudioBase64 {
            record.referenceAudioFilename = try writeReferenceAudioUnlocked(encoded, extensionHint: request.referenceAudioExtension, voiceID: id)
        }
        try writeRecordUnlocked(record)
        return record
    }

    func update(id: String, request: VoiceUpdateRequest) throws -> VoiceRecord {
        guard Self.isValidID(id) else { throw VoiceLibraryError.invalidID }
        lock.lock()
        defer { lock.unlock() }
        guard var record = try readRecordUnlocked(id) else { throw VoiceLibraryError.notFound }
        if let name = request.name { record.name = name }
        if let kind = request.kind { record.kind = kind }
        if request.prompt != nil { record.prompt = request.prompt }
        if request.voice != nil { record.voice = request.voice }
        if request.language != nil { record.language = request.language }
        if request.referenceText != nil { record.referenceText = request.referenceText }
        if let tags = request.tags { record.tags = tags }
        if let defaults = request.defaults { record.defaults = defaults }
        if let encoded = request.referenceAudioBase64 {
            record.referenceAudioFilename = try writeReferenceAudioUnlocked(encoded, extensionHint: request.referenceAudioExtension, voiceID: id)
        }
        record.updatedAt = Self.nowString()
        try writeRecordUnlocked(record)
        return record
    }

    func delete(id: String) throws {
        guard Self.isValidID(id) else { throw VoiceLibraryError.invalidID }
        lock.lock()
        defer { lock.unlock() }
        let dir = voiceDirectoryUnlocked(id)
        guard FileManager.default.fileExists(atPath: dir.path) else { throw VoiceLibraryError.notFound }
        try FileManager.default.removeItem(at: dir)
    }

    func referenceAudioURL(for record: VoiceRecord) -> URL? {
        guard let filename = record.referenceAudioFilename else { return nil }
        return voiceDirectoryUnlocked(record.id).appendingPathComponent(filename)
    }

    func previewURL(for id: String) throws -> URL {
        guard Self.isValidID(id) else { throw VoiceLibraryError.invalidID }
        let dir = rootURL.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("preview.wav")
    }

    func markPreview(id: String) throws -> VoiceRecord {
        guard Self.isValidID(id) else { throw VoiceLibraryError.invalidID }
        lock.lock()
        defer { lock.unlock() }
        guard var record = try readRecordUnlocked(id) else { throw VoiceLibraryError.notFound }
        record.previewFilename = "preview.wav"
        record.updatedAt = Self.nowString()
        try writeRecordUnlocked(record)
        return record
    }

    private func readRecordUnlocked(_ id: String) throws -> VoiceRecord? {
        let url = voiceDirectoryUnlocked(id).appendingPathComponent("voice.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try decoder.decode(VoiceRecord.self, from: Data(contentsOf: url))
    }

    private func writeRecordUnlocked(_ record: VoiceRecord) throws {
        let dir = voiceDirectoryUnlocked(record.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try encoder.encode(record)
        try data.write(to: dir.appendingPathComponent("voice.json"), options: [.atomic])
    }

    private func writeReferenceAudioUnlocked(_ encoded: String, extensionHint: String?, voiceID: String) throws -> String {
        let filename = "reference.\(Self.safeAudioExtension(extensionHint))"
        guard let data = Data(base64Encoded: encoded) else { throw VoiceLibraryError.invalidReferenceAudio }
        guard data.count <= 50 * 1024 * 1024 else { throw VoiceLibraryError.invalidReferenceAudio }
        try data.write(to: voiceDirectoryUnlocked(voiceID).appendingPathComponent(filename), options: [.atomic])
        return filename
    }

    private func voiceDirectoryUnlocked(_ id: String) -> URL {
        rootURL.appendingPathComponent(id, isDirectory: true)
    }

    static func normalizedID(_ raw: String) throws -> String {
        let lower = raw.lowercased()
        let mapped = lower.map { char -> Character in
            if char.isLetter || char.isNumber { return char }
            return "-"
        }
        let collapsed = String(mapped)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
        let id = String(collapsed.prefix(64))
        guard isValidID(id) else { throw VoiceLibraryError.invalidID }
        return id
    }

    static func isValidID(_ id: String) -> Bool {
        guard !id.isEmpty, id.count <= 64 else { return false }
        return id.allSatisfy { $0.isLowercase || $0.isNumber || $0 == "-" || $0 == "_" }
    }

    static func safeAudioExtension(_ raw: String?) -> String {
        switch raw?.lowercased() {
        case "wav", "wave": return "wav"
        case "m4a", "mp4": return "m4a"
        case "mp3": return "mp3"
        case "flac": return "flac"
        default: return "wav"
        }
    }

    static func nowString() -> String {
        ISO8601DateFormatter().string(from: Date())
    }
}

enum VoiceLibraryError: LocalizedError {
    case invalidID
    case conflict(String)
    case notFound
    case invalidReferenceAudio

    var errorDescription: String? {
        switch self {
        case .invalidID:
            return "invalid voice id; use lowercase letters, numbers, dash, or underscore"
        case .conflict(let message):
            return message
        case .notFound:
            return "voice not found"
        case .invalidReferenceAudio:
            return "invalid reference audio"
        }
    }
}
