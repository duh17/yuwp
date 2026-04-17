import Foundation

public enum ASRIPCTransport: String, Sendable, Codable, Equatable, CaseIterable {
    case http
    case stdio

    public var description: String {
        switch self {
        case .http:
            return "HTTP"
        case .stdio:
            return "Standard I/O"
        }
    }
}

public enum ASRIPCCommand: String, Sendable, Codable, Equatable {
    case info
    case create
    case feed
    case stop
}

public struct ASRIPCRequest: Sendable, Codable, Equatable {
    public let id: UInt64
    public let command: ASRIPCCommand
    public let sessionID: String?

    public init(id: UInt64, command: ASRIPCCommand, sessionID: String? = nil) {
        self.id = id
        self.command = command
        self.sessionID = sessionID
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case command
        case sessionID = "session_id"
    }
}

public struct ASRIPCResponse: Sendable, Codable, Equatable {
    public let id: UInt64
    public let ok: Bool
    public let error: String?
    public let sessionID: String?
    public let status: String?
    public let model: String?
    public let sampleRate: Int?
    public let chunkSec: Double?
    public let finalAccuracyPassEnabled: Bool?
    public let text: String?
    public let committedText: String?
    public let activeText: String?
    public let updateKind: String?
    public let batchCorrected: Bool?
    public let isFinal: Bool?

    public init(
        id: UInt64,
        ok: Bool,
        error: String? = nil,
        sessionID: String? = nil,
        status: String? = nil,
        model: String? = nil,
        sampleRate: Int? = nil,
        chunkSec: Double? = nil,
        finalAccuracyPassEnabled: Bool? = nil,
        text: String? = nil,
        committedText: String? = nil,
        activeText: String? = nil,
        updateKind: String? = nil,
        batchCorrected: Bool? = nil,
        isFinal: Bool? = nil
    ) {
        self.id = id
        self.ok = ok
        self.error = error
        self.sessionID = sessionID
        self.status = status
        self.model = model
        self.sampleRate = sampleRate
        self.chunkSec = chunkSec
        self.finalAccuracyPassEnabled = finalAccuracyPassEnabled
        self.text = text
        self.committedText = committedText
        self.activeText = activeText
        self.updateKind = updateKind
        self.batchCorrected = batchCorrected
        self.isFinal = isFinal
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case ok
        case error
        case sessionID = "session_id"
        case status
        case model
        case sampleRate = "sample_rate"
        case chunkSec = "chunk_sec"
        case finalAccuracyPassEnabled = "final_accuracy_pass_enabled"
        case text
        case committedText = "committed_text"
        case activeText = "active_text"
        case updateKind = "update_kind"
        case batchCorrected = "batch_corrected"
        case isFinal = "is_final"
    }
}

public struct ASRIPCFrame: Sendable, Equatable {
    public let metadata: Data
    public let binary: Data

    public init(metadata: Data, binary: Data = Data()) {
        self.metadata = metadata
        self.binary = binary
    }
}

public enum ASRIPCFrameCodec {
    public static let headerSize = 8
    public static let maxMetadataSize = 1_048_576
    public static let maxBinarySize = 16_777_216

    public static func encodeFrame(metadata: Data, binary: Data = Data()) -> Data {
        precondition(metadata.count <= Int(UInt32.max), "metadata too large")
        precondition(binary.count <= Int(UInt32.max), "binary payload too large")

        var metadataLength = UInt32(metadata.count).bigEndian
        var binaryLength = UInt32(binary.count).bigEndian

        var frame = Data(capacity: headerSize + metadata.count + binary.count)
        withUnsafeBytes(of: &metadataLength) { frame.append(contentsOf: $0) }
        withUnsafeBytes(of: &binaryLength) { frame.append(contentsOf: $0) }
        frame.append(metadata)
        frame.append(binary)
        return frame
    }

    public static func decodeHeader(_ header: Data) -> (metadataLength: Int, binaryLength: Int)? {
        guard header.count == headerSize else { return nil }

        let metadataLength = (UInt32(header[0]) << 24)
            | (UInt32(header[1]) << 16)
            | (UInt32(header[2]) << 8)
            | UInt32(header[3])
        let binaryLength = (UInt32(header[4]) << 24)
            | (UInt32(header[5]) << 16)
            | (UInt32(header[6]) << 8)
            | UInt32(header[7])

        let metadata = Int(metadataLength)
        let binary = Int(binaryLength)
        guard metadata >= 0, binary >= 0 else { return nil }
        guard metadata <= maxMetadataSize, binary <= maxBinarySize else { return nil }
        return (metadata, binary)
    }
}

public enum ASRIPCCodec {
    private static let encoder = JSONEncoder()
    private static let decoder = JSONDecoder()

    public static func encode(_ request: ASRIPCRequest, binary: Data = Data()) throws -> Data {
        let metadata = try encoder.encode(request)
        return ASRIPCFrameCodec.encodeFrame(metadata: metadata, binary: binary)
    }

    public static func encode(_ response: ASRIPCResponse, binary: Data = Data()) throws -> Data {
        let metadata = try encoder.encode(response)
        return ASRIPCFrameCodec.encodeFrame(metadata: metadata, binary: binary)
    }

    public static func decodeRequest(metadata: Data) throws -> ASRIPCRequest {
        try decoder.decode(ASRIPCRequest.self, from: metadata)
    }

    public static func decodeResponse(metadata: Data) throws -> ASRIPCResponse {
        try decoder.decode(ASRIPCResponse.self, from: metadata)
    }
}
