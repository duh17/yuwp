import Foundation

public let asrServerUsage = "Usage: asr-server <streaming-model-dir> [--batch-model <dir>] [--aligner-model <dir>] [--disable-batch-retranscribe] [--port 9748] [--host 127.0.0.1] [--parent-pid <pid>] [--warmup]"

public struct ASRServerCLIConfiguration: Equatable {
    public let modelPath: String
    public let port: UInt16
    public let host: String
    public let parentPID: Int32?
    public let warmup: Bool
    public let batchModelPath: String?
    public let alignerModelPath: String?
    public let batchRetranscribeEnabled: Bool

    public init(
        modelPath: String,
        port: UInt16 = 9748,
        host: String = "127.0.0.1",
        parentPID: Int32? = nil,
        warmup: Bool = false,
        batchModelPath: String? = nil,
        alignerModelPath: String? = nil,
        batchRetranscribeEnabled: Bool = true
    ) {
        self.modelPath = modelPath
        self.port = port
        self.host = host
        self.parentPID = parentPID
        self.warmup = warmup
        self.batchModelPath = batchModelPath
        self.alignerModelPath = alignerModelPath
        self.batchRetranscribeEnabled = batchRetranscribeEnabled
    }
}

public enum ASRServerCLIError: Error, Equatable {
    case missingModelPath
    case missingValue(flag: String)
    case invalidPort(String)
    case invalidParentPID(String)
    case unknownOption(String)
}

extension ASRServerCLIError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingModelPath:
            return asrServerUsage
        case .missingValue(let flag):
            return "\(flag) requires a value"
        case .invalidPort:
            return "--port requires a number"
        case .invalidParentPID:
            return "--parent-pid requires a pid"
        case .unknownOption(let flag):
            return "Unknown option: \(flag)"
        }
    }
}

public func parseASRServerCLI(arguments: [String]) throws -> ASRServerCLIConfiguration {
    var args = arguments
    guard !args.isEmpty else {
        throw ASRServerCLIError.missingModelPath
    }

    let modelPath = args.removeFirst()
    var port: UInt16 = 9748
    var host = "127.0.0.1"
    var parentPID: Int32?
    var warmup = false
    var batchModelPath: String?
    var alignerModelPath: String?
    var batchRetranscribeEnabled = true

    while !args.isEmpty {
        switch args.removeFirst() {
        case "--port":
            guard !args.isEmpty else { throw ASRServerCLIError.missingValue(flag: "--port") }
            let raw = args.removeFirst()
            guard let parsed = UInt16(raw) else { throw ASRServerCLIError.invalidPort(raw) }
            port = parsed
        case "--host":
            guard !args.isEmpty else { throw ASRServerCLIError.missingValue(flag: "--host") }
            host = args.removeFirst()
        case "--parent-pid":
            guard !args.isEmpty else { throw ASRServerCLIError.missingValue(flag: "--parent-pid") }
            let raw = args.removeFirst()
            guard let parsed = Int32(raw) else { throw ASRServerCLIError.invalidParentPID(raw) }
            parentPID = parsed
        case "--warmup":
            warmup = true
        case "--batch-model":
            guard !args.isEmpty else { throw ASRServerCLIError.missingValue(flag: "--batch-model") }
            batchModelPath = args.removeFirst()
        case "--aligner-model":
            guard !args.isEmpty else { throw ASRServerCLIError.missingValue(flag: "--aligner-model") }
            alignerModelPath = args.removeFirst()
        case "--disable-batch-retranscribe":
            batchRetranscribeEnabled = false
        case let flag:
            throw ASRServerCLIError.unknownOption(flag)
        }
    }

    return ASRServerCLIConfiguration(
        modelPath: modelPath,
        port: port,
        host: host,
        parentPID: parentPID,
        warmup: warmup,
        batchModelPath: batchModelPath,
        alignerModelPath: alignerModelPath,
        batchRetranscribeEnabled: batchRetranscribeEnabled
    )
}
