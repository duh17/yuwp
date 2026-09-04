import ASRIPC
import Foundation
import NativeASR

public let asrServerUsage = "Usage: yuwp-asr serve [--model <path-or-repo-id>] [--batch-model <dir>] [--aligner-model <dir>] [--batch-chunking <automatic|vad|energy>] [--transport <http|stdio> (default: stdio)] [--disable-vad] [--disable-batch-retranscribe] [--port \(ASRIPCDefaults.defaultHTTPPort)] [--host 127.0.0.1] [--parent-pid <pid>] [--warmup]"

public struct ASRServerCLIConfiguration: Equatable {
    public let modelSpec: String?
    public let port: UInt16
    public let host: String
    public let parentPID: Int32?
    public let warmup: Bool
    public let batchModelPath: String?
    public let alignerModelPath: String?
    public let batchRetranscribeEnabled: Bool
    public let vadEnabled: Bool
    public let batchChunking: BatchChunkingMode
    public let transport: ASRIPCTransport

    public init(
        modelSpec: String? = nil,
        port: UInt16 = ASRIPCDefaults.defaultHTTPPort,
        host: String = "127.0.0.1",
        parentPID: Int32? = nil,
        warmup: Bool = false,
        batchModelPath: String? = nil,
        alignerModelPath: String? = nil,
        batchRetranscribeEnabled: Bool = true,
        vadEnabled: Bool = true,
        batchChunking: BatchChunkingMode = .automatic,
        transport: ASRIPCTransport = .stdio
    ) {
        self.modelSpec = modelSpec?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.port = port
        self.host = host
        self.parentPID = parentPID
        self.warmup = warmup
        self.batchModelPath = batchModelPath
        self.alignerModelPath = alignerModelPath
        self.batchRetranscribeEnabled = batchRetranscribeEnabled
        self.vadEnabled = vadEnabled
        self.batchChunking = batchChunking
        self.transport = transport
    }
}

public enum ASRServerCLIError: Error, Equatable {
    case missingValue(flag: String)
    case invalidPort(String)
    case invalidParentPID(String)
    case invalidTransport(String)
    case invalidBatchChunking(String)
    case unknownOption(String)
}

extension ASRServerCLIError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingValue(let flag):
            return "\(flag) requires a value"
        case .invalidPort:
            return "--port requires a number"
        case .invalidParentPID:
            return "--parent-pid requires a pid"
        case .invalidTransport(let value):
            return "--transport must be one of: http, stdio (got '\(value)')"
        case .invalidBatchChunking(let value):
            return "--batch-chunking must be one of: automatic, vad, energy (got '\(value)')"
        case .unknownOption(let flag):
            return "Unknown option: \(flag)"
        }
    }
}

public func parseASRServerCLI(arguments: [String]) throws -> ASRServerCLIConfiguration {
    var args = arguments
    var positionalModelSpec: String?
    if let first = args.first, !first.hasPrefix("-") {
        positionalModelSpec = args.removeFirst()
    }

    var explicitModelSpec: String?
    var port: UInt16 = ASRIPCDefaults.defaultHTTPPort
    var host = "127.0.0.1"
    var parentPID: Int32?
    var warmup = false
    var batchModelPath: String?
    var alignerModelPath: String?
    var batchRetranscribeEnabled = true
    var vadEnabled = true
    var batchChunking: BatchChunkingMode = .automatic
    var batchChunkingExplicit = false
    var transport: ASRIPCTransport = .stdio

    while !args.isEmpty {
        switch args.removeFirst() {
        case "--model":
            guard !args.isEmpty else { throw ASRServerCLIError.missingValue(flag: "--model") }
            explicitModelSpec = args.removeFirst()
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
        case "--batch-chunking":
            guard !args.isEmpty else { throw ASRServerCLIError.missingValue(flag: "--batch-chunking") }
            let value = args.removeFirst().trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard let parsed = BatchChunkingMode(rawValue: value) else {
                throw ASRServerCLIError.invalidBatchChunking(value)
            }
            batchChunking = parsed
            batchChunkingExplicit = true
        case "--disable-vad":
            vadEnabled = false
            if !batchChunkingExplicit {
                batchChunking = .energy
            }
        case "--transport":
            guard !args.isEmpty else { throw ASRServerCLIError.missingValue(flag: "--transport") }
            let value = args.removeFirst().trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard let parsed = ASRIPCTransport(rawValue: value) else {
                throw ASRServerCLIError.invalidTransport(value)
            }
            transport = parsed
        case let flag where flag.hasPrefix("-"):
            throw ASRServerCLIError.unknownOption(flag)
        case let value:
            throw ASRServerCLIError.unknownOption(value)
        }
    }

    return ASRServerCLIConfiguration(
        modelSpec: explicitModelSpec ?? positionalModelSpec,
        port: port,
        host: host,
        parentPID: parentPID,
        warmup: warmup,
        batchModelPath: batchModelPath,
        alignerModelPath: alignerModelPath,
        batchRetranscribeEnabled: batchRetranscribeEnabled,
        vadEnabled: vadEnabled,
        batchChunking: batchChunking,
        transport: transport
    )
}
