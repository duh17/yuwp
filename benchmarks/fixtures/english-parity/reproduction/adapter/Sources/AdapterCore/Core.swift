import Foundation
import ASRIPC

public enum AdapterError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): message }
    }
}

public enum PCM {
    public static let maxPacketSamples = 1600
    public static func samples(_ bytes: Data) throws -> [Float] {
        guard !bytes.isEmpty, bytes.count.isMultiple(of: 2), bytes.count <= maxPacketSamples * 2 else {
            throw AdapterError.invalid("feed requires 1...1600 mono s16le samples at 16000 Hz")
        }
        let bytes = [UInt8](bytes)
        return stride(from: 0, to: bytes.count, by: 2).map {
            Float(Int16(bitPattern: UInt16(bytes[$0]) | UInt16(bytes[$0 + 1]) << 8)) / 32768
        }
    }
}

public enum FrameReader {
    /// Blocking exact reads. Clean EOF is valid only before the next header.
    public static func next(read: (Int) throws -> Data) throws -> ASRIPCFrame? {
        func exact(_ size: Int, cleanEOF: Bool = false) throws -> Data? {
            var result = Data()
            while result.count < size {
                let part = try read(size - result.count)
                guard !part.isEmpty else {
                    if cleanEOF && result.isEmpty { return nil }
                    throw AdapterError.invalid("truncated frame")
                }
                guard part.count <= size - result.count else { throw AdapterError.invalid("read exceeded requested length") }
                result.append(part)
            }
            return result
        }
        guard let header = try exact(8, cleanEOF: true) else { return nil }
        guard let lengths = ASRIPCFrameCodec.decodeHeader(header), lengths.metadataLength > 0 else {
            throw AdapterError.invalid("invalid frame lengths (JSON <=1MiB, binary <=16MiB)")
        }
        guard let metadata = try exact(lengths.metadataLength), let binary = try exact(lengths.binaryLength) else {
            throw AdapterError.invalid("missing frame body")
        }
        return ASRIPCFrame(metadata: metadata, binary: binary)
    }
}

public struct PreviewSchedule {
    public static let cap = 45 * 16000
    public private(set) var received = 0
    private var nextPreview = 16000
    public init() {}
    public mutating func receive(_ count: Int) throws -> Bool {
        guard count > 0, count <= Self.cap - received else { throw AdapterError.invalid("45 second input cap exceeded or empty audio") }
        received += count
        guard received >= nextPreview else { return false }
        nextPreview = (received / 16000 + 1) * 16000
        return true
    }
}

public enum Transcript {
    public static func join(committed: String, active: String) -> String {
        [committed, active].filter { !$0.isEmpty }.joined(separator: " ")
    }
    public static func response(id: UInt64, sessionID: String, text: String, final: Bool) -> ASRIPCResponse {
        let committed = final ? text : ""
        let active = final ? "" : text
        return ASRIPCResponse(id: id, ok: true, sessionID: sessionID,
            text: join(committed: committed, active: active), committedText: committed,
            activeText: active, updateKind: final ? "final" : "partial",
            batchCorrected: false, isFinal: final)
    }
}

@MainActor public protocol BenchmarkBackend: AnyObject {
    var modelName: String { get }
    var chunkSeconds: Double { get }
    var finalAccuracyPassEnabled: Bool { get }
    func prepare() async throws
    func create() async throws
    func feed(_ samples: [Float]) async throws -> String
    func finish() async throws -> String
    func discard()
}

/// The executable has one blocking-read/await/write loop, never spawned feed tasks.
/// Busy rejection additionally prevents accidental actor reentrancy by future callers.
@MainActor public final class SessionEngine {
    private let backend: any BenchmarkBackend
    private var ready = false
    private var busy = false
    private var sessionID: String?
    private var received = 0
    public init(backend: any BenchmarkBackend) { self.backend = backend }
    public func prepare() async throws {
        ready = false
        try await backend.prepare()
        ready = true
    }
    public func handle(_ request: ASRIPCRequest, binary: Data) async -> ASRIPCResponse {
        guard !busy else { return ASRIPCResponse(id: request.id, ok: false, error: "operation already in flight") }
        busy = true
        defer { busy = false }
        var invalidateOnFailure = false
        do {
            guard ready else { throw AdapterError.invalid("models not ready") }
            if let language = request.language, !["en", "English"].contains(language) {
                throw AdapterError.invalid("only English (en or English) is supported")
            }
            if request.command != .feed && !binary.isEmpty { throw AdapterError.invalid("binary audio only valid for feed") }
            switch request.command {
            case .info:
                guard request.sessionID == nil else { throw AdapterError.invalid("info does not accept session_id") }
                return ASRIPCResponse(id: request.id, ok: true, status: "ready", model: backend.modelName,
                    sampleRate: 16000, chunkSec: backend.chunkSeconds, finalAccuracyPassEnabled: backend.finalAccuracyPassEnabled)
            case .create:
                guard request.sessionID == nil, sessionID == nil else { throw AdapterError.invalid("one stream only; create cannot specify session_id") }
                invalidateOnFailure = true
                try await backend.create()
                let id = UUID().uuidString
                sessionID = id
                received = 0
                return ASRIPCResponse(id: request.id, ok: true, sessionID: id)
            case .feed, .stop:
                guard let id = sessionID, request.sessionID == id else { throw AdapterError.invalid("unknown or missing session_id") }
                invalidateOnFailure = true
                if request.command == .feed {
                    let samples = try PCM.samples(binary)
                    guard samples.count <= PreviewSchedule.cap - received else { throw AdapterError.invalid("45 second input cap exceeded") }
                    let text = try await backend.feed(samples)
                    received += samples.count
                    return Transcript.response(id: request.id, sessionID: id, text: text, final: false)
                }
                guard received > 0 else { throw AdapterError.invalid("cannot finalize an empty stream") }
                let text = try await backend.finish()
                backend.discard()
                sessionID = nil
                return Transcript.response(id: request.id, sessionID: id, text: text, final: true)
            }
        } catch {
            if invalidateOnFailure { backend.discard(); sessionID = nil; received = 0 }
            return ASRIPCResponse(id: request.id, ok: false, error: error.localizedDescription, sessionID: request.sessionID)
        }
    }
}
