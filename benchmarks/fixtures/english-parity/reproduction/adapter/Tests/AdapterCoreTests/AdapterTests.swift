import Foundation
import Testing
import ASRIPC
@testable import AdapterCore

@Test func pcmIsSignedLittleEndianAndNormalized() throws {
    #expect(try PCM.samples(Data([0, 128, 255, 255, 0, 0, 255, 127])) == [-1, -1.0 / 32768, 0, 32767.0 / 32768])
    #expect(throws: AdapterError.self) { try PCM.samples(Data([1])) }
    #expect(throws: AdapterError.self) { try PCM.samples(Data()) }
    #expect(throws: AdapterError.self) { try PCM.samples(Data(repeating: 0, count: 3202)) }
}

@Test func framingHandlesFragmentationEOFAndLimits() throws {
    let request = ASRIPCRequest(id: 42, command: .feed, sessionID: "s")
    let encoded = try ASRIPCCodec.encode(request, binary: Data([1, 2]))
    #expect(Array(encoded.prefix(4)) == [0, 0, 0, UInt8(encoded.count - 10)])
    var bytes = encoded
    let frame = try FrameReader.next { n in
        let part = Data(bytes.prefix(min(n, 1)))
        bytes.removeFirst(part.count)
        return part
    }
    #expect(try ASRIPCCodec.decodeRequest(metadata: #require(frame).metadata) == request)
    #expect(frame?.binary == Data([1, 2]))
    #expect(try FrameReader.next { _ in Data() } == nil)
    for truncated in [Data(encoded.prefix(3)), Data(encoded.dropLast())] {
        var remaining = truncated
        #expect(throws: AdapterError.self) {
            try FrameReader.next { n in
                let part = Data(remaining.prefix(n)); remaining.removeFirst(part.count); return part
            }
        }
    }
    #expect(ASRIPCFrameCodec.decodeHeader(Data([0, 16, 0, 1, 0, 0, 0, 0])) == nil)
    #expect(ASRIPCFrameCodec.decodeHeader(Data([0, 0, 0, 1, 1, 0, 0, 1])) == nil)
    #expect(throws: AdapterError.self) {
        try FrameReader.next { _ in Data([0, 16, 0, 1, 0, 0, 0, 0]) }
    }
}

@Test func previewClockUsesReceivedSamplesAndHardCap() throws {
    var schedule = PreviewSchedule()
    for _ in 0..<9 { #expect(try !schedule.receive(1600)) }
    #expect(try schedule.receive(1600))
    #expect(try !schedule.receive(1))
    #expect(try schedule.receive(15999))
    #expect(schedule.received == 32000)
    #expect(try schedule.receive(720000 - 32000))
    #expect(throws: AdapterError.self) { try schedule.receive(1) }
    #expect(schedule.received == 720000)
}

@Test func transcriptReconstructionCommitsOnlyAtFinal() {
    let partial = Transcript.response(id: 1, sessionID: "s", text: "hello world", final: false)
    #expect(partial.text == "hello world")
    #expect(partial.committedText == "")
    #expect(partial.activeText == "hello world")
    let final = Transcript.response(id: 2, sessionID: "s", text: "hello again", final: true)
    #expect(final.committedText == "hello again")
    #expect(final.activeText == "")
    #expect(final.isFinal == true)
    #expect(Transcript.join(committed: "hello", active: "world") == "hello world")
    #expect(Transcript.join(committed: "", active: "") == "")
}

@MainActor private final class FakeBackend: BenchmarkBackend {
    var starts = 0
    var feeds = 0
    var finishes = 0
    var fails = false
    var ready = false
    var feedEntered: CheckedContinuation<Void, Never>?
    var feedRelease: CheckedContinuation<Void, Never>?
    var suspendFeed = false
    let modelName = "fake"
    let chunkSeconds = 1.0
    let finalAccuracyPassEnabled = true
    func prepare() async throws { ready = true }
    func create() async throws { starts += 1 }
    func feed(_ samples: [Float]) async throws -> String {
        feeds += 1
        if suspendFeed {
            await withCheckedContinuation { continuation in
                feedRelease = continuation
                feedEntered?.resume()
                feedEntered = nil
            }
        }
        if fails { throw AdapterError.invalid("inference failed") }
        return "preview"
    }
    func finish() async throws -> String { finishes += 1; return "final" }
    func discard() { }
}

@MainActor @Test func sessionErrorsAreFailuresAndNeverEmptySuccess() async throws {
    let backend = FakeBackend()
    let engine = SessionEngine(backend: backend)
    let early = await engine.handle(ASRIPCRequest(id: 1, command: .info), binary: Data())
    #expect(early.ok == false)
    try await engine.prepare()
    let info = await engine.handle(ASRIPCRequest(id: 2, command: .info), binary: Data())
    #expect(info.status == "ready")
    #expect(info.finalAccuracyPassEnabled == true)
    let badLanguage = await engine.handle(ASRIPCRequest(id: 3, command: .create, language: "zh"), binary: Data())
    #expect(!badLanguage.ok)
    #expect(backend.starts == 0)
    let created = await engine.handle(ASRIPCRequest(id: 4, command: .create, language: "en"), binary: Data())
    let sid = try #require(created.sessionID)
    #expect(created.ok)
    let duplicate = await engine.handle(ASRIPCRequest(id: 5, command: .create, language: "en"), binary: Data())
    #expect(!duplicate.ok)
    let wrong = await engine.handle(ASRIPCRequest(id: 6, command: .feed, sessionID: "wrong"), binary: Data([0, 0]))
    #expect(!wrong.ok)
    #expect(backend.feeds == 0)
    let fed = await engine.handle(ASRIPCRequest(id: 7, command: .feed, sessionID: sid), binary: Data([0, 0]))
    #expect(fed.activeText == "preview")
    #expect(fed.committedText == "")
    backend.fails = true
    let failed = await engine.handle(ASRIPCRequest(id: 8, command: .feed, sessionID: sid), binary: Data([0, 0]))
    #expect(!failed.ok)
    #expect(failed.text == nil)
    let stop = await engine.handle(ASRIPCRequest(id: 9, command: .stop, sessionID: sid), binary: Data())
    #expect(!stop.ok)
    #expect(backend.finishes == 0)
    let next = await engine.handle(ASRIPCRequest(id: 10, command: .create, language: "English"), binary: Data())
    #expect(next.ok)
    #expect(backend.starts == 2)
}

@MainActor @Test func finalizationAndCapFailClosed() async throws {
    let backend = FakeBackend()
    let engine = SessionEngine(backend: backend)
    try await engine.prepare()
    func create(_ id: UInt64) async throws -> String {
        let response = await engine.handle(ASRIPCRequest(id: id, command: .create, language: "en"), binary: Data())
        return try #require(response.sessionID)
    }
    let empty = try await create(1)
    let emptyStop = await engine.handle(ASRIPCRequest(id: 2, command: .stop, sessionID: empty), binary: Data())
    #expect(!emptyStop.ok)
    let malformed = try await create(3)
    let badFeed = await engine.handle(ASRIPCRequest(id: 4, command: .feed, sessionID: malformed), binary: Data([1]))
    #expect(!badFeed.ok)
    #expect(backend.feeds == 0)
    let sid = try await create(5)
    let packet = Data(repeating: 0, count: 3200)
    for id in 6..<456 {
        let response = await engine.handle(ASRIPCRequest(id: UInt64(id), command: .feed, sessionID: sid), binary: packet)
        #expect(response.ok)
    }
    let overcap = await engine.handle(ASRIPCRequest(id: 456, command: .feed, sessionID: sid), binary: Data([0, 0]))
    #expect(!overcap.ok)
    #expect(backend.feeds == 450)
    let afterCap = await engine.handle(ASRIPCRequest(id: 457, command: .stop, sessionID: sid), binary: Data())
    #expect(!afterCap.ok)
    #expect(backend.finishes == 0)
    let finalSID = try await create(458)
    _ = await engine.handle(ASRIPCRequest(id: 459, command: .feed, sessionID: finalSID), binary: packet)
    let final = await engine.handle(ASRIPCRequest(id: 460, command: .stop, sessionID: finalSID), binary: Data())
    #expect(final.ok && final.isFinal == true)
    #expect(final.text == "final" && final.committedText == "final" && final.activeText == "")
    let duplicate = await engine.handle(ASRIPCRequest(id: 461, command: .stop, sessionID: finalSID), binary: Data())
    #expect(!duplicate.ok)
    #expect(backend.finishes == 1)
}

@MainActor @Test func finishCannotReenterSuspendedFeed() async throws {
    let backend = FakeBackend()
    let engine = SessionEngine(backend: backend)
    try await engine.prepare()
    let created = await engine.handle(ASRIPCRequest(id: 1, command: .create), binary: Data())
    let sid = try #require(created.sessionID)
    backend.suspendFeed = true
    var feedTask: Task<ASRIPCResponse, Never>?
    await withCheckedContinuation { entered in
        backend.feedEntered = entered
        feedTask = Task { @MainActor in
            await engine.handle(ASRIPCRequest(id: 2, command: .feed, sessionID: sid), binary: Data([0, 0]))
        }
    }
    let feed = try #require(feedTask)
    let stop = await engine.handle(ASRIPCRequest(id: 3, command: .stop, sessionID: sid), binary: Data())
    #expect(!stop.ok)
    #expect(backend.finishes == 0)
    backend.feedRelease?.resume()
    backend.feedRelease = nil
    #expect(await feed.value.ok)
}

