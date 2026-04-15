import Foundation
import Testing
@testable import Yuwp

@Suite("Terminal injection regressions")
struct TerminalInjectionRegressionTests {
    @Test @MainActor func mixedLanguageTraceCommitsExpectedFinalText() async throws {
        let trace = try loadTraceFixture("terminal-trace-mixed-language.json")
        let terminal = ImmediateTerminalBuffer()
        let harness = makeHarness(
            transport: terminal.transport,
            bulkInserter: BulkInsertProxy { terminal.paste($0) },
            terminalText: { terminal.snapshot() }
        )

        try await replay(trace: trace, into: harness.session, stt: harness.stt)
        harness.drainInjector()

        #expect(harness.terminalText() == trace.finalText)
        #expect(harness.finishCount == 1)
    }

    @Test @MainActor func longTailCorrectionTraceCommitsExpectedFinalText() async throws {
        let trace = try loadTraceFixture("terminal-trace-long-tail-correction.json")
        let terminal = ImmediateTerminalBuffer()
        let harness = makeHarness(
            transport: terminal.transport,
            bulkInserter: BulkInsertProxy { terminal.paste($0) },
            terminalText: { terminal.snapshot() }
        )

        try await replay(trace: trace, into: harness.session, stt: harness.stt)
        harness.drainInjector()

        #expect(harness.terminalText() == trace.finalText)
        #expect(harness.finishCount == 1)
    }

    @Test func longTailCorrectionRequiresLargeRewriteAtCommit() throws {
        let trace = try loadTraceFixture("terminal-trace-long-tail-correction.json")
        let previous = try #require(trace.updates.dropLast().last?.text)
        let finalText = try #require(trace.updates.last?.text)

        let (backspaces, suffix) = CGEventInjector.diff(old: previous, new: finalText)

        #expect(backspaces == 42)
        #expect(suffix == "vent injection. Let's see why this happens.")
    }

    @Test func mixedLanguageTraceRequiresLargeRewriteAtCommit() throws {
        let trace = try loadTraceFixture("terminal-trace-mixed-language.json")
        let previous = try #require(trace.updates.dropLast().last?.text)
        let finalText = try #require(trace.updates.last?.text)

        let (backspaces, suffix) = CGEventInjector.diff(old: previous, new: finalText)

        #expect(backspaces == 133)
        #expect(suffix.count == 230)
        #expect(suffix.hasPrefix("Now I'll say Chinese."))
    }

    @Test @MainActor func finishedCanArriveBeforeBufferedTerminalAppliesQueuedEvents() async throws {
        let trace = try loadTraceFixture("terminal-trace-long-tail-correction.json")
        let terminal = BufferedTerminalBuffer()
        let harness = makeHarness(
            transport: terminal.transport,
            bulkInserter: BulkInsertProxy { terminal.paste($0) },
            terminalText: { terminal.snapshot() }
        )
        var submittedAtFinish: String?
        harness.session.onEvent = { event in
            if case .finished = event {
                harness.finishCount += 1
                submittedAtFinish = terminal.snapshot()
            }
        }

        try await replay(trace: trace, into: harness.session, stt: harness.stt)

        let snapshot = try #require(submittedAtFinish)
        #expect(snapshot != trace.finalText)

        terminal.flushAll()
        #expect(terminal.snapshot() == trace.finalText)
        #expect(harness.finishCount == 1)
    }

    // MARK: - Helpers

    @MainActor
    private func makeHarness(
        transport: CGEventTransport,
        bulkInserter: (any ClipboardPasting)?,
        terminalText: @escaping @MainActor () -> String
    ) -> SessionHarness {
        let queue = DispatchQueue(label: "test.cg.inject.regression")
        let stt = MockSttSession()
        let audio = MockAudioCapture()
        let injector = CGEventInjector(
            eventQueue: queue,
            transport: transport,
            bulkInserter: bulkInserter
        )
        let session = DictationSession(sttSession: stt, textInjector: injector, audioCapture: audio)
        let box = FinishCounterBox()
        session.onEvent = { event in
            if case .finished = event {
                box.count += 1
            }
        }
        return SessionHarness(
            session: session,
            stt: stt,
            eventQueue: queue,
            finishCounter: box,
            terminalText: terminalText
        )
    }

    @MainActor
    private func replay(
        trace: TranscriptTraceFixture,
        into session: DictationSession,
        stt: MockSttSession
    ) async throws {
        let updates = trace.updates
        let final = try #require(updates.last)
        let nonFinal = updates.dropLast()

        session.start()
        for update in nonFinal {
            stt.onUpdate?(update.asTranscriptUpdate)
            await Task.yield()
        }

        _ = session.stop()
        stt.onUpdate?(final.asTranscriptUpdate)
        await Task.yield()
    }

    private func loadTraceFixture(_ name: String) throws -> TranscriptTraceFixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures")
            .appendingPathComponent(name)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(TranscriptTraceFixture.self, from: data)
    }
}

// MARK: - Fixture model

private struct TranscriptTraceFixture: Decodable {
    let finalText: String
    let updates: [TraceUpdate]
}

private struct TraceUpdate: Decodable {
    let kind: TranscriptUpdateKind
    let text: String

    var asTranscriptUpdate: TranscriptUpdate {
        TranscriptUpdate(kind: kind, text: text)
    }
}

// MARK: - Harness

@MainActor
private struct SessionHarness {
    let session: DictationSession
    let stt: MockSttSession
    let eventQueue: DispatchQueue
    let finishCounter: FinishCounterBox
    let terminalText: @MainActor () -> String

    var finishCount: Int {
        get { finishCounter.count }
        nonmutating set { finishCounter.count = newValue }
    }

    func drainInjector() {
        eventQueue.sync {}
    }
}

@MainActor
private final class FinishCounterBox {
    var count = 0
}

// MARK: - Terminal models

@MainActor
private final class BulkInsertProxy: ClipboardPasting {
    private let handler: @Sendable (String) -> Void

    init(handler: @escaping @Sendable (String) -> Void) {
        self.handler = handler
    }

    func pasteViaClipboard(_ text: String) {
        handler(text)
    }
}

private final class ImmediateTerminalBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = ""

    lazy var transport: CGEventTransport = CGEventTransport(
        postText: { [weak self] text, shouldContinue in
            guard let self, shouldContinue() else { return }
            self.type(text)
        },
        postBackspaces: { [weak self] count, shouldContinue in
            guard let self, shouldContinue() else { return }
            self.backspace(count)
        }
    )

    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        return buffer
    }

    func paste(_ text: String) {
        type(text)
    }

    private func type(_ text: String) {
        lock.lock()
        buffer += text
        lock.unlock()
    }

    private func backspace(_ count: Int) {
        lock.lock()
        defer { lock.unlock() }
        for _ in 0..<count where !buffer.isEmpty {
            buffer.removeLast()
        }
    }
}

private final class BufferedTerminalBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = ""
    private var pending: [() -> Void] = []

    lazy var transport: CGEventTransport = CGEventTransport(
        postText: { [weak self] text, shouldContinue in
            guard let self, shouldContinue() else { return }
            self.enqueue { self.type(text) }
        },
        postBackspaces: { [weak self] count, shouldContinue in
            guard let self, shouldContinue() else { return }
            self.enqueue { self.backspace(count) }
        }
    )

    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        return buffer
    }

    func flushAll() {
        while true {
            let op: (() -> Void)? = {
                lock.lock()
                defer { lock.unlock() }
                guard !pending.isEmpty else { return nil }
                return pending.removeFirst()
            }()
            guard let op else { return }
            op()
        }
    }

    func paste(_ text: String) {
        enqueue { self.type(text) }
    }

    private func enqueue(_ operation: @escaping () -> Void) {
        lock.lock()
        pending.append(operation)
        lock.unlock()
    }

    private func type(_ text: String) {
        lock.lock()
        buffer += text
        lock.unlock()
    }

    private func backspace(_ count: Int) {
        lock.lock()
        defer { lock.unlock() }
        for _ in 0..<count where !buffer.isEmpty {
            buffer.removeLast()
        }
    }
}
