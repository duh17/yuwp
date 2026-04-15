import Foundation
import Testing
@testable import Yuwp

@Suite("CGEventInjector diff algorithm")
struct CGEventInjectorTests {
    private final class OperationRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var operations: [String] = []

        func append(_ operation: String) {
            lock.lock()
            operations.append(operation)
            lock.unlock()
        }

        func snapshot() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            return operations
        }
    }

    @MainActor
    private final class PasteRecorder: ClipboardPasting {
        private(set) var pasted: [String] = []

        func pasteViaClipboard(_ text: String) {
            pasted.append(text)
        }
    }

    @Test func emptyToText() {
        let (bs, suffix) = CGEventInjector.diff(old: "", new: "hello")
        #expect(bs == 0)
        #expect(suffix == "hello")
    }

    @Test func appendOnly() {
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "hello world")
        #expect(bs == 0)
        #expect(suffix == " world")
    }

    @Test func fullReplacement() {
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "world")
        #expect(bs == 5)
        #expect(suffix == "world")
    }

    @Test func partialReplacement() {
        // "hello" → "help": common prefix "hel" (3), removes "lo" (2 backspaces), types "p"
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "help")
        #expect(bs == 2)
        #expect(suffix == "p")
    }

    @Test func identical() {
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "hello")
        #expect(bs == 0)
        #expect(suffix == "")
    }

    @Test func deleteAll() {
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "")
        #expect(bs == 5)
        #expect(suffix == "")
    }

    @Test func bothEmpty() {
        let (bs, suffix) = CGEventInjector.diff(old: "", new: "")
        #expect(bs == 0)
        #expect(suffix == "")
    }

    @Test func streamingPartials() {
        // Simulates incremental streaming: "" → "Hel" → "Hell" → "Hello"
        var current = ""

        let (bs1, s1) = CGEventInjector.diff(old: current, new: "Hel")
        #expect(bs1 == 0)
        #expect(s1 == "Hel")
        current = "Hel"

        let (bs2, s2) = CGEventInjector.diff(old: current, new: "Hell")
        #expect(bs2 == 0)
        #expect(s2 == "l")
        current = "Hell"

        let (bs3, s3) = CGEventInjector.diff(old: current, new: "Hello")
        #expect(bs3 == 0)
        #expect(s3 == "o")
    }

    @Test func batchCorrectionTypo() {
        // "helo wrld" → "Hello world": case difference on first char kills common prefix
        let (bs, suffix) = CGEventInjector.diff(old: "helo wrld", new: "Hello world")
        #expect(bs == 9)
        #expect(suffix == "Hello world")
    }

    @Test func periodSnapCorrection() {
        // Streaming: "hello." → "hello world" (period removed, word added)
        let (bs, suffix) = CGEventInjector.diff(old: "hello.", new: "hello world")
        #expect(bs == 1)   // delete "."
        #expect(suffix == " world")
    }

    @Test func unicodeAppend() {
        let (bs, suffix) = CGEventInjector.diff(old: "你好", new: "你好世界")
        #expect(bs == 0)
        #expect(suffix == "世界")
    }

    @Test func unicodeFullReplacement() {
        // No common prefix between Chinese and ASCII
        let (bs, suffix) = CGEventInjector.diff(old: "你好世界", new: "Hello")
        #expect(bs == 4)
        #expect(suffix == "Hello")
    }

    @Test func singleCharAppend() {
        let (bs, suffix) = CGEventInjector.diff(old: "a", new: "ab")
        #expect(bs == 0)
        #expect(suffix == "b")
    }

    @Test func singleCharReplace() {
        let (bs, suffix) = CGEventInjector.diff(old: "a", new: "b")
        #expect(bs == 1)
        #expect(suffix == "b")
    }

    @Test func commitAfterStreamDiffers() {
        // Full session: stream "helo", then commit "Hello"
        var state = ""

        let (bs1, s1) = CGEventInjector.diff(old: state, new: "helo")
        state = "helo"
        #expect(bs1 == 0)
        #expect(s1 == "helo")

        // Commit with batch-corrected text
        let (bs2, s2) = CGEventInjector.diff(old: state, new: "Hello")
        #expect(bs2 == 4)   // backspace all of "helo" (no common prefix — 'h' != 'H')
        #expect(s2 == "Hello")
    }

    @Test func commitAfterStreamIdentical() {
        var state = ""

        let (bs1, s1) = CGEventInjector.diff(old: state, new: "hello")
        state = "hello"
        #expect(bs1 == 0)
        #expect(s1 == "hello")

        // Batch correction returns same text
        let (bs2, s2) = CGEventInjector.diff(old: state, new: "hello")
        #expect(bs2 == 0)
        #expect(s2 == "")
    }

    @Test @MainActor func injectReturnsBeforeTransportFinishes() {
        let started = DispatchSemaphore(value: 0)
        let unblock = DispatchSemaphore(value: 0)
        let queue = DispatchQueue(label: "test.cg.inject.async")
        let injector = CGEventInjector(
            eventQueue: queue,
            transport: CGEventTransport(
                postText: { _, _ in
                    started.signal()
                    _ = unblock.wait(timeout: .now() + 1)
                },
                postBackspaces: { _, _ in }
            )
        )

        let t0 = Date()
        injector.inject("hello")
        let elapsed = Date().timeIntervalSince(t0)

        #expect(elapsed < 0.05)
        #expect(started.wait(timeout: .now() + 0.5) == .success)
        unblock.signal()
        queue.sync {}
    }

    @Test @MainActor func commitDrainsQueuedPreviewBeforeReturning() {
        let recorder = OperationRecorder()
        let queue = DispatchQueue(label: "test.cg.inject.commit")
        let injector = CGEventInjector(
            eventQueue: queue,
            transport: CGEventTransport(
                postText: { text, shouldContinue in
                    guard shouldContinue() else { return }
                    recorder.append("text:\(text)")
                },
                postBackspaces: { count, shouldContinue in
                    guard count > 0 else { return }
                    guard shouldContinue() else { return }
                    recorder.append("bs:\(count)")
                }
            )
        )

        injector.inject("hel")
        injector.commit("hello")

        #expect(recorder.snapshot() == ["text:hel", "text:lo"])
    }

    @Test @MainActor func releaseCancelsQueuedPreviewBeforeTyping() {
        let recorder = OperationRecorder()
        let started = DispatchSemaphore(value: 0)
        let unblock = DispatchSemaphore(value: 0)
        let queue = DispatchQueue(label: "test.cg.inject.release")
        let injector = CGEventInjector(
            eventQueue: queue,
            transport: CGEventTransport(
                postText: { text, shouldContinue in
                    for character in text {
                        started.signal()
                        _ = unblock.wait(timeout: .now() + 1)
                        guard shouldContinue() else { return }
                        recorder.append("char:\(character)")
                    }
                },
                postBackspaces: { _, _ in }
            )
        )

        injector.inject("hello")
        #expect(started.wait(timeout: .now() + 0.5) == .success)
        injector.release()

        for _ in 0..<5 {
            unblock.signal()
        }
        queue.sync {}

        #expect(recorder.snapshot().isEmpty)
    }

    @Test @MainActor func rewriteCommitUsesBulkPasteInsteadOfTypingReplacementTail() {
        let recorder = OperationRecorder()
        let pasteRecorder = PasteRecorder()
        let queue = DispatchQueue(label: "test.cg.inject.bulk-paste")
        let injector = CGEventInjector(
            eventQueue: queue,
            transport: CGEventTransport(
                postText: { text, shouldContinue in
                    guard shouldContinue() else { return }
                    recorder.append("text:\(text)")
                },
                postBackspaces: { count, shouldContinue in
                    guard count > 0 else { return }
                    guard shouldContinue() else { return }
                    recorder.append("bs:\(count)")
                }
            ),
            bulkInserter: pasteRecorder
        )

        injector.inject("hello world")
        injector.commit("hello there")

        #expect(recorder.snapshot() == ["text:hello world", "bs:5"])
        #expect(pasteRecorder.pasted == ["there"])
    }

    @Test @MainActor func appendOnlyCommitKeepsKeyboardTypingInsteadOfUsingPaste() {
        let recorder = OperationRecorder()
        let pasteRecorder = PasteRecorder()
        let queue = DispatchQueue(label: "test.cg.inject.append-commit")
        let injector = CGEventInjector(
            eventQueue: queue,
            transport: CGEventTransport(
                postText: { text, shouldContinue in
                    guard shouldContinue() else { return }
                    recorder.append("text:\(text)")
                },
                postBackspaces: { count, shouldContinue in
                    guard count > 0 else { return }
                    guard shouldContinue() else { return }
                    recorder.append("bs:\(count)")
                }
            ),
            bulkInserter: pasteRecorder
        )

        injector.inject("hello")
        injector.commit("hello world")

        #expect(recorder.snapshot() == ["text:hello", "text: world"])
        #expect(pasteRecorder.pasted.isEmpty)
    }
}
