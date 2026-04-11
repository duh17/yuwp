import AppKit
import CoreGraphics
import Foundation

typealias CGEventContinuationCheck = @Sendable () -> Bool

struct CGEventTransport: Sendable {
    let postText: @Sendable (_ text: String, _ shouldContinue: CGEventContinuationCheck) -> Void
    let postBackspaces: @Sendable (_ count: Int, _ shouldContinue: CGEventContinuationCheck) -> Void
}

private final class CGEventQueueState: @unchecked Sendable {
    var renderedText = ""
}

private final class CGEventCancellationState: @unchecked Sendable {
    private let lock = NSLock()
    private var generation = 0

    func snapshot() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return generation
    }

    func invalidate() {
        lock.lock()
        generation += 1
        lock.unlock()
    }

    func isCurrent(_ snapshot: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return generation == snapshot
    }
}

/// Injects text into any focused app by simulating keyboard events.
///
/// Works for terminals and other apps that don't expose AX text editing.
/// Streams via a character-diff: backspaces removed chars, types new suffix.
/// Partial updates are paced on a dedicated queue so terminal typing doesn't
/// block the main actor while the dictation bubble is animating.
@MainActor
final class CGEventInjector: TextInjecting {

    // MARK: - TextInjecting

    let surfaceMode: DictationSurfaceMode = .terminal
    private(set) var targetPosition: NSPoint

    func captureTarget() {
        // No-op — screen point captured at init time by TextInjectorFactory.
    }

    /// Stream partial text via diff: backspace removed chars, type new suffix.
    func inject(_ text: String) {
        let queue = eventQueue
        let queueState = self.queueState
        let transport = self.transport
        let cancellation = self.cancellation
        let generation = cancellation.snapshot()
        queue.async {
            Self.applyTargetText(
                text,
                queueState: queueState,
                transport: transport,
                cancellation: cancellation,
                generation: generation
            )
        }
    }

    /// Commit the final text: replace any streamed partial with the corrected version.
    func commit(_ text: String) {
        let queueState = self.queueState
        let transport = self.transport
        let cancellation = self.cancellation
        let generation = cancellation.snapshot()
        let previousText = eventQueue.sync {
            let previousText = queueState.renderedText
            Self.applyTargetText(
                text,
                queueState: queueState,
                transport: transport,
                cancellation: cancellation,
                generation: generation
            )
            queueState.renderedText = ""
            return previousText
        }

        if previousText == text {
            yuwpLog("CGEvent commit: text unchanged (\(text.count) chars)")
        } else {
            yuwpLog("CGEvent commit: replaced \(previousText.count) chars with \(text.count) chars")
        }
    }

    func release() {
        cancellation.invalidate()
        let queueState = self.queueState
        eventQueue.async {
            queueState.renderedText = ""
        }
    }

    // MARK: - Init

    init(screenPoint: NSPoint = .zero) {
        self.targetPosition = screenPoint
        self.eventQueue = DispatchQueue(label: "yuwp.cg-event-injector", qos: .userInitiated)
        self.transport = Self.liveTransport
    }

    init(
        screenPoint: NSPoint = .zero,
        eventQueue: DispatchQueue,
        transport: CGEventTransport
    ) {
        self.targetPosition = screenPoint
        self.eventQueue = eventQueue
        self.transport = transport
    }

    // MARK: - Diff (pure, testable)

    /// Compute the edit needed to replace `old` with `new`.
    /// Returns (backspaces to delete, new suffix to type).
    nonisolated static func diff(old: String, new: String) -> (backspaces: Int, suffix: String) {
        let commonLen = zip(old, new).prefix(while: { $0 == $1 }).count
        let backspaces = old.count - commonLen
        let suffix = String(new.dropFirst(commonLen))
        return (backspaces, suffix)
    }

    // MARK: - Private

    private let eventQueue: DispatchQueue
    private let queueState = CGEventQueueState()
    private let cancellation = CGEventCancellationState()
    private let transport: CGEventTransport

    nonisolated private static let liveTransport = CGEventTransport(
        postText: { text, shouldContinue in
            guard !text.isEmpty else { return }
            for character in text {
                guard shouldContinue() else { return }
                postUnicodeCharacter(character)
                Thread.sleep(forTimeInterval: interKeyDelay)
            }
        },
        postBackspaces: { count, shouldContinue in
            guard count > 0 else { return }
            for _ in 0..<count {
                guard shouldContinue() else { return }
                postBackspace()
                Thread.sleep(forTimeInterval: interKeyDelay)
            }
        }
    )

    nonisolated private static func applyTargetText(
        _ text: String,
        queueState: CGEventQueueState,
        transport: CGEventTransport,
        cancellation: CGEventCancellationState,
        generation: Int
    ) {
        guard cancellation.isCurrent(generation) else { return }
        let (backspaces, suffix) = diff(old: queueState.renderedText, new: text)
        transport.postBackspaces(backspaces) {
            cancellation.isCurrent(generation)
        }
        transport.postText(suffix) {
            cancellation.isCurrent(generation)
        }
        guard cancellation.isCurrent(generation) else { return }
        queueState.renderedText = text
    }

    nonisolated private static func postUnicodeCharacter(_ character: Character) {
        var utf16 = Array(String(character).utf16)
        let len = utf16.count

        if let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true) {
            down.keyboardSetUnicodeString(stringLength: len, unicodeString: &utf16)
            down.post(tap: .cgSessionEventTap)
        }
        if let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false) {
            up.keyboardSetUnicodeString(stringLength: len, unicodeString: &utf16)
            up.post(tap: .cgSessionEventTap)
        }
    }

    nonisolated private static func postBackspace() {
        let backspaceKeyCode: CGKeyCode = 51
        if let down = CGEvent(keyboardEventSource: nil, virtualKey: backspaceKeyCode, keyDown: true) {
            down.post(tap: .cgSessionEventTap)
        }
        if let up = CGEvent(keyboardEventSource: nil, virtualKey: backspaceKeyCode, keyDown: false) {
            up.post(tap: .cgSessionEventTap)
        }
    }

    nonisolated private static let interKeyDelay: TimeInterval = 0.0015
}
