import Foundation
import Testing
@testable import Yuwp

// MARK: - Mock STT Session

final class MockSttSession: SttSession, @unchecked Sendable {
    var onPartial: ((String) -> Void)?
    var onFinal: ((String) -> Void)?
    var onError: ((String) -> Void)?

    var beginCallCount = 0
    var beginLanguage: String?
    var feedCallCount = 0
    var feedBytes = 0
    var endCallCount = 0

    func begin(language: String?) {
        beginCallCount += 1
        beginLanguage = language
    }

    func feedAudio(_ pcmData: Data) {
        feedCallCount += 1
        feedBytes += pcmData.count
    }

    func end() {
        endCallCount += 1
    }

    // Test helpers

    func simulatePartial(_ text: String) {
        onPartial?(text)
    }

    func simulateFinal(_ text: String) {
        onFinal?(text)
    }

    func simulateError(_ msg: String) {
        onError?(msg)
    }
}

// MARK: - Mock Audio Capture

final class MockAudioCapture: AudioCapturing, @unchecked Sendable {
    var onAudioLevel: (@Sendable (Float) -> Void)?

    var startCallCount = 0
    var stopCallCount = 0
    var onBufferCallback: ((@Sendable (Data) -> Void))?
    var stopReturnData: Data?

    func start(onBuffer: @escaping @Sendable (Data) -> Void) {
        startCallCount += 1
        onBufferCallback = onBuffer
    }

    func stop() -> Data? {
        stopCallCount += 1
        return stopReturnData
    }

    func simulateBuffer(_ data: Data) {
        onBufferCallback?(data)
    }

    func simulateAudioLevel(_ level: Float) {
        onAudioLevel?(level)
    }
}

// MARK: - Mock Text Injector

@MainActor
final class MockTextInjector: TextInjecting {
    var targetPosition: NSPoint = .zero
    var isLiveInjecting = false

    var captureCallCount = 0
    var injectCallCount = 0
    var commitCallCount = 0
    var releaseCallCount = 0
    var lastInjected: String?
    var lastCommitted: String?

    func captureTarget() {
        captureCallCount += 1
    }

    func inject(_ text: String) {
        injectCallCount += 1
        lastInjected = text
    }

    func commit(_ text: String) {
        commitCallCount += 1
        lastCommitted = text
    }

    func release() {
        releaseCallCount += 1
    }
}

// MARK: - Event Collector

@MainActor
final class EventCollector {
    var events: [DictationEvent] = []

    func handler(_ event: DictationEvent) {
        events.append(event)
    }
}
