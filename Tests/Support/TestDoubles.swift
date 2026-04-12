import Foundation
import Testing
@testable import Yuwp

// MARK: - Mock STT Session

final class MockSttSession: SttSession, @unchecked Sendable {
    var onUpdate: ((TranscriptUpdate) -> Void)?
    var onError: ((String) -> Void)?
    var debugSessionID: String?

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
        onUpdate?(TranscriptUpdate(kind: .partial, text: text))
    }

    func simulateSegmentCommit(_ text: String) {
        onUpdate?(TranscriptUpdate(kind: .segmentCommit, text: text))
    }

    func simulateFinal(_ text: String) {
        onUpdate?(TranscriptUpdate(kind: .final, text: text))
    }

    func simulateError(_ msg: String) {
        onError?(msg)
    }
}

// MARK: - Mock Audio Capture

final class MockAudioCapture: AudioCapturing, @unchecked Sendable {
    var onAudioLevel: (@Sendable (Float) -> Void)?
    var onWarning: (@Sendable (AudioCaptureWarning) -> Void)?

    var startCallCount = 0
    var stopCallCount = 0
    var onBufferCallback: ((@Sendable (Data) -> Void))?
    var stopReturnData: Data?
    var startShouldSucceed = true

    @discardableResult
    func start(onBuffer: @escaping @Sendable (Data) -> Void) -> Bool {
        startCallCount += 1
        if startShouldSucceed {
            onBufferCallback = onBuffer
        }
        return startShouldSucceed
    }

    func stop() -> Data? {
        stopCallCount += 1
        onBufferCallback = nil
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
    var surfaceMode: DictationSurfaceMode = .nativeField
    var targetPosition: NSPoint = .zero

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

    var presentations: [DictationPresentationState] {
        events.compactMap { event in
            guard case .presentation(let state) = event else { return nil }
            return state
        }
    }

    func handler(_ event: DictationEvent) {
        events.append(event)
    }
}
