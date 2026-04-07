import Foundation
import Testing
@testable import Yuwp

@Suite("DictationSession")
@MainActor
struct DictationSessionTests {

    // MARK: - Helpers

    private func makeSession() -> (
        session: DictationSession,
        stt: MockSttSession,
        audio: MockAudioCapture,
        injector: MockTextInjector,
        events: EventCollector
    ) {
        let stt = MockSttSession()
        let audio = MockAudioCapture()
        let injector = MockTextInjector()
        let events = EventCollector()

        let session = DictationSession(
            sttSession: stt,
            textInjector: injector,
            audioCapture: audio
        )
        session.onEvent = { events.handler($0) }

        return (session, stt, audio, injector, events)
    }

    // MARK: - Start

    @Test func startCapturesTargetAndBeginsSession() {
        let (session, stt, audio, injector, _) = makeSession()

        session.start()

        #expect(session.isActive)
        #expect(injector.captureCallCount == 1)
        #expect(stt.beginCallCount == 1)
        #expect(audio.startCallCount == 1)
    }

    @Test func startIsIdempotent() {
        let (session, stt, _, _, _) = makeSession()

        session.start()
        session.start() // second call should no-op

        #expect(stt.beginCallCount == 1)
    }

    // MARK: - Stop

    @Test func stopEndsSessionAndReturnsAudio() {
        let (session, stt, audio, _, _) = makeSession()
        audio.stopReturnData = Data([1, 2, 3, 4])

        session.start()
        let pcm = session.stop()

        #expect(!session.isActive)
        #expect(stt.endCallCount == 1)
        #expect(audio.stopCallCount == 1)
        #expect(pcm?.count == 4)
    }

    @Test func stopIsGuardedWhenNotActive() {
        let (session, stt, _, _, _) = makeSession()

        let pcm = session.stop()

        #expect(pcm == nil)
        #expect(stt.endCallCount == 0)
    }

    // MARK: - Partial Results

    @Test func partialResultInjectsText() async {
        let (session, stt, _, injector, _) = makeSession()
        session.start()

        stt.simulatePartial("Hello world")
        await Task.yield()

        #expect(injector.injectCallCount >= 1)
    }

    @Test func partialResultFiltersEmptyText() async {
        let (session, stt, _, injector, _) = makeSession()
        session.start()

        stt.simulatePartial("")
        await Task.yield()
        stt.simulatePartial("   ")
        await Task.yield()
        stt.simulatePartial("None")
        await Task.yield()
        stt.simulatePartial("none")
        await Task.yield()

        #expect(injector.injectCallCount == 0)
    }

    @Test func partialResultEmitsTranscriptEventWhenNotLiveInjecting() async {
        let (session, stt, _, injector, events) = makeSession()
        injector.isLiveInjecting = false
        session.start()

        stt.simulatePartial("Hello")
        await Task.yield()

        let transcriptEvents = events.events.filter {
            if case .partialTranscript = $0 { return true }
            return false
        }
        #expect(!transcriptEvents.isEmpty)
    }

    @Test func partialResultEmitsLiveInjectionWhenVerified() async {
        let (session, stt, _, injector, events) = makeSession()
        injector.isLiveInjecting = true
        injector.targetPosition = NSPoint(x: 100, y: 200)
        session.start()

        stt.simulatePartial("Hello")
        await Task.yield()

        let liveEvents = events.events.filter {
            if case .liveInjectionVerified = $0 { return true }
            return false
        }
        #expect(!liveEvents.isEmpty)
    }

    // MARK: - Final Result

    @Test func finalResultCommitsTextAndFinishes() async {
        let (session, stt, _, injector, events) = makeSession()
        session.start()
        _ = session.stop()

        stt.simulateFinal("Hello world")
        await Task.yield()

        #expect(injector.commitCallCount == 1)
        #expect(injector.lastCommitted == "Hello world")
        #expect(injector.releaseCallCount == 1)
        #expect(events.events.contains(.finished))
    }

    @Test func finalResultAfterStopReleasesInjector() async {
        let (session, stt, _, injector, _) = makeSession()
        session.start()
        _ = session.stop()

        stt.simulateFinal("Done")
        await Task.yield()

        #expect(injector.commitCallCount == 1)
        #expect(injector.releaseCallCount == 1)
    }

    // MARK: - Audio Start Failure

    @Test func startAbortsWhenAudioCaptureFails() {
        let (session, stt, audio, injector, events) = makeSession()
        audio.startShouldSucceed = false

        session.start()

        // Session should have aborted
        #expect(!session.isActive)
        #expect(stt.endCallCount == 1) // stt session cleaned up
        #expect(injector.releaseCallCount == 1) // injector released
        #expect(events.events.contains(.finished))
    }

    // MARK: - Audio Forwarding

    @Test func audioBuffersForwardedToSttSession() {
        let (session, stt, audio, _, _) = makeSession()
        session.start()

        let pcm = Data([0, 1, 2, 3])
        audio.simulateBuffer(pcm)

        #expect(stt.feedCallCount == 1)
        #expect(stt.feedBytes == 4)
    }

    @Test func audioLevelForwardedAsEvent() async {
        let (session, _, audio, _, events) = makeSession()
        session.start()

        audio.simulateAudioLevel(0.75)
        await Task.yield()

        #expect(events.events.contains(.audioLevel(0.75)))
    }
}
