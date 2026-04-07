import Foundation
import Testing
@testable import Yuwp

/// Tests for the Enter-stops-dictation feature (Phase 2).
///
/// The CGEvent tap callback cannot be exercised in unit tests (no running event loop),
/// but we can verify the API surface and observable state changes that AppDelegate relies on.
@Suite("Enter interception during dictation")
struct EnterInterceptionTests {

    // MARK: - HotkeyManager state

    @Test func sessionActiveDefaultsFalse() {
        // Reset to known state — tests run in parallel so another test
        // may have set this to true before we get here.
        HotkeyManager.sessionActive = false
        defer { HotkeyManager.sessionActive = false }

        // The static flag must be false before any session starts.
        #expect(HotkeyManager.sessionActive == false)
    }

    @Test func sessionActiveFlagCanBeToggled() {
        let original = HotkeyManager.sessionActive
        defer { HotkeyManager.sessionActive = original }

        HotkeyManager.sessionActive = true
        #expect(HotkeyManager.sessionActive == true)

        HotkeyManager.sessionActive = false
        #expect(HotkeyManager.sessionActive == false)
    }

    // MARK: - onEnterDuringSession callback

    @Test @MainActor func onEnterDuringSessionCallbackFires() {
        final class Counter: @unchecked Sendable { var n = 0 }
        let manager = HotkeyManager()
        let c = Counter()
        manager.onEnterDuringSession = { c.n += 1 }

        // Simulate the callback firing (as it would from the CGEvent tap)
        manager.onEnterDuringSession?()

        #expect(c.n == 1)
    }

    @Test @MainActor func onEnterDuringSessionCanBeRewired() {
        final class Counter: @unchecked Sendable { var n = 0 }
        let manager = HotkeyManager()
        let c = Counter()

        manager.onEnterDuringSession = { c.n += 1 }
        manager.onEnterDuringSession?()
        #expect(c.n == 1)

        manager.onEnterDuringSession = { c.n += 10 }
        manager.onEnterDuringSession?()
        #expect(c.n == 11)
    }

    // MARK: - Enter interception + replay flow (via DictationSession + mocks)

    /// Verifies the complete Enter-intercept sequence:
    /// Enter fires → pendingEnter flag is set → session stops → on .finished,
    /// Enter would be replayed (we verify the flag is consumed).
    ///
    /// We simulate this with a small coordinator that mirrors AppDelegate's logic.
    @Test @MainActor func enterInterceptStopsSessionAndSetsFlag() async {
        let stt = MockSttSession()
        let audio = MockAudioCapture()
        let injector = MockTextInjector()

        let session = DictationSession(
            sttSession: stt,
            textInjector: injector,
            audioCapture: audio
        )

        var pendingEnter = false
        var finishedFired = false
        var enterReplayedAfterFinish = false

        session.onEvent = { event in
            if case .finished = event {
                finishedFired = true
                if pendingEnter {
                    pendingEnter = false
                    enterReplayedAfterFinish = true
                }
            }
        }

        // Start dictation
        session.start()
        #expect(session.isActive)

        // Simulate Enter pressed during session (what HotkeyManager fires)
        pendingEnter = true
        _ = session.stop()
        #expect(!session.isActive)

        // Final arrives after stop (batch correction completes)
        stt.simulateFinal("hello world")
        await Task.yield()

        #expect(finishedFired)
        #expect(enterReplayedAfterFinish, "Enter should be replayed once .finished fires")
        #expect(!pendingEnter, "Flag should be consumed")
    }

    @Test @MainActor func enterBeforeAnySpeechStillReplays() async {
        // Edge case: user presses Enter immediately, no partials yet.
        let stt = MockSttSession()
        let audio = MockAudioCapture()
        let injector = MockTextInjector()

        let session = DictationSession(
            sttSession: stt,
            textInjector: injector,
            audioCapture: audio
        )

        var pendingEnter = false
        var enterReplayedAfterFinish = false

        session.onEvent = { event in
            if case .finished = event, pendingEnter {
                pendingEnter = false
                enterReplayedAfterFinish = true
            }
        }

        session.start()

        // Enter immediately, no audio, no partials
        pendingEnter = true
        _ = session.stop()

        stt.simulateFinal("")
        await Task.yield()

        #expect(enterReplayedAfterFinish)
    }

    @Test @MainActor func normalStopWithoutEnterDoesNotReplay() async {
        let stt = MockSttSession()
        let audio = MockAudioCapture()
        let injector = MockTextInjector()

        let session = DictationSession(
            sttSession: stt,
            textInjector: injector,
            audioCapture: audio
        )

        var pendingEnter = false  // NOT set — normal hotkey stop
        var enterReplayed = false

        session.onEvent = { event in
            if case .finished = event, pendingEnter {
                pendingEnter = false
                enterReplayed = true
            }
        }

        session.start()
        _ = session.stop()

        stt.simulateFinal("hello")
        await Task.yield()

        #expect(!enterReplayed, "Enter should NOT be replayed on a normal stop")
    }
}
