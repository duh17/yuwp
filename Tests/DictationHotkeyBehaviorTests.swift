import Testing
@testable import Yuwp

@Suite("Dictation hotkey behavior")
struct DictationHotkeyBehaviorTests {

    @Test func toggleModePressStartsAndStops() {
        var behavior = DictationHotkeyBehavior()

        let start = behavior.handle(
            phase: .pressed,
            mode: .toggle,
            isSessionActive: false
        )
        #expect(start == .start)

        let stop = behavior.handle(
            phase: .pressed,
            mode: .toggle,
            isSessionActive: true
        )
        #expect(stop == .stop)
    }

    @Test func toggleModeIgnoresRelease() {
        var behavior = DictationHotkeyBehavior()

        let action = behavior.handle(
            phase: .released,
            mode: .toggle,
            isSessionActive: true
        )
        #expect(action == .none)
    }

    @Test func pushToTalkStartsOnPressAndStopsOnRelease() {
        var behavior = DictationHotkeyBehavior()

        let start = behavior.handle(
            phase: .pressed,
            mode: .pushToTalk,
            isSessionActive: false
        )
        #expect(start == .start)
        #expect(behavior.sessionStartedByPushToTalk)

        let stop = behavior.handle(
            phase: .released,
            mode: .pushToTalk,
            isSessionActive: true
        )
        #expect(stop == .stop)
        #expect(!behavior.sessionStartedByPushToTalk)
    }

    @Test func pushToTalkIgnoresRepeatedPressWhileActive() {
        var behavior = DictationHotkeyBehavior()

        _ = behavior.handle(
            phase: .pressed,
            mode: .pushToTalk,
            isSessionActive: false
        )

        let repeated = behavior.handle(
            phase: .pressed,
            mode: .pushToTalk,
            isSessionActive: true
        )
        #expect(repeated == .none)
    }

    @Test func pushToTalkIgnoresReleaseIfSessionWasNotStartedByHold() {
        var behavior = DictationHotkeyBehavior()

        let release = behavior.handle(
            phase: .released,
            mode: .pushToTalk,
            isSessionActive: true
        )

        #expect(release == .none)
    }
}
