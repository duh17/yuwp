import Testing
@testable import Yuwp

@Suite("Dictation hotkey behavior")
struct DictationHotkeyBehaviorTests {
    @Test func pressStartsAndStops() {
        var behavior = DictationHotkeyBehavior()

        let start = behavior.handle(
            phase: .pressed,
            isSessionActive: false
        )
        #expect(start == .start)

        let stop = behavior.handle(
            phase: .pressed,
            isSessionActive: true
        )
        #expect(stop == .stop)
    }

    @Test func releaseIsIgnored() {
        var behavior = DictationHotkeyBehavior()

        let action = behavior.handle(
            phase: .released,
            isSessionActive: true
        )
        #expect(action == .none)
    }
}
