import Testing
@testable import Yuwp

@Suite("Modifier hotkey tracker")
struct ModifierHotkeyTrackerTests {
    @Test func matchingModifierTogglesPressedThenReleased() {
        var tracker = ModifierHotkeyTracker(binding: KeyBinding(keyCode: 62, modifiers: 0))!

        #expect(tracker.handleFlagsChanged(keyCode: 62, timestamp: 100) == .pressed)
        #expect(tracker.handleFlagsChanged(keyCode: 62, timestamp: 100.05) == .released)
    }

    @Test func doubleTapModifierEmitsSinglePressOnSecondTap() {
        var tracker = ModifierHotkeyTracker(binding: KeyBinding(keyCode: 61, modifiers: 0, activation: .doubleTap))!

        #expect(tracker.handleFlagsChanged(keyCode: 61, timestamp: 100) == nil)
        #expect(tracker.handleFlagsChanged(keyCode: 61, timestamp: 100.05) == nil)
        #expect(tracker.handleFlagsChanged(keyCode: 61, timestamp: 100.20) == nil)
        #expect(tracker.handleFlagsChanged(keyCode: 61, timestamp: 100.25) == .pressed)
    }

    @Test func ignoresOtherKeysAndNonModifierBindings() {
        var tracker = ModifierHotkeyTracker(binding: KeyBinding(keyCode: 62, modifiers: 0))!
        #expect(tracker.handleFlagsChanged(keyCode: 59, timestamp: 100) == nil)

        let comboTracker = ModifierHotkeyTracker(binding: .ctrlBacktick)
        #expect(comboTracker == nil)
    }
}
