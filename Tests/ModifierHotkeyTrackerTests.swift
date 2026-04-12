import Testing
@testable import Yuwp

@Suite("Modifier hotkey tracker")
struct ModifierHotkeyTrackerTests {
    @Test func matchingModifierTogglesPressedThenReleased() {
        var tracker = ModifierHotkeyTracker(binding: KeyBinding(keyCode: 62, modifiers: 0))!

        #expect(tracker.handleFlagsChanged(keyCode: 62) == .pressed)
        #expect(tracker.handleFlagsChanged(keyCode: 62) == .released)
    }

    @Test func ignoresOtherKeysAndNonModifierBindings() {
        var tracker = ModifierHotkeyTracker(binding: KeyBinding(keyCode: 62, modifiers: 0))!
        #expect(tracker.handleFlagsChanged(keyCode: 59) == nil)

        let comboTracker = ModifierHotkeyTracker(binding: .ctrlBacktick)
        #expect(comboTracker == nil)
    }
}
