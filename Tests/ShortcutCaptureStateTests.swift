import Testing
@testable import Yuwp

@Suite("Shortcut capture state")
struct ShortcutCaptureStateTests {
    @Test func capturesModifierOnlyBindingOnRelease() {
        var state = ShortcutCaptureState()

        let armed = state.handleFlagsChanged(keyCode: 62)
        #expect(armed == .none)
        #expect(state.pendingModifierKeyCode == 62)

        let captured = state.handleFlagsChanged(keyCode: 62)
        #expect(captured == .captured(KeyBinding(keyCode: 62, modifiers: 0)))
        #expect(state.pendingModifierKeyCode == nil)
    }

    @Test func capturesComboAfterModifierPress() {
        var state = ShortcutCaptureState()

        _ = state.handleFlagsChanged(keyCode: 59)
        let captured = state.handleKeyDown(keyCode: 50, modifiers: 0x40000)

        #expect(captured == .captured(.ctrlBacktick))
        #expect(state.pendingModifierKeyCode == nil)
    }

    @Test func rejectsBareNonModifierKeys() {
        var state = ShortcutCaptureState()

        let result = state.handleKeyDown(keyCode: 2, modifiers: 0)

        #expect(result == .invalid)
        #expect(state.pendingModifierKeyCode == nil)
    }
}
