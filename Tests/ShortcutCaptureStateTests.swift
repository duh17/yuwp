import Testing
@testable import Yuwp

@Suite("Shortcut capture state")
struct ShortcutCaptureStateTests {
    @Test func capturesModifierOnlyBindingAfterTimeout() {
        var state = ShortcutCaptureState()

        let t0 = 100.0
        let armed = state.handleFlagsChanged(keyCode: 62, timestamp: t0)
        #expect(armed == .none)
        #expect(state.pendingModifierKeyCode == 62)

        let released = state.handleFlagsChanged(keyCode: 62, timestamp: t0 + 0.05)
        #expect(released == .none)
        #expect(state.pendingModifierKeyCode == nil)

        let captured = state.resolveTimeout(at: t0 + KeyBindingTiming.doubleTapTimeout + 0.06)
        #expect(captured == .captured(KeyBinding(keyCode: 62, modifiers: 0)))
    }

    @Test func capturesModifierDoubleTapBinding() {
        var state = ShortcutCaptureState()

        let t0 = 100.0
        _ = state.handleFlagsChanged(keyCode: 61, timestamp: t0)
        _ = state.handleFlagsChanged(keyCode: 61, timestamp: t0 + 0.05)
        _ = state.handleFlagsChanged(keyCode: 61, timestamp: t0 + 0.20)
        let captured = state.handleFlagsChanged(keyCode: 61, timestamp: t0 + 0.25)

        #expect(captured == .captured(KeyBinding(keyCode: 61, modifiers: 0, activation: .doubleTap)))
    }

    @Test func capturesComboAfterModifierPress() {
        var state = ShortcutCaptureState()

        _ = state.handleFlagsChanged(keyCode: 59, timestamp: 100)
        let captured = state.handleKeyDown(keyCode: 50, modifiers: 0x40000, timestamp: 100.05)

        #expect(captured == .captured(.ctrlBacktick))
        #expect(state.pendingModifierKeyCode == nil)
    }

    @Test func rejectsBareNonModifierKeys() {
        var state = ShortcutCaptureState()

        let result = state.handleKeyDown(keyCode: 2, modifiers: 0, timestamp: 100)

        #expect(result == .invalid)
        #expect(state.pendingModifierKeyCode == nil)
    }
}
