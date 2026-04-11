import Testing
@testable import Yuwp

@Suite("TextInjectorFactory strategy")
@MainActor
struct TextInjectorFactoryTests {
    @Test func noFocusedElementFallsBackToClipboard() {
        #expect(
            TextInjectorFactory.decideStrategy(
                hasFocusedElement: false,
                role: nil,
                hasReadableSelectionRange: false,
                canWriteSelectedText: false
            ) == .clipboard
        )
    }

    @Test func nonEditableRoleUsesCgEvent() {
        #expect(
            TextInjectorFactory.decideStrategy(
                hasFocusedElement: true,
                role: "AXButton",
                hasReadableSelectionRange: false,
                canWriteSelectedText: false
            ) == .cgEvent
        )
    }

    @Test func missingSelectionRangeUsesClipboard() {
        #expect(
            TextInjectorFactory.decideStrategy(
                hasFocusedElement: true,
                role: "AXTextField",
                hasReadableSelectionRange: false,
                canWriteSelectedText: false
            ) == .clipboard
        )
    }

    @Test func failedWriteProbeUsesClipboard() {
        #expect(
            TextInjectorFactory.decideStrategy(
                hasFocusedElement: true,
                role: "AXTextArea",
                hasReadableSelectionRange: true,
                canWriteSelectedText: false
            ) == .clipboard
        )
    }

    @Test func editableWritableRoleUsesAxInjector() {
        #expect(
            TextInjectorFactory.decideStrategy(
                hasFocusedElement: true,
                role: "AXSearchField",
                hasReadableSelectionRange: true,
                canWriteSelectedText: true
            ) == .ax
        )
    }

    @Test func missingRoleUsesCgEvent() {
        #expect(
            TextInjectorFactory.decideStrategy(
                hasFocusedElement: true,
                role: nil,
                hasReadableSelectionRange: true,
                canWriteSelectedText: true
            ) == .cgEvent
        )
    }
}
