import Testing
import ApplicationServices
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

    @Test func failedWriteProbeUsesCgEvent() {
        #expect(
            TextInjectorFactory.decideStrategy(
                hasFocusedElement: true,
                role: "AXTextArea",
                hasReadableSelectionRange: true,
                canWriteSelectedText: false
            ) == .cgEvent
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

    @Test func axCapabilityCheckUsesValueAttributeAndPreservesSelectionState() {
        let element = AXUIElementCreateSystemWide()
        var observedAttribute: String?

        let canUseAX = TextInjectorFactory.supportsAXTextInjection(
            focused: element,
            role: "AXTextField",
            hasReadableSelectionRange: true,
            attributeIsSettable: { _, attribute in
                observedAttribute = attribute as String
                return true
            }
        )

        #expect(canUseAX)
        #expect(observedAttribute == (kAXValueAttribute as String))
    }

    @Test func axCapabilityCheckFallsBackWhenValueAttributeIsNotSettable() {
        let element = AXUIElementCreateSystemWide()

        #expect(
            TextInjectorFactory.supportsAXTextInjection(
                focused: element,
                role: "AXTextArea",
                hasReadableSelectionRange: true,
                attributeIsSettable: { _, _ in false }
            ) == false
        )
    }
}
