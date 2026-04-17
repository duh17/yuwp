import AppKit
import Testing
@testable import Yuwp

@Suite("Dictation chimes")
struct DictationChimeTests {
    @Test func mechanicalStartUsesAudibleSound() {
        #expect(DictationChimeRole.start.mechanicalSoundName == NSSound.Name("Funk"))
    }

    @Test @MainActor func missingConfiguredSoundFallsBackToBeep() {
        var didBeep = false
        let player = DictationChimePlayer(
            namedSoundFactory: { _ in nil },
            customSoundFactory: { _ in nil },
            beep: { didBeep = true }
        )

        player.play(.start, config: DictationChimeConfig(selection: .systemDefault, customAsset: nil))

        #expect(didBeep)
    }
}
