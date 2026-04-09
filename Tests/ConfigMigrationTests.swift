import Foundation
import Testing
@testable import Yuwp

@Suite("Config")
struct ConfigMigrationTests {

    @Test @MainActor func defaultsToCtrlBacktickToggleLocalhostAndDefaultPort() {
        let (defaults, suiteName) = makeDefaults()
        defer { UserDefaults().removePersistentDomain(forName: suiteName) }

        let config = Config(defaults: defaults)

        #expect(config.dictationBinding == .ctrlBacktick)
        #expect(config.dictationInteractionMode == .toggle)
        #expect(config.serverMode == .localhost)
        #expect(config.serverPort == 9748)
    }

    @Test @MainActor func storesDictationBindingModeAndServerMode() {
        let (defaults, suiteName) = makeDefaults()
        defer { UserDefaults().removePersistentDomain(forName: suiteName) }

        let config = Config(defaults: defaults)
        config.dictationBinding = .optionSpace
        config.dictationInteractionMode = .pushToTalk
        config.serverMode = .allInterfaces
        config.serverPort = 8899

        #expect(config.dictationBinding == .optionSpace)
        #expect(config.dictationInteractionMode == .pushToTalk)
        #expect(config.serverMode == .allInterfaces)
        #expect(config.serverPort == 8899)
    }

    @Test @MainActor func keyBindingDescriptionsAreReadable() {
        #expect(KeyBinding.ctrlBacktick.description == "Ctrl+`")
        #expect(KeyBinding.optionSpace.description == "⌥+Space")
        #expect(KeyBinding.commandShiftD.description == "⌘+⇧+D")
    }

    private func makeDefaults() -> (UserDefaults, String) {
        let suiteName = "yuwp.tests.config.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return (defaults, suiteName)
    }
}
