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
        #expect(config.streamingModel == "mlx-community/Qwen3-ASR-0.6B-4bit")
        #expect(config.batchModel == "mlx-community/Qwen3-ASR-0.6B-4bit")
        #expect(config.batchRetranscribeEnabled)
        #expect(!config.saveRecordings)
        #expect(config.usesDefaultRecordingsDir)
        #expect(config.recordingsDir == config.defaultRecordingsDir)
    }

    @Test @MainActor func storesDictationBindingModeAndServerMode() {
        let (defaults, suiteName) = makeDefaults()
        defer { UserDefaults().removePersistentDomain(forName: suiteName) }

        let config = Config(defaults: defaults)
        config.dictationBinding = .optionSpace
        config.dictationInteractionMode = .pushToTalk
        config.serverMode = .allInterfaces
        config.serverPort = 8899
        config.streamingModel = "mlx-community/Qwen3-ASR-1.7B-bf16"
        config.batchModel = "/tmp/yuwp/models/final"
        config.batchRetranscribeEnabled = false
        config.saveRecordings = true
        config.setRecordingsDir(FileManager.default.temporaryDirectory.appendingPathComponent("yuwp-tests-recordings", isDirectory: true))

        #expect(config.dictationBinding == .optionSpace)
        #expect(config.dictationInteractionMode == .pushToTalk)
        #expect(config.serverMode == .allInterfaces)
        #expect(config.serverPort == 8899)
        #expect(config.streamingModel == "mlx-community/Qwen3-ASR-1.7B-bf16")
        #expect(config.batchModel == "/tmp/yuwp/models/final")
        #expect(!config.batchRetranscribeEnabled)
        #expect(config.saveRecordings)
        #expect(!config.usesDefaultRecordingsDir)
        #expect(config.recordingsDir == FileManager.default.temporaryDirectory.appendingPathComponent("yuwp-tests-recordings", isDirectory: true).standardizedFileURL)

        config.resetRecordingsDir()
        #expect(config.usesDefaultRecordingsDir)
        #expect(config.recordingsDir == config.defaultRecordingsDir)
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
