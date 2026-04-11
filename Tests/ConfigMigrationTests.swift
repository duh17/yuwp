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
        #expect(config.audioInputSelection == .systemDefault)
        #expect(config.serverMode == .localhost)
        #expect(config.serverPort == 9748)
        #expect(config.transcriptionModel == "mlx-community/Qwen3-ASR-0.6B-4bit")
        #expect(config.batchCommitEnabled)
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
        config.audioInputSelection = .device(uid: "test-mic")
        config.serverMode = .allInterfaces
        config.serverPort = 8899
        config.transcriptionModel = "mlx-community/Qwen3-ASR-1.7B-bf16"
        config.batchCommitEnabled = false
        config.saveRecordings = true
        config.setRecordingsDir(FileManager.default.temporaryDirectory.appendingPathComponent("yuwp-tests-recordings", isDirectory: true))

        #expect(config.dictationBinding == .optionSpace)
        #expect(config.dictationInteractionMode == .pushToTalk)
        #expect(config.audioInputSelection == .device(uid: "test-mic"))
        #expect(config.serverMode == .allInterfaces)
        #expect(config.serverPort == 8899)
        #expect(config.transcriptionModel == "mlx-community/Qwen3-ASR-1.7B-bf16")
        #expect(!config.batchCommitEnabled)
        #expect(config.saveRecordings)
        #expect(!config.usesDefaultRecordingsDir)
        #expect(config.recordingsDir == FileManager.default.temporaryDirectory.appendingPathComponent("yuwp-tests-recordings", isDirectory: true).standardizedFileURL)

        config.resetRecordingsDir()
        #expect(config.usesDefaultRecordingsDir)
        #expect(config.recordingsDir == config.defaultRecordingsDir)
    }

    @Test @MainActor func readsLegacyModelKeys() {
        let (defaults, suiteName) = makeDefaults()
        defer { UserDefaults().removePersistentDomain(forName: suiteName) }

        defaults.set("legacy-streaming", forKey: "streamingModel")
        defaults.set("legacy-batch", forKey: "batchModel")
        defaults.set(false, forKey: "batchRetranscribeEnabled")

        let config = Config(defaults: defaults)
        #expect(config.transcriptionModel == "legacy-streaming")
        #expect(!config.batchCommitEnabled)
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
