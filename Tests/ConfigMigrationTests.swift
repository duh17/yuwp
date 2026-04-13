import Foundation
import Testing
@testable import Yuwp

@Suite("Config")
struct ConfigMigrationTests {

    @Test @MainActor func defaultsToCtrlBacktickLocalhostAndDefaultPort() {
        let (defaults, suiteName) = makeDefaults()
        defer { UserDefaults().removePersistentDomain(forName: suiteName) }

        let config = Config(defaults: defaults)

        #expect(config.dictationBinding == .ctrlBacktick)
        #expect(config.audioInputSelection == .systemDefault)
        #expect(config.serverMode == .localhost)
        #expect(config.serverPort == 9748)
        #expect(config.transcriptionModel == "mlx-community/Qwen3-ASR-0.6B-4bit")
        #expect(config.batchCommitEnabled)
        #expect(!config.saveRecordings)
        #expect(config.usesDefaultRecordingsDir)
        #expect(config.recordingsDir == config.defaultRecordingsDir)
        #expect(config.micPanelAnimation == .default)
        #expect(config.startChime == .default)
        #expect(config.stopChime == .default)
    }

    @Test @MainActor func storesDictationBindingAndServerMode() {
        let (defaults, suiteName) = makeDefaults()
        defer { UserDefaults().removePersistentDomain(forName: suiteName) }

        let config = Config(defaults: defaults)
        config.dictationBinding = KeyBinding(keyCode: 61, modifiers: 0, activation: .doubleTap)
        config.audioInputSelection = .device(uid: "test-mic")
        config.serverMode = .allInterfaces
        config.serverPort = 8899
        config.transcriptionModel = "mlx-community/Qwen3-ASR-1.7B-bf16"
        config.batchCommitEnabled = false
        config.saveRecordings = true
        config.setRecordingsDir(FileManager.default.temporaryDirectory.appendingPathComponent("yuwp-tests-recordings", isDirectory: true))

        #expect(config.dictationBinding == KeyBinding(keyCode: 61, modifiers: 0, activation: .doubleTap))
        #expect(config.audioInputSelection == .device(uid: "test-mic"))
        #expect(config.serverMode == .allInterfaces)
        #expect(config.serverPort == 8899)
        #expect(config.transcriptionModel == "mlx-community/Qwen3-ASR-1.7B-bf16")
        #expect(!config.batchCommitEnabled)
        #expect(config.saveRecordings)
        #expect(!config.usesDefaultRecordingsDir)
        #expect(config.recordingsDir == FileManager.default.temporaryDirectory.appendingPathComponent("yuwp-tests-recordings", isDirectory: true).standardizedFileURL)

        let customAnimation = MicPanelAnimationConfig(
            selection: .custom,
            custom: MicPanelAnimationCustom.default
        )
        config.micPanelAnimation = customAnimation
        #expect(config.micPanelAnimation == customAnimation)

        let customAsset = ImportedSoundAsset(relativePath: "test.wav", displayName: "Test")
        let customStartChime = DictationChimeConfig(selection: .custom, customAsset: customAsset)
        let customStopChime = DictationChimeConfig(selection: .soft, customAsset: nil)
        config.startChime = customStartChime
        config.stopChime = customStopChime
        #expect(config.startChime == customStartChime)
        #expect(config.stopChime == customStopChime)

        config.resetRecordingsDir()
        #expect(config.usesDefaultRecordingsDir)
        #expect(config.recordingsDir == config.defaultRecordingsDir)
    }

    @Test @MainActor func invalidDoubleTapComboFallsBackToSinglePress() {
        let (defaults, suiteName) = makeDefaults()
        defer { UserDefaults().removePersistentDomain(forName: suiteName) }

        defaults.set(50, forKey: "dictationBindingKeyCode")
        defaults.set(0x40000, forKey: "dictationBindingModifiers")
        defaults.set("doubleTap", forKey: "dictationBindingActivation")

        let config = Config(defaults: defaults)
        #expect(config.dictationBinding == .ctrlBacktick)
    }

    @Test @MainActor func readsPriorModelKeys() {
        let (defaults, suiteName) = makeDefaults()
        defer { UserDefaults().removePersistentDomain(forName: suiteName) }

        defaults.set("saved-streaming", forKey: "streamingModel")
        defaults.set("saved-batch", forKey: "batchModel")
        defaults.set(false, forKey: "batchRetranscribeEnabled")

        let config = Config(defaults: defaults)
        #expect(config.transcriptionModel == "saved-streaming")
        #expect(!config.batchCommitEnabled)
    }

    @Test @MainActor func keyBindingDescriptionsAreReadable() {
        #expect(KeyBinding.ctrlBacktick.description == "Ctrl+`")
        #expect(KeyBinding.optionSpace.description == "⌥+Space")
        #expect(KeyBinding.commandShiftD.description == "⌘+⇧+D")
        #expect(KeyBinding(keyCode: 62, modifiers: 0).description == "Right Ctrl")
        #expect(KeyBinding(keyCode: 61, modifiers: 0, activation: .doubleTap).description == "Right ⌥ (double tap)")
    }

    private func makeDefaults() -> (UserDefaults, String) {
        let suiteName = "yuwp.tests.config.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return (defaults, suiteName)
    }
}
