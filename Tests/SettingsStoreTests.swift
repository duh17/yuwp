import ASRIPC
import Foundation
import Testing
@testable import Yuwp

@Suite("SettingsStore")
struct SettingsStoreTests {
    @Test @MainActor func experimentalInsertionTogglesUpdateSnapshotAndCallbacks() {
        let store = SettingsStore(snapshot: makeSnapshot())
        var textFieldValue: Bool?
        var terminalValue: Bool?
        store.onExperimentalDirectTextFieldInsertionChange = { textFieldValue = $0 }
        store.onExperimentalDirectTerminalInsertionChange = { terminalValue = $0 }

        store.setExperimentalDirectTextFieldInsertionEnabled(true)
        store.setExperimentalDirectTerminalInsertionEnabled(true)

        #expect(store.snapshot.experimentalDirectTextFieldInsertionEnabled)
        #expect(store.snapshot.experimentalDirectTerminalInsertionEnabled)
        #expect(textFieldValue == true)
        #expect(terminalValue == true)
    }

    @Test @MainActor func diagnosticLoggingToggleUpdatesSnapshotAndCallback() {
        let store = SettingsStore(snapshot: makeSnapshot())
        var value: Bool?
        store.onDiagnosticLoggingChange = { value = $0 }

        store.setDiagnosticLoggingEnabled(true)

        #expect(store.snapshot.diagnosticLoggingEnabled)
        #expect(value == true)
    }

    @Test @MainActor func dictationLanguageSettingsUpdateSnapshotAndCallbacks() {
        let store = SettingsStore(snapshot: makeSnapshot())
        var mode: DictationLanguageMode?
        var language: String?
        store.onDictationLanguageModeChange = { mode = $0 }
        store.onFixedDictationLanguageChange = { language = $0 }

        store.setDictationLanguageMode(.fixed)
        store.setFixedDictationLanguage("Chinese")

        #expect(store.snapshot.dictationLanguageMode == .fixed)
        #expect(store.snapshot.fixedDictationLanguage == "Chinese")
        #expect(mode == .fixed)
        #expect(language == "Chinese")
    }

    @Test @MainActor func selectingCustomMicPanelSeedsDefaultCustomValues() {
        let store = SettingsStore(snapshot: makeSnapshot())
        var received: MicPanelAnimationConfig?
        store.onMicPanelAnimationChange = { received = $0 }

        store.setMicPanelAnimationSelection(.custom)

        #expect(store.snapshot.micPanelAnimation.selection == .custom)
        #expect(store.snapshot.micPanelAnimation.custom == .default)
        #expect(received?.selection == .custom)
        #expect(received?.custom == .default)
    }

    @Test @MainActor func updatingCustomMicPanelClampsValues() {
        let store = SettingsStore(snapshot: makeSnapshot())

        store.updateMicPanelAnimationCustom {
            $0.smoothingAttack = 9
            $0.glowAlphaScale = -1
        }

        #expect(store.snapshot.micPanelAnimation.selection == .custom)
        #expect(store.snapshot.micPanelAnimation.custom?.smoothingAttack == 1.0)
        #expect(store.snapshot.micPanelAnimation.custom?.glowAlphaScale == 0.0)
    }

    @Test @MainActor func invalidServerPortSetsAlertWithoutCallingHandler() {
        let store = SettingsStore(snapshot: makeSnapshot())
        var receivedPort: UInt16?
        store.onServerPortChange = { receivedPort = $0 }
        store.serverPortDraft = "70000"

        store.applyServerPortDraft()

        #expect(store.alert == .invalidServerPort)
        #expect(receivedPort == nil)
    }

    @Test @MainActor func chimeSelectionAndPreviewRouteByRole() {
        let store = SettingsStore(snapshot: makeSnapshot())
        var receivedStart: DictationChimeConfig?
        var receivedStop: DictationChimeConfig?
        var previewRole: DictationChimeRole?
        var previewConfig: DictationChimeConfig?

        store.onStartChimeChange = { receivedStart = $0 }
        store.onStopChimeChange = { receivedStop = $0 }
        store.onPreviewChime = { role, config in
            previewRole = role
            previewConfig = config
        }

        store.setChimeSelection(.soft, for: .start)
        store.setChimeSelection(.mechanical, for: .stop)
        store.previewChime(.stop)

        #expect(store.snapshot.startChime.selection == .soft)
        #expect(store.snapshot.stopChime.selection == .mechanical)
        #expect(receivedStart?.selection == .soft)
        #expect(receivedStop?.selection == .mechanical)
        #expect(previewRole == .stop)
        #expect(previewConfig?.selection == .mechanical)
    }

    @Test @MainActor func selectedDownloadModelRoutesToHandler() {
        let store = SettingsStore(snapshot: makeSnapshot())
        var downloadedRepoId: String?
        store.onDownloadModel = { downloadedRepoId = $0 }
        store.selectedDownloadModelRepoId = "mlx-community/Qwen3-ASR-1.7B-bf16"

        store.downloadSelectedModel()

        #expect(downloadedRepoId == "mlx-community/Qwen3-ASR-1.7B-bf16")
    }

    @Test @MainActor func downloadButtonShowsCurrentWhenSelectedModelIsInstalledAndActive() {
        let repoId = "mlx-community/Qwen3-ASR-0.6B-4bit"
        let modelDir = ModelLocator.managedDirectory(forRepoId: repoId)
        try? FileManager.default.removeItem(at: modelDir)
        try? FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        for file in ModelLocator.requiredFiles {
            FileManager.default.createFile(atPath: modelDir.appendingPathComponent(file).path, contents: Data())
        }
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let store = SettingsStore(snapshot: makeSnapshot(transcriptionModel: repoId))
        store.selectedDownloadModelRepoId = repoId

        #expect(store.selectedDownloadModelIsManagedInstalled)
        #expect(store.selectedDownloadModelIsCurrent)
        #expect(store.downloadButtonTitle == "Current")
        #expect(store.downloadRowStatusText == "Installed in Application Support and active now.")
        #expect(!store.canDownloadSelectedModel)
    }

    @Test @MainActor func alignerDownloadStatusStaysInAlignerRow() {
        var snapshot = makeSnapshot()
        snapshot.alignerDownloadStatus = "Word-level alignment model 55% — Downloading model.safetensors…"
        snapshot.isModelDownloadInProgress = true

        let store = SettingsStore(snapshot: snapshot)
        store.selectedDownloadModelRepoId = "mlx-community/placeholder-transcription-model"

        #expect(store.downloadRowStatusText != snapshot.alignerDownloadStatus)
        #expect(store.alignerDownloadRowStatusText == snapshot.alignerDownloadStatus)
        #expect(store.downloadButtonTitle == "Download")
        #expect(store.alignerDownloadButtonTitle == "Downloading…")
        #expect(!store.canDownloadSelectedModel)
        #expect(!store.canDownloadAligner)
    }

    @Test @MainActor func transportDescriptionAndPortDescriptionReflectModeAndTransport() {
        var snapshot = makeSnapshot()
        snapshot.serverMode = .localhost
        snapshot.asrTransport = .stdio
        let localhostStdio = SettingsStore(snapshot: snapshot)
        #expect(localhostStdio.asrTransportDescriptionText.contains("direct pipes"))
        #expect(localhostStdio.serverPortDescriptionText.contains("does not use a local TCP port"))

        snapshot.asrTransport = .http
        let localhostHttp = SettingsStore(snapshot: snapshot)
        #expect(localhostHttp.asrTransportDescriptionText.contains("localhost HTTP"))
        #expect(localhostHttp.serverPortDescriptionText == "Use a custom port if you need Yuwp to avoid another local service.")

        snapshot.serverMode = .allInterfaces
        let lan = SettingsStore(snapshot: snapshot)
        #expect(lan.asrTransportDescriptionText == "Local network mode requires HTTP transport.")

        snapshot.serverMode = .off
        let off = SettingsStore(snapshot: snapshot)
        #expect(off.asrTransportDescriptionText == "Choose how Yuwp talks to its ASR process when the server is enabled.")
    }

    @Test @MainActor func setASRTransportUpdatesSnapshotAndCallback() {
        let store = SettingsStore(snapshot: makeSnapshot())
        var callback: ASRIPCTransport?
        store.onASRTransportChange = { callback = $0 }

        store.setASRTransport(.stdio)

        #expect(store.snapshot.asrTransport == .stdio)
        #expect(callback == .stdio)
    }

    private func makeSnapshot(transcriptionModel: String = "mlx-community/Qwen3-ASR-0.6B-4bit") -> SettingsSnapshot {
        SettingsSnapshot(
            dictationBinding: .ctrlBacktick,
            audioInputSelection: .systemDefault,
            availableAudioInputs: [],
            dictationLanguageMode: .mixed,
            fixedDictationLanguage: "English",
            supportedDictationLanguages: DictationLanguageCatalog.fallbackSupportedLanguages,
            serverMode: .localhost,
            serverPort: 7936,
            asrTransport: .http,
            transcriptionModel: transcriptionModel,
            batchCommitEnabled: true,
            transcriptionDownloadStatus: nil,
            alignerDownloadStatus: nil,
            isModelDownloadInProgress: false,
            alignerModelRepoId: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
            alignerInstalled: false,
            saveRecordings: false,
            diagnosticLoggingEnabled: false,
            recordingsDir: FileManager.default.temporaryDirectory.appendingPathComponent("yuwp-tests-recordings", isDirectory: true),
            usingDefaultRecordingsDir: true,
            micPanelAnimation: .default,
            startChime: .default,
            stopChime: .default
        )
    }
}
