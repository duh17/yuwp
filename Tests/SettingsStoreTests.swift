import Foundation
import Testing
@testable import Yuwp

@Suite("SettingsStore")
struct SettingsStoreTests {
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

    private func makeSnapshot(transcriptionModel: String = "mlx-community/Qwen3-ASR-0.6B-4bit") -> SettingsSnapshot {
        SettingsSnapshot(
            dictationBinding: .ctrlBacktick,
            audioInputSelection: .systemDefault,
            availableAudioInputs: [],
            serverMode: .localhost,
            serverPort: 9748,
            transcriptionModel: transcriptionModel,
            batchCommitEnabled: true,
            modelDownloadStatus: nil,
            saveRecordings: false,
            recordingsDir: FileManager.default.temporaryDirectory.appendingPathComponent("yuwp-tests-recordings", isDirectory: true),
            usingDefaultRecordingsDir: true,
            micPanelAnimation: .default,
            startChime: .default,
            stopChime: .default
        )
    }
}
