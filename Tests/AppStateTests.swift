import Foundation
import Testing
@testable import Yuwp

@Suite("AppState")
struct AppStateTests {
    @Test func shortcutStartsListeningWhenProviderIsReady() {
        var state = AppState(
            settings: AppSettingsState(dictationMode: .toggle, serverMode: .localhost),
            hasAccessibilityPermission: true,
            providerState: .ready
        )

        let effects = state.send(.shortcutReceived(ShortcutEvent(command: .dictation, phase: .pressed)))

        #expect(state.sessionPhase == .listening)
        #expect(effects == [.startDictation])
    }

    @Test func shortcutWhileStoppingDoesNotStartAnotherSession() {
        var state = AppState(
            settings: AppSettingsState(dictationMode: .toggle, serverMode: .localhost),
            hasAccessibilityPermission: true,
            providerState: .ready,
            sessionPhase: .stopping
        )

        let effects = state.send(.shortcutReceived(ShortcutEvent(command: .dictation, phase: .pressed)))

        #expect(state.sessionPhase == .stopping)
        #expect(effects.isEmpty)
    }

    @Test func enterInterceptedStopsThenFinishedReplaysEnter() {
        var state = AppState(
            settings: AppSettingsState(dictationMode: .toggle, serverMode: .localhost),
            hasAccessibilityPermission: true,
            providerState: .ready,
            sessionPhase: .listening
        )

        let stopEffects = state.send(.enterIntercepted)

        #expect(state.sessionPhase == .stopping)
        #expect(state.pendingEnterReplay)
        #expect(stopEffects == [
            .log("Enter intercepted — stopping dictation, will replay Enter after commit"),
            .stopDictation,
            .hideMicPanel,
        ])

        let finishEffects = state.send(.sessionEvent(.finished))

        #expect(state.sessionPhase == .idle)
        #expect(!state.pendingEnterReplay)
        #expect(finishEffects == [.hideMicPanel, .replayEnter])
    }

    @Test func shortcutLogsMissingModelWhileProviderIsUnavailable() {
        var state = AppState(
            settings: AppSettingsState(dictationMode: .toggle, serverMode: .localhost),
            hasAccessibilityPermission: true,
            providerState: .starting,
            missingConfiguredModelLabels: ["Transcription"]
        )

        let effects = state.send(.shortcutReceived(ShortcutEvent(command: .dictation, phase: .pressed)))

        #expect(state.sessionPhase == .idle)
        #expect(effects == [
            .log("Transcription model missing — open Settings → Transcription to fix it")
        ])
    }

    @Test func providerReadyLogsReadiness() {
        var state = AppState()

        let effects = state.send(.providerStateChanged(.ready))

        #expect(state.providerState == .ready)
        #expect(effects == [.log("STT provider ready")])
    }

    @Test func statusDisplayUsesReducerState() {
        let state = AppState(
            settings: AppSettingsState(dictationMode: .toggle, serverMode: .allInterfaces),
            hasAccessibilityPermission: true,
            providerState: .ready,
            sessionPhase: .idle,
            pendingEnterReplay: false,
            modelDownloadStatus: nil,
            missingConfiguredModelLabels: []
        )

        let status = state.statusDisplay(port: 9748)

        #expect(status == AppStatusDisplay(
            title: "Ready (0.0.0.0:9748)",
            symbolName: "checkmark.circle.fill",
            isEnabled: false,
            behavior: .none
        ))
    }
}
