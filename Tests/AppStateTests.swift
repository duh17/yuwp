import ASRIPC
import Foundation
import Testing
@testable import Yuwp

@Suite("AppState")
struct AppStateTests {
    @Test func shortcutStartsListeningWhenProviderIsReady() {
        var state = AppState(
            settings: AppSettingsState(serverMode: .localhost),
            hasAccessibilityPermission: true,
            microphonePermission: .granted,
            providerState: .ready
        )

        let effects = state.send(.shortcutReceived(ShortcutEvent(command: .dictation, phase: .pressed)))

        #expect(state.sessionPhase == .listening)
        #expect(effects == [.startDictation])
    }

    @Test func shortcutWhileStoppingDoesNotStartAnotherSession() {
        var state = AppState(
            settings: AppSettingsState(serverMode: .localhost),
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
            settings: AppSettingsState(serverMode: .localhost),
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
            settings: AppSettingsState(serverMode: .localhost),
            hasAccessibilityPermission: true,
            microphonePermission: .granted,
            providerState: .starting,
            missingConfiguredModelLabels: ["Model"]
        )

        let effects = state.send(.shortcutReceived(ShortcutEvent(command: .dictation, phase: .pressed)))

        #expect(state.sessionPhase == .idle)
        #expect(effects == [
            .log("Model missing — open Settings → Model to download it")
        ])
    }

    @Test func shortcutRequestsMicrophonePermissionWhenUndetermined() {
        var state = AppState(
            settings: AppSettingsState(serverMode: .localhost),
            hasAccessibilityPermission: true,
            microphonePermission: .notDetermined,
            providerState: .ready
        )

        let effects = state.send(.shortcutReceived(ShortcutEvent(command: .dictation, phase: .pressed)))

        #expect(state.sessionPhase == .idle)
        #expect(effects == [
            .log("Microphone permission required — requesting access"),
            .requestMicrophonePermission,
        ])
    }

    @Test func shortcutLogsWhenMicrophonePermissionDenied() {
        var state = AppState(
            settings: AppSettingsState(serverMode: .localhost),
            hasAccessibilityPermission: true,
            microphonePermission: .denied,
            providerState: .ready
        )

        let effects = state.send(.shortcutReceived(ShortcutEvent(command: .dictation, phase: .pressed)))

        #expect(state.sessionPhase == .idle)
        #expect(effects == [
            .log("Microphone permission denied — open System Settings → Privacy & Security → Microphone")
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
            settings: AppSettingsState(serverMode: .allInterfaces),
            hasAccessibilityPermission: true,
            providerState: .ready,
            sessionPhase: .idle,
            pendingEnterReplay: false,
            activeModelDownload: nil,
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

    @Test func statusDisplayShowsStdioWhenConfiguredForLocalhost() {
        let state = AppState(
            settings: AppSettingsState(serverMode: .localhost, asrTransport: .stdio),
            hasAccessibilityPermission: true,
            providerState: .ready
        )

        let status = state.statusDisplay(port: 9748)

        #expect(status == AppStatusDisplay(
            title: "Ready (stdio)",
            symbolName: "checkmark.circle.fill",
            isEnabled: false,
            behavior: .none
        ))
    }

    @Test func statusDisplayShowsMicrophoneActionWhenDenied() {
        let state = AppState(
            settings: AppSettingsState(serverMode: .localhost),
            hasAccessibilityPermission: true,
            microphonePermission: .denied,
            providerState: .ready
        )

        let status = state.statusDisplay(port: 9748)

        #expect(status == AppStatusDisplay(
            title: "Grant Microphone Permission",
            symbolName: "mic.slash.fill",
            isEnabled: true,
            behavior: .openMicrophoneSettings
        ))
    }

    @Test func statusDisplayShowsSettingsActionWhenModelMissing() {
        let state = AppState(
            settings: AppSettingsState(serverMode: .localhost),
            hasAccessibilityPermission: true,
            providerState: .ready,
            missingConfiguredModelLabels: ["Model"]
        )

        let status = state.statusDisplay(port: 9748)

        #expect(status == AppStatusDisplay(
            title: "Model missing",
            symbolName: "exclamationmark.triangle.fill",
            isEnabled: true,
            behavior: .openSettings
        ))
    }

    @Test func statusDisplayShowsReadyWithWarningWhenOnlyAlignmentModelMissing() {
        let state = AppState(
            settings: AppSettingsState(serverMode: .localhost),
            hasAccessibilityPermission: true,
            providerState: .ready,
            missingConfiguredModelLabels: ["Word-level Alignment"]
        )

        let status = state.statusDisplay(port: 9748)

        #expect(status == AppStatusDisplay(
            title: "Ready (word-level alignment missing)",
            symbolName: "exclamationmark.triangle.fill",
            isEnabled: true,
            behavior: .openSettings
        ))
    }

    @Test func modelDownloadStatusTracksRepoLifecycle() {
        var state = AppState()

        _ = state.send(.modelDownloadStatusChanged(repoId: "repo-a", status: "Downloading…"))
        #expect(state.activeModelDownload == ModelDownloadActivity(repoId: "repo-a", status: "Downloading…"))

        _ = state.send(.modelDownloadStatusChanged(repoId: "repo-b", status: nil))
        #expect(state.activeModelDownload == ModelDownloadActivity(repoId: "repo-a", status: "Downloading…"))

        _ = state.send(.modelDownloadStatusChanged(repoId: "repo-a", status: nil))
        #expect(state.activeModelDownload == nil)
    }

    @Test func statusDisplayShowsHttpEndpointOnLocalhostWhenUsingHTTPTransport() {
        let state = AppState(
            settings: AppSettingsState(serverMode: .localhost, asrTransport: .http),
            hasAccessibilityPermission: true,
            providerState: .ready
        )

        let status = state.statusDisplay(port: 9748)

        #expect(status == AppStatusDisplay(
            title: "Ready (127.0.0.1:9748)",
            symbolName: "checkmark.circle.fill",
            isEnabled: false,
            behavior: .none
        ))
    }

    @Test func serverRuntimePolicyBehaviorIsConsistent() {
        #expect(ServerRuntimePolicy.effectiveTransport(serverMode: .localhost, requestedTransport: .stdio) == .stdio)
        #expect(ServerRuntimePolicy.effectiveTransport(serverMode: .localhost, requestedTransport: .http) == .http)
        #expect(ServerRuntimePolicy.effectiveTransport(serverMode: .allInterfaces, requestedTransport: .stdio) == .http)
        #expect(ServerRuntimePolicy.effectiveTransport(serverMode: .allInterfaces, requestedTransport: .http) == .http)

        #expect(ServerRuntimePolicy.canUseTransport(.stdio, in: .localhost))
        #expect(ServerRuntimePolicy.canUseTransport(.http, in: .localhost))
        #expect(ServerRuntimePolicy.canUseTransport(.http, in: .allInterfaces))
        #expect(!ServerRuntimePolicy.canUseTransport(.stdio, in: .allInterfaces))

        #expect(!ServerRuntimePolicy.shouldRestartProviderForPortChange(serverMode: .localhost, transport: .stdio))
        #expect(ServerRuntimePolicy.shouldRestartProviderForPortChange(serverMode: .localhost, transport: .http))
        #expect(ServerRuntimePolicy.shouldRestartProviderForPortChange(serverMode: .allInterfaces, transport: .http))
        #expect(ServerRuntimePolicy.shouldRestartProviderForPortChange(serverMode: .allInterfaces, transport: .stdio))
    }
}
