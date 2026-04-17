import ASRIPC
import Foundation

enum AppSessionPhase: String, Sendable, Equatable {
    case idle
    case listening
    case stopping

    var isCapturingAudio: Bool {
        self == .listening
    }
}

struct AppSettingsState: Sendable, Equatable {
    var serverMode: ServerMode = .localhost
    var asrTransport: ASRIPCTransport = .stdio
}

enum MicrophonePermissionState: Sendable, Equatable {
    case notDetermined
    case granted
    case denied
}

enum AppAction: Sendable, Equatable {
    case shortcutReceived(ShortcutEvent)
    case micPanelDismissed
    case enterIntercepted
    case sessionStopRequested
    case sessionEvent(DictationEvent)
    case providerStateChanged(ASRServerState)
    case accessibilityPermissionChanged(Bool)
    case microphonePermissionChanged(MicrophonePermissionState)
    case settingsChanged(AppSettingsState)
    case missingConfiguredModelsChanged([String])
    case modelDownloadStatusChanged(repoId: String, status: String?)
}

enum AppEffect: Sendable, Equatable {
    case startDictation
    case stopDictation
    case replayEnter
    case requestMicrophonePermission
    case presentMicPanel(DictationPresentationState)
    case updateMicLevel(Float)
    case hideMicPanel
    case log(String)
}

enum AppStatusBehavior: Sendable, Equatable {
    case none
    case openAccessibilitySettings
    case openMicrophoneSettings
    case openSettings
}

struct AppStatusDisplay: Sendable, Equatable {
    let title: String
    let symbolName: String
    let isEnabled: Bool
    let behavior: AppStatusBehavior
}

struct ModelDownloadActivity: Sendable, Equatable {
    let repoId: String
    let status: String
}

struct AppState: Sendable, Equatable {
    var settings = AppSettingsState()
    var hasAccessibilityPermission = false
    var microphonePermission: MicrophonePermissionState = .notDetermined
    var providerState: ASRServerState = .stopped
    var sessionPhase: AppSessionPhase = .idle
    var pendingEnterReplay = false
    var activeModelDownload: ModelDownloadActivity?
    var missingConfiguredModelLabels: [String] = []
    var hotkeyBehavior = DictationHotkeyBehavior()

    var isProviderReady: Bool {
        providerState == .ready
    }

    var statusItemSymbolName: String {
        sessionPhase == .listening ? "waveform.circle.fill" : "waveform"
    }

    mutating func send(
        _ action: AppAction,
        dictationBindingDescription: String? = nil
    ) -> [AppEffect] {
        switch action {
        case .shortcutReceived(let event):
            return handleShortcutEvent(event)

        case .micPanelDismissed:
            guard sessionPhase == .listening else { return [] }
            return [
                .log("Escape pressed — stopping dictation"),
                .stopDictation,
                .hideMicPanel,
            ].withTransitionToStopping(using: &hotkeyBehavior, sessionPhase: &sessionPhase)

        case .enterIntercepted:
            guard sessionPhase == .listening else { return [] }
            pendingEnterReplay = true
            return [
                .log("Enter intercepted — stopping dictation, will replay Enter after commit"),
                .stopDictation,
                .hideMicPanel,
            ].withTransitionToStopping(using: &hotkeyBehavior, sessionPhase: &sessionPhase)

        case .sessionStopRequested:
            guard sessionPhase == .listening else { return [] }
            return [
                .stopDictation,
                .hideMicPanel,
            ].withTransitionToStopping(using: &hotkeyBehavior, sessionPhase: &sessionPhase)

        case .sessionEvent(let event):
            return handleSessionEvent(event)

        case .providerStateChanged(let state):
            providerState = state
            return state == .ready ? [.log("STT provider ready")] : []

        case .accessibilityPermissionChanged(let granted):
            hasAccessibilityPermission = granted
            guard granted, let dictationBindingDescription else { return [] }
            return [.log("Ready. \(dictationBindingDescription) to dictate.")]

        case .microphonePermissionChanged(let state):
            microphonePermission = state
            return []

        case .settingsChanged(let settings):
            self.settings = settings
            return []

        case .missingConfiguredModelsChanged(let labels):
            missingConfiguredModelLabels = labels
            return []

        case .modelDownloadStatusChanged(let repoId, let status):
            if let status {
                activeModelDownload = ModelDownloadActivity(repoId: repoId, status: status)
            } else if activeModelDownload?.repoId == repoId {
                activeModelDownload = nil
            }
            return []
        }
    }

    func statusDisplay(port: UInt16) -> AppStatusDisplay {
        if !hasAccessibilityPermission {
            return AppStatusDisplay(
                title: "Grant Accessibility Permission",
                symbolName: "hand.raised.fill",
                isEnabled: true,
                behavior: .openAccessibilitySettings
            )
        }

        if microphonePermission == .denied {
            return AppStatusDisplay(
                title: "Grant Microphone Permission",
                symbolName: "mic.slash.fill",
                isEnabled: true,
                behavior: .openMicrophoneSettings
            )
        }

        if settings.serverMode == .off {
            return AppStatusDisplay(
                title: "Server mode is off",
                symbolName: "power",
                isEnabled: false,
                behavior: .none
            )
        }

        if let activeModelDownload {
            return AppStatusDisplay(
                title: activeModelDownload.status,
                symbolName: "arrow.down.circle",
                isEnabled: false,
                behavior: .none
            )
        }

        if !missingConfiguredModelLabels.isEmpty {
            if providerState == .ready,
               missingConfiguredModelLabels.count == 1,
               missingConfiguredModelLabels.first == "Word-level Alignment" {
                return AppStatusDisplay(
                    title: "Ready (word-level alignment missing)",
                    symbolName: "exclamationmark.triangle.fill",
                    isEnabled: true,
                    behavior: .openSettings
                )
            }

            let title: String
            if missingConfiguredModelLabels.count == 1, let label = missingConfiguredModelLabels.first {
                title = "\(label) missing"
            } else {
                title = "\(missingConfiguredModelLabels.joined(separator: " + ")) missing"
            }
            return AppStatusDisplay(
                title: title,
                symbolName: "exclamationmark.triangle.fill",
                isEnabled: true,
                behavior: .openSettings
            )
        }

        let title: String
        let symbolName: String
        switch providerState {
        case .disabled:
            title = "Server mode is off"
            symbolName = "power"
        case .stopped:
            title = "Stopped"
            symbolName = "stop.circle"
        case .starting:
            title = "Loading model..."
            symbolName = "arrow.trianglehead.clockwise"
        case .ready:
            if settings.serverMode == .allInterfaces || settings.asrTransport == .http {
                let endpoint = settings.serverMode == .allInterfaces
                    ? "0.0.0.0:\(port)"
                    : "127.0.0.1:\(port)"
                title = "Ready (\(endpoint))"
            } else {
                title = "Ready (stdio)"
            }
            symbolName = "checkmark.circle.fill"
        case .error(let message):
            title = message
            symbolName = "exclamationmark.triangle.fill"
        }

        return AppStatusDisplay(title: title, symbolName: symbolName, isEnabled: false, behavior: .none)
    }

    private mutating func handleShortcutEvent(_ event: ShortcutEvent) -> [AppEffect] {
        guard event.command == .dictation else { return [] }

        let action = hotkeyBehavior.handle(
            phase: event.phase,
            isSessionActive: sessionPhase.isCapturingAudio
        )

        switch action {
        case .start:
            return handleStartRequest()
        case .stop:
            guard sessionPhase == .listening else { return [] }
            return [
                .stopDictation,
                .hideMicPanel,
            ].withTransitionToStopping(using: &hotkeyBehavior, sessionPhase: &sessionPhase)
        case .none:
            return []
        }
    }

    private mutating func handleStartRequest() -> [AppEffect] {
        guard sessionPhase == .idle else { return [] }

        guard settings.serverMode != .off else {
            return [.log("Server mode is off — enable This Mac only or Local network to dictate")]
        }

        if microphonePermission == .notDetermined {
            return [
                .log("Microphone permission required — requesting access"),
                .requestMicrophonePermission,
            ]
        }

        guard microphonePermission != .denied else {
            return [.log("Microphone permission denied — open System Settings → Privacy & Security → Microphone")]
        }

        guard providerState == .ready else {
            if missingConfiguredModelLabels.isEmpty {
                return [.log("Model still loading, please wait...")]
            }
            if missingConfiguredModelLabels.count == 1, let label = missingConfiguredModelLabels.first {
                return [.log("\(label) missing — open Settings → Model to download it")]
            }
            return [.log("\(missingConfiguredModelLabels.joined(separator: " + ")) missing — open Settings → Model to download them")]
        }

        sessionPhase = .listening
        return [.startDictation]
    }

    private mutating func handleSessionEvent(_ event: DictationEvent) -> [AppEffect] {
        switch event {
        case .presentation(let state):
            return [.presentMicPanel(state)]

        case .audioLevel(let level):
            return [.updateMicLevel(level)]

        case .finished:
            sessionPhase = .idle
            hotkeyBehavior.sessionDidEnd()
            let shouldReplayEnter = pendingEnterReplay
            pendingEnterReplay = false

            var effects: [AppEffect] = [.hideMicPanel]
            if shouldReplayEnter {
                effects.append(.replayEnter)
            }
            return effects
        }
    }
}

private extension Array where Element == AppEffect {
    func withTransitionToStopping(
        using hotkeyBehavior: inout DictationHotkeyBehavior,
        sessionPhase: inout AppSessionPhase
    ) -> [AppEffect] {
        hotkeyBehavior.sessionDidEnd()
        sessionPhase = .stopping
        return self
    }
}
