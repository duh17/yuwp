import ASRIPC
import SwiftUI

private enum SettingsSidebarSection: String, CaseIterable, Identifiable {
    case dictation
    case transcription
    case recordings
    case network
    case feedback

    var id: String { rawValue }

    var title: String {
        switch self {
        case .dictation: "Dictation"
        case .transcription: "Model"
        case .recordings: "Recordings"
        case .network: "Network"
        case .feedback: "Feedback"
        }
    }

    var systemImage: String {
        switch self {
        case .dictation: "mic"
        case .transcription: "waveform.and.magnifyingglass"
        case .recordings: "record.circle"
        case .network: "network"
        case .feedback: "slider.horizontal.3"
        }
    }
}

struct SettingsView: View {
    private static func initialSelection() -> SettingsSidebarSection {
        let raw = ProcessInfo.processInfo.environment["YUWP_SETTINGS_INITIAL_SECTION"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return raw.flatMap(SettingsSidebarSection.init(rawValue:)) ?? .dictation
    }

    @ObservedObject var store: SettingsStore
    @State private var selection: SettingsSidebarSection = Self.initialSelection()

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            detail
        }
        .frame(minWidth: 920, minHeight: 620)
        .background(Color(nsColor: .windowBackgroundColor))
        .alert(item: $store.alert) { alert in
            switch alert {
            case .invalidServerPort:
                return Alert(
                    title: Text("Invalid Port"),
                    message: Text("Enter a port between 1 and 65535."),
                    dismissButton: .default(Text("OK"))
                )
            }
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Settings")
                .font(.title3.weight(.semibold))
                .padding(.horizontal, 16)
                .padding(.top, 16)

            VStack(spacing: 6) {
                ForEach(SettingsSidebarSection.allCases) { item in
                    Button {
                        selection = item
                    } label: {
                        Label(item.title, systemImage: item.systemImage)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                    }
                    .buttonStyle(.plain)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(selection == item ? Color.accentColor : Color.clear)
                    )
                    .foregroundStyle(selection == item ? Color.white : Color.primary)
                }
            }
            .padding(.horizontal, 12)

            Spacer()
        }
        .frame(width: 220, alignment: .topLeading)
        .background(Color(nsColor: .underPageBackgroundColor))
    }

    private var detail: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                pageHeader(title: selection.title, subtitle: pageSubtitle(for: selection))

                switch selection {
                case .dictation:
                    dictationContent
                case .transcription:
                    transcriptionContent
                case .recordings:
                    recordingsContent
                case .network:
                    networkContent
                case .feedback:
                    feedbackContent
                }
            }
            .padding(24)
            .frame(maxWidth: 820, alignment: .leading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func pageHeader(title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 28, weight: .semibold))
            Text(subtitle)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func pageSubtitle(for section: SettingsSidebarSection) -> String {
        switch section {
        case .dictation:
            "Choose how Yuwp starts dictation and which shortcut triggers it."
        case .transcription:
            "Choose the transcription model, manage downloads, and install the optional word-level alignment model for timestamped output."
        case .recordings:
            "Keep source audio if you want a paper trail for debugging, QA, or re-transcription later."
        case .network:
            "Choose whether Yuwp exposes its local transcription server only to this Mac or to other devices on your local network. Local network mode is unauthenticated and unencrypted."
        case .feedback:
            "Tune how the mic panel animates while dictation is active, and choose the sounds Yuwp plays when dictation starts and stops."
        }
    }

    private var dictationContent: some View {
        SettingsGroup {
            SettingsControlRow(
                title: "Shortcut",
                subtitle: "Global hotkey used to start and stop dictation. Record a modifier twice to use it as a double-tap shortcut.",
                topAligned: true
            ) {
                ShortcutRecorderRepresentable(
                    binding: Binding(
                        get: { store.snapshot.dictationBinding },
                        set: { store.setDictationBinding($0) }
                    ),
                    defaultBinding: .ctrlBacktick,
                    onRecordingChange: { store.setDictationBindingRecording($0) }
                )
                .frame(minWidth: 360, idealWidth: 520, maxWidth: 620, minHeight: 30, alignment: .trailing)
                .accessibilityLabel("Shortcut")
            }

            SettingsDivider()

            SettingsControlRow(
                title: "Microphone",
                subtitle: store.audioInputDescriptionText,
                topAligned: true
            ) {
                Picker("Microphone", selection: Binding(
                    get: { store.snapshot.audioInputSelection.persistenceString },
                    set: { store.setAudioInputSelection(AudioInputSelection(persistenceString: $0)) }
                )) {
                    ForEach(store.audioInputSelections, id: \.persistenceString) { selection in
                        Text(audioInputTitle(for: selection)).tag(selection.persistenceString)
                    }
                }
                .labelsHidden()
                .frame(width: 360, alignment: .trailing)
            }

            SettingsDivider()

            SettingsControlRow(
                title: "Direct text-field insertion (Experimental)",
                subtitle: "When off (recommended), Yuwp shows the growing preview bubble and pastes on commit instead of typing directly into AX-editable fields."
            ) {
                Toggle("Direct text-field insertion", isOn: Binding(
                    get: { store.snapshot.experimentalDirectTextFieldInsertionEnabled },
                    set: { store.setExperimentalDirectTextFieldInsertionEnabled($0) }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }

            SettingsDivider()

            SettingsControlRow(
                title: "Direct terminal insertion (Experimental)",
                subtitle: "When off (recommended), Yuwp avoids CGEvent keypress injection in terminals and uses preview bubble + paste on commit."
            ) {
                Toggle("Direct terminal insertion", isOn: Binding(
                    get: { store.snapshot.experimentalDirectTerminalInsertionEnabled },
                    set: { store.setExperimentalDirectTerminalInsertionEnabled($0) }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }
        }
    }

    private var transcriptionContent: some View {
        SettingsGroup {
            SettingsBlockRow(
                title: "Model",
                subtitle: store.modelStatusText
            ) {
                HStack(spacing: 8) {
                    TextField(
                        "mlx-community/Qwen3-ASR-0.6B-4bit or /path/to/model",
                        text: $store.transcriptionModelDraft
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 320)
                    .accessibilityLabel("Transcription model")

                    Button("Choose…") {
                        store.chooseModelDirectory()
                    }
                    .fixedSize(horizontal: true, vertical: false)

                    Button("Use") {
                        store.applyTranscriptionModelDraft()
                    }
                    .fixedSize(horizontal: true, vertical: false)
                }
            }

            SettingsDivider()

            SettingsControlRow(
                title: "Final Accuracy Pass",
                subtitle: "Retranscribe settled segments to improve final accuracy. Adds a little commit latency."
            ) {
                Toggle("Final Accuracy Pass", isOn: Binding(
                    get: { store.snapshot.batchCommitEnabled },
                    set: { store.setBatchCommitEnabled($0) }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }

            SettingsDivider()

            SettingsControlRow(
                title: "Language Mode",
                subtitle: store.dictationLanguageModeSummaryText,
                topAligned: true
            ) {
                Picker("Language mode", selection: Binding(
                    get: { store.snapshot.dictationLanguageMode.rawValue },
                    set: { rawValue in
                        if let mode = DictationLanguageMode(rawValue: rawValue) {
                            store.setDictationLanguageMode(mode)
                        }
                    }
                )) {
                    ForEach(store.dictationLanguageModes, id: \.rawValue) { mode in
                        Text(mode.title).tag(mode.rawValue)
                    }
                }
                .labelsHidden()
                .frame(width: 260, alignment: .trailing)
            }

            if store.snapshot.dictationLanguageMode == .fixed {
                SettingsDivider()

                SettingsControlRow(
                    title: "Fixed Language",
                    subtitle: store.fixedDictationLanguageDescriptionText,
                    topAligned: true
                ) {
                    Picker("Fixed language", selection: Binding(
                        get: { store.snapshot.fixedDictationLanguage },
                        set: { store.setFixedDictationLanguage($0) }
                    )) {
                        ForEach(store.supportedDictationLanguages, id: \.self) { language in
                            Text(language).tag(language)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 220, alignment: .trailing)
                }
            }

            SettingsDivider()

            SettingsBlockRow(
                title: "Download",
                subtitle: store.downloadRowStatusText
            ) {
                HStack(spacing: 8) {
                    Picker("Download model", selection: $store.selectedDownloadModelRepoId) {
                        ForEach(store.downloadableModels, id: \.repoId) { model in
                            Text(model.label).tag(model.repoId)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 260, alignment: .trailing)
                    .disabled(store.isModelDownloadInProgress)

                    Button(store.downloadButtonTitle) {
                        store.downloadSelectedModel()
                    }
                    .fixedSize(horizontal: true, vertical: false)
                    .layoutPriority(1)
                    .disabled(!store.canDownloadSelectedModel)
                }
            }

            SettingsDivider()

            SettingsBlockRow(
                title: "Word-level Alignment",
                subtitle: store.alignerDownloadRowStatusText
            ) {
                HStack(spacing: 8) {
                    Text(store.alignerModelDisplayName)
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(width: 260, alignment: .trailing)

                    Button(store.alignerDownloadButtonTitle) {
                        store.downloadModel(repoId: store.snapshot.alignerModelRepoId)
                    }
                    .fixedSize(horizontal: true, vertical: false)
                    .layoutPriority(1)
                    .disabled(!store.canDownloadAligner)
                }
            }
        }
    }

    private var recordingsContent: some View {
        SettingsGroup {
            SettingsControlRow(
                title: "Save Recordings",
                subtitle: "Save microphone audio to disk after each dictation session."
            ) {
                Toggle("Save Recordings", isOn: Binding(
                    get: { store.snapshot.saveRecordings },
                    set: { store.setSaveRecordings($0) }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }

            SettingsDivider()

            SettingsControlRow(
                title: "Diagnostic Logging",
                subtitle: "Write troubleshooting logs to stderr. Off by default."
            ) {
                Toggle("Diagnostic Logging", isOn: Binding(
                    get: { store.snapshot.diagnosticLoggingEnabled },
                    set: { store.setDiagnosticLoggingEnabled($0) }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }

            SettingsDivider()

            SettingsBlockRow(
                title: "Recording Location",
                subtitle: store.recordingsPathText,
                muted: !store.snapshot.saveRecordings
            ) {
                HStack(spacing: 8) {
                    Button("Choose…") { store.chooseRecordingsDirectory() }
                        .fixedSize(horizontal: true, vertical: false)
                    Button("Reset Default") { store.resetRecordingsDirectory() }
                        .fixedSize(horizontal: true, vertical: false)
                    Button("Reveal in Finder") { store.revealRecordingsDirectory() }
                        .fixedSize(horizontal: true, vertical: false)
                }
                .disabled(!store.snapshot.saveRecordings)
            }
            .opacity(store.snapshot.saveRecordings ? 1.0 : 0.55)
        }
    }

    private var networkContent: some View {
        VStack(alignment: .leading, spacing: 18) {
            SettingsGroup {
                SettingsControlRow(
                    title: "Availability",
                    subtitle: store.serverModeDescriptionText,
                    topAligned: true
                ) {
                    Picker("Server availability", selection: Binding(
                        get: { store.snapshot.serverMode.rawValue },
                        set: { rawValue in
                            if let mode = ServerMode(rawValue: rawValue) {
                                store.setServerMode(mode)
                            }
                        }
                    )) {
                        ForEach(ServerMode.allCases, id: \.rawValue) { mode in
                            Text(mode.description).tag(mode.rawValue)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 220, alignment: .trailing)
                }

                SettingsDivider()

                SettingsControlRow(
                    title: "App Transport",
                    subtitle: store.asrTransportDescriptionText,
                    topAligned: true
                ) {
                    Picker("App transport", selection: Binding(
                        get: { store.snapshot.asrTransport.rawValue },
                        set: { rawValue in
                            if let transport = ASRIPCTransport(rawValue: rawValue) {
                                store.setASRTransport(transport)
                            }
                        }
                    )) {
                        ForEach(ASRIPCTransport.allCases, id: \.rawValue) { transport in
                            Text(transport.settingsTitle).tag(transport.rawValue)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 220, alignment: .trailing)
                    .disabled(store.snapshot.serverMode == .allInterfaces)
                    .opacity(store.snapshot.serverMode == .allInterfaces ? 0.55 : 1.0)
                }

                SettingsDivider()

                SettingsBlockRow(
                    title: "Port",
                    subtitle: store.serverPortDescriptionText,
                    muted: store.snapshot.serverMode == .off
                ) {
                    HStack(spacing: 8) {
                        TextField("\(ASRIPCDefaults.defaultHTTPPort)", text: $store.serverPortDraft)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 110)
                            .accessibilityLabel("Server port")

                        Button("Apply") {
                            store.applyServerPortDraft()
                        }
                        .fixedSize(horizontal: true, vertical: false)

                        Text("1–65535")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .disabled(store.snapshot.serverMode == .off)
                    .opacity(store.snapshot.serverMode == .off ? 0.55 : 1.0)
                }
            }

            if store.snapshot.serverMode == .allInterfaces {
                InsetSettingsCard(
                    title: "Security warning",
                    subtitle: "Local network mode exposes Yuwp’s HTTP API to other devices on your network. The API is unauthenticated and unencrypted. Only enable this on trusted networks."
                ) {
                    EmptyView()
                }
            }
        }
    }

    private var feedbackContent: some View {
        VStack(alignment: .leading, spacing: 18) {
            SettingsGroup {
                SettingsControlRow(
                    title: "Mic Panel",
                    subtitle: store.micPanelAnimationSummaryText,
                    topAligned: true
                ) {
                    Picker("Mic panel animation", selection: Binding(
                        get: { store.micPanelAnimationSelection.rawValue },
                        set: { rawValue in
                            if let selection = MicPanelAnimationSelection(rawValue: rawValue) {
                                store.setMicPanelAnimationSelection(selection)
                            }
                        }
                    )) {
                        ForEach(MicPanelAnimationSelection.allCases, id: \.rawValue) { selection in
                            Text(selection.title).tag(selection.rawValue)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 220, alignment: .trailing)
                }
            }

            if store.micPanelAnimationSelection == .custom {
                InsetSettingsCard(
                    title: "Custom animation tuning",
                    subtitle: "These controls adjust how quickly the panel responds, how much the bars drift at idle, and how strongly the border glow reacts to your voice."
                ) {
                    VStack(spacing: 12) {
                        SettingsSliderRow(
                            label: "Attack",
                            value: Binding(
                                get: { store.micPanelAnimationCustom.smoothingAttack },
                                set: { value in store.updateMicPanelAnimationCustom { $0.smoothingAttack = value } }
                            ),
                            range: 0.05...1.0,
                            format: "%.2f"
                        )
                        SettingsSliderRow(
                            label: "Decay",
                            value: Binding(
                                get: { store.micPanelAnimationCustom.smoothingDecay },
                                set: { value in store.updateMicPanelAnimationCustom { $0.smoothingDecay = value } }
                            ),
                            range: 0.02...1.0,
                            format: "%.2f"
                        )
                        SettingsSliderRow(
                            label: "Motion",
                            value: Binding(
                                get: { store.micPanelAnimationCustom.idleBarAmplitude },
                                set: { value in store.updateMicPanelAnimationCustom { $0.idleBarAmplitude = value } }
                            ),
                            range: 0.0...6.0,
                            format: "%.1f"
                        )
                        SettingsSliderRow(
                            label: "Energy",
                            value: Binding(
                                get: { store.micPanelAnimationCustom.levelBarScale },
                                set: { value in store.updateMicPanelAnimationCustom { $0.levelBarScale = value } }
                            ),
                            range: 0.2...1.8,
                            format: "%.2f"
                        )
                        SettingsSliderRow(
                            label: "Glow",
                            value: Binding(
                                get: { store.micPanelAnimationCustom.glowAlphaScale },
                                set: { value in store.updateMicPanelAnimationCustom { $0.glowAlphaScale = value } }
                            ),
                            range: 0.0...0.8,
                            format: "%.2f"
                        )
                    }
                }
            }

            SettingsGroup {
                chimeRow(for: .start)
                SettingsDivider()
                chimeRow(for: .stop)
            }
        }
    }

    @ViewBuilder
    private func chimeRow(for role: DictationChimeRole) -> some View {
        SettingsBlockRow(
            title: role.title,
            subtitle: store.chimeSummaryText(for: role)
        ) {
            VStack(alignment: .trailing, spacing: 8) {
                HStack(spacing: 8) {
                    Picker("\(role.title) sound", selection: Binding(
                        get: { store.chimeConfig(for: role).selection.rawValue },
                        set: { rawValue in
                            if let selection = DictationChimeSelection(rawValue: rawValue) {
                                store.setChimeSelection(selection, for: role)
                            }
                        }
                    )) {
                        ForEach(DictationChimeSelection.allCases, id: \.rawValue) { selection in
                            Text(selection.title).tag(selection.rawValue)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 220, alignment: .trailing)

                    Button("Preview") {
                        store.previewChime(role)
                    }
                    .fixedSize(horizontal: true, vertical: false)
                }

                if store.chimeConfig(for: role).selection == .custom {
                    HStack(spacing: 8) {
                        Text(store.customChimeDisplayName(for: role))
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        Button("Choose…") {
                            store.chooseCustomChime(for: role)
                        }
                        .fixedSize(horizontal: true, vertical: false)
                    }
                }
            }
        }
    }

    private func audioInputTitle(for selection: AudioInputSelection) -> String {
        switch selection {
        case .systemDefault:
            if let defaultDevice = store.snapshot.availableAudioInputs.first(where: \.isDefault) {
                return "System Default — \(defaultDevice.name)"
            }
            return "System Default"
        case .device(let uid):
            if let device = store.snapshot.availableAudioInputs.first(where: { $0.uid == uid }) {
                return device.menuTitle
            }
            return "Unavailable device"
        }
    }
}

private struct InsetSettingsCard<Content: View>: View {
    let title: String
    let subtitle: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.system(size: 12, weight: .semibold))
            Text(subtitle)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
    }
}

private struct SettingsGroup<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        VStack(spacing: 0) {
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.black.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.6), lineWidth: 1)
        )
    }
}

private struct SettingsControlRow<Control: View>: View {
    let title: String
    let subtitle: String
    var topAligned = false
    @ViewBuilder let control: Control

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: topAligned ? .top : .center, spacing: 16) {
                Text(title)
                    .font(.body.weight(.medium))
                Spacer(minLength: 24)
                control
            }

            if !subtitle.isEmpty {
                Text(subtitle)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 620, alignment: .leading)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }
}

private struct SettingsBlockRow<Control: View>: View {
    let title: String
    let subtitle: String
    var muted = false
    @ViewBuilder let control: Control

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 16) {
                Text(title)
                    .font(.body.weight(.medium))
                Spacer(minLength: 24)
                control
            }

            if !subtitle.isEmpty {
                Text(subtitle)
                    .font(.footnote)
                    .foregroundStyle(muted ? .tertiary : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 620, alignment: .leading)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }
}

private struct SettingsDivider: View {
    var body: some View {
        Divider()
            .overlay(Color(nsColor: .separatorColor).opacity(0.5))
            .padding(.horizontal, 16)
    }
}

private struct SettingsSliderRow: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Text(label)
                .frame(width: 70, alignment: .leading)
            Slider(value: $value, in: range)
            Text(String(format: format, value))
                .font(.system(.footnote, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 48, alignment: .trailing)
        }
    }
}
