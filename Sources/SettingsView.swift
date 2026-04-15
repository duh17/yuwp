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
            "Choose the transcription model and whether to run a final accuracy pass when text settles."
        case .recordings:
            "Keep source audio if you want a paper trail for debugging, QA, or re-transcription later."
        case .network:
            "Choose whether Yuwp exposes its local transcription server only to this Mac or to other devices on your local network."
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
                .frame(width: 360, height: 30, alignment: .trailing)
            }

            SettingsDivider()

            SettingsControlRow(
                title: "Microphone",
                subtitle: store.audioInputDescriptionText,
                topAligned: true
            ) {
                Picker("", selection: Binding(
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
                Toggle("", isOn: Binding(
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
                Toggle("", isOn: Binding(
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

                    Button("Choose…") {
                        store.chooseModelDirectory()
                    }

                    Button("Use") {
                        store.applyTranscriptionModelDraft()
                    }
                }
            }

            SettingsDivider()

            SettingsControlRow(
                title: "Final Accuracy Pass",
                subtitle: "Retranscribe settled segments to improve final accuracy. Adds a little commit latency."
            ) {
                Toggle("", isOn: Binding(
                    get: { store.snapshot.batchCommitEnabled },
                    set: { store.setBatchCommitEnabled($0) }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }

            SettingsDivider()

            SettingsBlockRow(
                title: "Download",
                subtitle: store.downloadRowStatusText
            ) {
                HStack(spacing: 8) {
                    Picker("", selection: $store.selectedDownloadModelRepoId) {
                        ForEach(store.downloadableModels, id: \.repoId) { model in
                            Text(model.label).tag(model.repoId)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 260, alignment: .trailing)
                    .disabled(store.modelDownloadStatusText != nil)

                    Button(store.downloadButtonTitle) {
                        store.downloadSelectedModel()
                    }
                    .disabled(!store.canDownloadSelectedModel)
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
                Toggle("", isOn: Binding(
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
                Toggle("", isOn: Binding(
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
                    Button("Reset Default") { store.resetRecordingsDirectory() }
                    Button("Reveal in Finder") { store.revealRecordingsDirectory() }
                }
                .disabled(!store.snapshot.saveRecordings)
            }
            .opacity(store.snapshot.saveRecordings ? 1.0 : 0.55)
        }
    }

    private var networkContent: some View {
        SettingsGroup {
            SettingsControlRow(
                title: "Availability",
                subtitle: store.serverModeDescriptionText,
                topAligned: true
            ) {
                Picker("", selection: Binding(
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

            SettingsBlockRow(
                title: "Port",
                subtitle: "Use a custom port if you need Yuwp to avoid another local service.",
                muted: store.snapshot.serverMode == .off
            ) {
                HStack(spacing: 8) {
                    TextField("9748", text: $store.serverPortDraft)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 110)

                    Button("Apply") {
                        store.applyServerPortDraft()
                    }

                    Text("1–65535")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .disabled(store.snapshot.serverMode == .off)
                .opacity(store.snapshot.serverMode == .off ? 0.55 : 1.0)
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
                    Picker("", selection: Binding(
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
                    Picker("", selection: Binding(
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
                }

                if store.chimeConfig(for: role).selection == .custom {
                    HStack(spacing: 8) {
                        Text(store.customChimeDisplayName(for: role))
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        Button("Choose…") {
                            store.chooseCustomChime(for: role)
                        }
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
