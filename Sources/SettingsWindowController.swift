import AppKit

@MainActor
final class SettingsWindowController: NSWindowController {
    var onDictationModeChange: ((DictationInteractionMode) -> Void)?
    var onDictationBindingChange: ((KeyBinding) -> Void)?
    var onDictationBindingRecordingChange: ((Bool) -> Void)?
    var onAudioInputSelectionChange: ((AudioInputSelection) -> Void)?
    var onServerModeChange: ((ServerMode) -> Void)?
    var onServerPortChange: ((UInt16) -> Void)?
    var onSaveRecordingsChange: ((Bool) -> Void)?
    var onChooseRecordingsDirectory: (() -> Void)?
    var onResetRecordingsDirectory: (() -> Void)?
    var onRevealRecordingsDirectory: (() -> Void)?
    var onModelPresetChange: ((Int) -> Void)?
    var onBatchCommitChange: ((Bool) -> Void)?
    var onApplyModelSpec: ((String) -> Void)?
    var onDownloadModel: ((String) -> Void)?

    private let customPresetTitle = "Custom configuration"
    private let rowLabelWidth: CGFloat = 170

    private let dictationModeControl = NSSegmentedControl(labels: ["Toggle", "Push to Talk"], trackingMode: .selectOne, target: nil, action: nil)
    private let shortcutRecorder = ShortcutRecorderView(defaultBinding: .ctrlBacktick)
    private let audioInputPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let audioInputDescriptionLabel = NSTextField(wrappingLabelWithString: "")
    private var audioInputSelections: [AudioInputSelection] = []

    private let modelPresetPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let presetDescriptionLabel = NSTextField(wrappingLabelWithString: "")
    private let batchCommitCheckbox = NSButton(checkboxWithTitle: "Use a batch pass when committing segments", target: nil, action: nil)
    private let modelField = NSTextField(frame: .zero)
    private let applyModelButton = NSButton(title: "Use", target: nil, action: nil)
    private let downloadModelPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let modelStatusLabel = NSTextField(labelWithString: "")
    private let advancedModelsHintLabel = NSTextField(wrappingLabelWithString: "Use custom model settings only if you want to override the selected profile. This same model is used for live decoding and batch segment commits.")
    private let modelEditor = NSStackView()

    private let saveRecordingsCheckbox = NSButton(checkboxWithTitle: "Save audio recordings", target: nil, action: nil)
    private let recordingsLocationLabel = NSTextField(wrappingLabelWithString: "")
    private let chooseRecordingsButton = NSButton(title: "Choose…", target: nil, action: nil)
    private let resetRecordingsButton = NSButton(title: "Reset Default", target: nil, action: nil)
    private let revealRecordingsButton = NSButton(title: "Reveal in Finder", target: nil, action: nil)
    private let recordingsLocationView = NSStackView()

    private let serverModePopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let serverModeDescriptionLabel = NSTextField(wrappingLabelWithString: "")
    private let serverPortField = NSTextField(frame: .zero)
    private let applyPortButton = NSButton(title: "Apply", target: nil, action: nil)
    private let portControlRow = NSStackView()

    init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 760, height: 760),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Yuwp Settings"
        window.center()
        window.minSize = NSSize(width: 720, height: 560)
        super.init(window: window)
        buildUI()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func sync(
        dictationMode: DictationInteractionMode,
        dictationBinding: KeyBinding,
        audioInputSelection: AudioInputSelection,
        availableAudioInputs: [AudioInputDeviceDescriptor],
        serverMode: ServerMode,
        serverPort: UInt16,
        transcriptionModel: String,
        batchCommitEnabled: Bool,
        saveRecordings: Bool,
        recordingsDir: URL,
        usingDefaultRecordingsDir: Bool
    ) {
        dictationModeControl.selectedSegment = dictationMode == .toggle ? 0 : 1
        shortcutRecorder.binding = dictationBinding
        reloadAudioInputPopup(availableAudioInputs, selection: audioInputSelection)
        audioInputDescriptionLabel.stringValue = audioInputDescription(
            for: audioInputSelection,
            availableInputs: availableAudioInputs
        )

        let currentPreset = ModelPreset.current()
        if let preset = currentPreset {
            modelPresetPopup.selectItem(at: ModelPreset.presets.firstIndex { $0.label == preset.label } ?? 0)
        } else {
            modelPresetPopup.selectItem(withTitle: customPresetTitle)
        }
        presetDescriptionLabel.stringValue = presetDescriptionText(currentPreset)

        batchCommitCheckbox.state = batchCommitEnabled ? .on : .off
        modelField.stringValue = transcriptionModel
        modelStatusLabel.stringValue = modelStatusText(
            spec: transcriptionModel,
            enabled: true,
            disabledText: nil
        )

        saveRecordingsCheckbox.state = saveRecordings ? .on : .off
        recordingsLocationLabel.stringValue = recordingsPathText(for: recordingsDir, usingDefault: usingDefaultRecordingsDir)
        recordingsLocationLabel.textColor = saveRecordings ? .secondaryLabelColor : .tertiaryLabelColor

        if let index = ServerMode.allCases.firstIndex(of: serverMode) {
            serverModePopup.selectItem(at: index)
        }
        serverModeDescriptionLabel.stringValue = serverModeDescription(for: serverMode)
        serverPortField.stringValue = "\(serverPort)"

        downloadModelPopup.selectItem(at: 0)

        setControlsEnabled(saveRecordings, in: recordingsLocationView)
        setControlsEnabled(serverMode != .off, in: portControlRow)
    }

    private func buildUI() {
        guard let contentView = window?.contentView else { return }

        dictationModeControl.target = self
        dictationModeControl.action = #selector(dictationModeChanged(_:))

        shortcutRecorder.onChange = { [weak self] binding in
            self?.onDictationBindingChange?(binding)
        }
        shortcutRecorder.onRecordingChange = { [weak self] isRecording in
            self?.onDictationBindingRecordingChange?(isRecording)
        }
        audioInputPopup.target = self
        audioInputPopup.action = #selector(audioInputChanged(_:))
        audioInputDescriptionLabel.textColor = .secondaryLabelColor
        audioInputDescriptionLabel.maximumNumberOfLines = 3

        modelPresetPopup.addItems(withTitles: ModelPreset.presets.map(\.label) + [customPresetTitle])
        modelPresetPopup.target = self
        modelPresetPopup.action = #selector(modelPresetChanged(_:))

        presetDescriptionLabel.textColor = .secondaryLabelColor
        presetDescriptionLabel.maximumNumberOfLines = 3

        batchCommitCheckbox.target = self
        batchCommitCheckbox.action = #selector(batchCommitChanged(_:))

        configureModelField(modelField)
        configureButton(applyModelButton)
        applyModelButton.target = self
        applyModelButton.action = #selector(applyModel(_:))
        configureDownloadPopup(downloadModelPopup, action: #selector(downloadModelSelectionChanged(_:)))
        configureModelEditor(
            modelEditor,
            field: modelField,
            applyButton: applyModelButton,
            downloadPopup: downloadModelPopup,
            statusLabel: modelStatusLabel
        )

        advancedModelsHintLabel.textColor = .secondaryLabelColor
        advancedModelsHintLabel.maximumNumberOfLines = 3

        saveRecordingsCheckbox.target = self
        saveRecordingsCheckbox.action = #selector(saveRecordingsChanged(_:))
        configureButton(chooseRecordingsButton)
        chooseRecordingsButton.target = self
        chooseRecordingsButton.action = #selector(chooseRecordingsDirectory(_:))
        configureButton(resetRecordingsButton)
        resetRecordingsButton.target = self
        resetRecordingsButton.action = #selector(resetRecordingsDirectory(_:))
        configureButton(revealRecordingsButton)
        revealRecordingsButton.target = self
        revealRecordingsButton.action = #selector(revealRecordingsDirectory(_:))
        recordingsLocationLabel.maximumNumberOfLines = 3
        configureRecordingsLocationView()

        serverModePopup.addItems(withTitles: ServerMode.allCases.map(\.description))
        serverModePopup.target = self
        serverModePopup.action = #selector(serverModeChanged(_:))
        serverModeDescriptionLabel.textColor = .secondaryLabelColor
        serverModeDescriptionLabel.maximumNumberOfLines = 3

        serverPortField.placeholderString = "9748"
        serverPortField.alignment = .right
        serverPortField.font = .monospacedDigitSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
        configureButton(applyPortButton)
        applyPortButton.target = self
        applyPortButton.action = #selector(applyPort(_:))
        configurePortRow()

        let header = makeHeader()
        let dictationSection = makeSection(
            title: "Dictation",
            subtitle: "Choose how Yuwp starts dictation and which shortcut triggers it.",
            body: [
                makeRow(label: "Shortcut", control: shortcutRecorder),
                makeRow(label: "Behavior", control: dictationModeControl),
                makeRow(label: "Microphone", control: makeControlWithCaption(audioInputPopup, captionLabel: audioInputDescriptionLabel)),
            ]
        )
        let transcriptionSection = makeSection(
            title: "Transcription",
            subtitle: "Choose a profile for speed or accuracy. Optionally run a slower batch pass whenever Yuwp commits a segment, including the trailing segment when you stop.",
            body: [
                makeRow(label: "Profile", control: makeControlWithCaption(modelPresetPopup, captionLabel: presetDescriptionLabel)),
                makeRow(label: "Segment Commit", control: batchCommitCheckbox),
                makeInsetSection(
                    title: "Advanced model settings",
                    subtitle: advancedModelsHintLabel,
                    body: [
                        makeRow(label: "Model", control: modelEditor),
                    ]
                ),
            ]
        )
        let recordingsSection = makeSection(
            title: "Recordings",
            subtitle: "Keep source audio if you want a paper trail for debugging, QA, or re-transcription later.",
            body: [
                makeRow(label: "Audio Files", control: saveRecordingsCheckbox),
                makeRow(label: "Recording Location", control: recordingsLocationView),
            ]
        )
        let networkSection = makeSection(
            title: "Network",
            subtitle: "Choose whether Yuwp exposes its local transcription server only to this Mac or to other devices on your local network.",
            body: [
                makeRow(label: "Availability", control: makeControlWithCaption(serverModePopup, captionLabel: serverModeDescriptionLabel)),
                makeRow(label: "Port", control: portControlRow),
            ]
        )

        let stack = NSStackView(views: [header, dictationSection, transcriptionSection, recordingsSection, networkSection])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 20
        stack.translatesAutoresizingMaskIntoConstraints = false

        for section in [dictationSection, transcriptionSection, recordingsSection, networkSection] {
            section.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        }

        let documentView = NSView()
        documentView.translatesAutoresizingMaskIntoConstraints = false
        documentView.addSubview(stack)

        let scrollView = NSScrollView()
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.documentView = documentView

        contentView.addSubview(scrollView)
        NSLayoutConstraint.activate([
            scrollView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            scrollView.topAnchor.constraint(equalTo: contentView.topAnchor),
            scrollView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),

            documentView.leadingAnchor.constraint(equalTo: scrollView.contentView.leadingAnchor),
            documentView.trailingAnchor.constraint(equalTo: scrollView.contentView.trailingAnchor),
            documentView.topAnchor.constraint(equalTo: scrollView.contentView.topAnchor),
            documentView.bottomAnchor.constraint(equalTo: scrollView.contentView.bottomAnchor),
            documentView.widthAnchor.constraint(equalTo: scrollView.contentView.widthAnchor),

            stack.leadingAnchor.constraint(equalTo: documentView.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: documentView.trailingAnchor, constant: -24),
            stack.topAnchor.constraint(equalTo: documentView.topAnchor, constant: 24),
            stack.bottomAnchor.constraint(equalTo: documentView.bottomAnchor, constant: -24),
        ])
    }

    private func makeHeader() -> NSView {
        let title = NSTextField(labelWithString: "Settings")
        title.font = .systemFont(ofSize: 28, weight: .semibold)

        let subtitle = helperLabel("Choose how Yuwp listens, transcribes, stores recordings, and shares its local server.")
        subtitle.maximumNumberOfLines = 2

        let stack = NSStackView(views: [title, subtitle])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 6
        return stack
    }

    private func makeSection(title: String, subtitle: String, body: [NSView]) -> NSView {
        let titleLabel = NSTextField(labelWithString: title)
        titleLabel.font = .systemFont(ofSize: 15, weight: .semibold)

        let subtitleLabel = helperLabel(subtitle)
        subtitleLabel.maximumNumberOfLines = 3

        let content = NSStackView(views: [titleLabel, subtitleLabel] + body)
        content.orientation = .vertical
        content.alignment = .leading
        content.spacing = 12

        return CardView(content: content, padding: NSEdgeInsets(top: 18, left: 18, bottom: 18, right: 18), cornerRadius: 14, fillColor: .controlBackgroundColor)
    }

    private func makeInsetSection(title: String, subtitle: NSTextField, body: [NSView]) -> NSView {
        let titleLabel = NSTextField(labelWithString: title)
        titleLabel.font = .systemFont(ofSize: 12, weight: .semibold)
        subtitle.maximumNumberOfLines = 4

        let content = NSStackView(views: [titleLabel, subtitle] + body)
        content.orientation = .vertical
        content.alignment = .leading
        content.spacing = 10

        return CardView(content: content, padding: NSEdgeInsets(top: 16, left: 16, bottom: 16, right: 16), cornerRadius: 12, fillColor: .windowBackgroundColor)
    }

    private func makeRow(label title: String, control: NSView) -> NSView {
        let grid = NSGridView(views: [[label(title), control]])
        grid.translatesAutoresizingMaskIntoConstraints = false
        grid.rowSpacing = 8
        grid.columnSpacing = 18
        grid.xPlacement = .leading
        grid.row(at: 0).yPlacement = .top
        grid.column(at: 0).width = rowLabelWidth
        return grid
    }

    private func makeControlWithCaption(_ control: NSView, captionLabel: NSTextField) -> NSView {
        let stack = NSStackView(views: [control, captionLabel])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 6
        return stack
    }

    private func configureModelEditor(
        _ container: NSStackView,
        field: NSTextField,
        applyButton: NSButton,
        downloadPopup: NSPopUpButton,
        statusLabel: NSTextField
    ) {
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.maximumNumberOfLines = 2

        let controls = NSStackView()
        controls.orientation = .horizontal
        controls.alignment = .centerY
        controls.spacing = 8
        controls.translatesAutoresizingMaskIntoConstraints = false

        field.setContentHuggingPriority(.defaultLow, for: .horizontal)
        field.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        field.widthAnchor.constraint(greaterThanOrEqualToConstant: 320).isActive = true
        downloadPopup.setContentHuggingPriority(.required, for: .horizontal)

        controls.addArrangedSubview(field)
        controls.addArrangedSubview(applyButton)
        controls.addArrangedSubview(downloadPopup)

        container.orientation = .vertical
        container.alignment = .leading
        container.spacing = 6
        container.addArrangedSubview(controls)
        container.addArrangedSubview(statusLabel)
    }

    private func configureRecordingsLocationView() {
        recordingsLocationView.orientation = .vertical
        recordingsLocationView.alignment = .leading
        recordingsLocationView.spacing = 8

        let buttons = NSStackView()
        buttons.orientation = .horizontal
        buttons.alignment = .centerY
        buttons.spacing = 8
        buttons.addArrangedSubview(chooseRecordingsButton)
        buttons.addArrangedSubview(resetRecordingsButton)
        buttons.addArrangedSubview(revealRecordingsButton)

        recordingsLocationView.addArrangedSubview(recordingsLocationLabel)
        recordingsLocationView.addArrangedSubview(buttons)
    }

    private func configurePortRow() {
        portControlRow.orientation = .horizontal
        portControlRow.alignment = .centerY
        portControlRow.spacing = 8

        serverPortField.widthAnchor.constraint(equalToConstant: 96).isActive = true
        portControlRow.addArrangedSubview(serverPortField)
        portControlRow.addArrangedSubview(applyPortButton)
        portControlRow.addArrangedSubview(helperLabel("1–65535"))
    }

    private func configureModelField(_ field: NSTextField) {
        field.placeholderString = "mlx-community/Qwen3-ASR-0.6B-4bit or /path/to/model"
        field.font = .monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
    }

    private func configureButton(_ button: NSButton) {
        button.bezelStyle = .rounded
    }

    private func configureDownloadPopup(_ popup: NSPopUpButton, action: Selector) {
        popup.removeAllItems()
        popup.addItem(withTitle: "Download…")
        for model in DownloadableASRModel.supported {
            popup.addItem(withTitle: model.label)
            popup.lastItem?.representedObject = model.repoId
        }
        popup.target = self
        popup.action = action
    }

    private func recordingsPathText(for url: URL, usingDefault: Bool) -> String {
        let home = NSHomeDirectory()
        let path = url.path.hasPrefix(home)
            ? "~" + String(url.path.dropFirst(home.count))
            : url.path
        return usingDefault ? "Default location: \(path)" : path
    }

    private func reloadAudioInputPopup(
        _ availableInputs: [AudioInputDeviceDescriptor],
        selection: AudioInputSelection
    ) {
        audioInputPopup.removeAllItems()
        audioInputSelections = [.systemDefault] + availableInputs.map(\.selection)

        if let defaultDevice = availableInputs.first(where: \.isDefault) {
            audioInputPopup.addItem(withTitle: "System Default — \(defaultDevice.name)")
        } else {
            audioInputPopup.addItem(withTitle: "System Default")
        }
        audioInputPopup.lastItem?.representedObject = AudioInputSelection.systemDefault.persistenceString

        for device in availableInputs {
            audioInputPopup.addItem(withTitle: device.menuTitle)
            audioInputPopup.lastItem?.representedObject = device.selection.persistenceString
        }

        if case .device(let uid) = selection,
           !availableInputs.contains(where: { $0.uid == uid }) {
            audioInputSelections.append(selection)
            audioInputPopup.addItem(withTitle: "Unavailable device")
            audioInputPopup.lastItem?.representedObject = selection.persistenceString
        }

        if let index = audioInputSelections.firstIndex(of: selection) {
            audioInputPopup.selectItem(at: index)
        } else {
            audioInputPopup.selectItem(at: 0)
        }
    }

    private func audioInputDescription(
        for selection: AudioInputSelection,
        availableInputs: [AudioInputDeviceDescriptor]
    ) -> String {
        switch selection {
        case .systemDefault:
            if let defaultDevice = availableInputs.first(where: \.isDefault) {
                return "Follows the current macOS default input: \(defaultDevice.detailText). Best when you switch microphones often."
            }
            return "Follows the current macOS default input device."
        case .device(let uid):
            if let device = availableInputs.first(where: { $0.uid == uid }) {
                return "Pinned to \(device.detailText). Yuwp will capture this device’s native format and convert it to 16 kHz mono for transcription."
            }
            return "The selected device is currently unavailable. Yuwp will fall back to the system default input until it reconnects."
        }
    }

    private func presetDescriptionText(_ preset: ModelPreset?) -> String {
        guard let preset else {
            return "Using a custom model configuration. The advanced model settings below control transcription."
        }
        switch preset.label {
        case "Fast":
            return "Lower latency with the smaller model. Best default for quick local dictation."
        case "Best Accuracy":
            return "Uses the larger model for better recognition quality, at the cost of more compute."
        default:
            return preset.summary
        }
    }

    private func serverModeDescription(for mode: ServerMode) -> String {
        switch mode {
        case .off:
            return "Turns off Yuwp’s bundled transcription server. Dictation won’t work until you turn it back on."
        case .localhost:
            return "Only Yuwp and other apps on this Mac can connect to the server."
        case .allInterfaces:
            return "Makes the server available on your local network so other devices can connect to this Mac."
        }
    }

    private func modelStatusText(spec: String, enabled: Bool, disabledText: String?) -> String {
        let name = ModelLocator.displayName(for: spec)
        guard enabled else { return disabledText ?? "Disabled" }
        let installed = ModelLocator.resolve(spec) != nil
        return installed ? "Installed: \(name)" : "Missing: \(name)"
    }

    private func setControlsEnabled(_ enabled: Bool, in view: NSView) {
        if let control = view as? NSControl {
            control.isEnabled = enabled
        }
        view.alphaValue = enabled ? 1.0 : 0.55
        for subview in view.subviews {
            setControlsEnabled(enabled, in: subview)
        }
    }

    private func label(_ string: String) -> NSTextField {
        let field = NSTextField(labelWithString: string)
        field.lineBreakMode = .byWordWrapping
        return field
    }

    private func helperLabel(_ string: String) -> NSTextField {
        let field = label(string)
        field.textColor = .secondaryLabelColor
        return field
    }

    @objc private func dictationModeChanged(_ sender: NSSegmentedControl) {
        let mode: DictationInteractionMode = sender.selectedSegment == 0 ? .toggle : .pushToTalk
        onDictationModeChange?(mode)
    }

    @objc private func audioInputChanged(_ sender: NSPopUpButton) {
        guard sender.indexOfSelectedItem >= 0,
              sender.indexOfSelectedItem < audioInputSelections.count else { return }
        onAudioInputSelectionChange?(audioInputSelections[sender.indexOfSelectedItem])
    }

    @objc private func modelPresetChanged(_ sender: NSPopUpButton) {
        let index = sender.indexOfSelectedItem
        guard index >= 0, index < ModelPreset.presets.count else { return }
        onModelPresetChange?(index)
    }

    @objc private func batchCommitChanged(_ sender: NSButton) {
        onBatchCommitChange?(sender.state == .on)
    }

    @objc private func applyModel(_ sender: NSButton) {
        onApplyModelSpec?(modelField.stringValue)
    }

    @objc private func downloadModelSelectionChanged(_ sender: NSPopUpButton) {
        guard sender.indexOfSelectedItem > 0,
              let repoId = sender.selectedItem?.representedObject as? String else { return }
        sender.selectItem(at: 0)
        onDownloadModel?(repoId)
    }

    @objc private func saveRecordingsChanged(_ sender: NSButton) {
        onSaveRecordingsChange?(sender.state == .on)
    }

    @objc private func chooseRecordingsDirectory(_ sender: NSButton) {
        onChooseRecordingsDirectory?()
    }

    @objc private func resetRecordingsDirectory(_ sender: NSButton) {
        onResetRecordingsDirectory?()
    }

    @objc private func revealRecordingsDirectory(_ sender: NSButton) {
        onRevealRecordingsDirectory?()
    }

    @objc private func serverModeChanged(_ sender: NSPopUpButton) {
        guard sender.indexOfSelectedItem >= 0,
              sender.indexOfSelectedItem < ServerMode.allCases.count else { return }
        onServerModeChange?(ServerMode.allCases[sender.indexOfSelectedItem])
    }

    @objc private func applyPort(_ sender: NSButton) {
        let trimmed = serverPortField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int(trimmed), (1...65_535).contains(value), let port = UInt16(exactly: value) else {
            NSSound.beep()
            return
        }
        onServerPortChange?(port)
        serverPortField.stringValue = "\(port)"
    }
}

private final class CardView: NSView {
    private let fillColor: NSColor
    private let cornerRadius: CGFloat

    init(content: NSView, padding: NSEdgeInsets, cornerRadius: CGFloat, fillColor: NSColor) {
        self.fillColor = fillColor
        self.cornerRadius = cornerRadius
        super.init(frame: .zero)
        translatesAutoresizingMaskIntoConstraints = false
        wantsLayer = true
        updateLayerStyle()

        content.translatesAutoresizingMaskIntoConstraints = false
        addSubview(content)
        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: leadingAnchor, constant: padding.left),
            content.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -padding.right),
            content.topAnchor.constraint(equalTo: topAnchor, constant: padding.top),
            content.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -padding.bottom),
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        updateLayerStyle()
    }

    private func updateLayerStyle() {
        layer?.cornerRadius = cornerRadius
        layer?.borderWidth = 1
        layer?.borderColor = NSColor.separatorColor.cgColor
        layer?.backgroundColor = fillColor.cgColor
    }
}

@MainActor
private final class ShortcutRecorderView: NSStackView {
    var onChange: ((KeyBinding) -> Void)?
    var onRecordingChange: ((Bool) -> Void)?
    var binding: KeyBinding {
        didSet {
            guard !isRecording else { return }
            bindingLabel.stringValue = binding.description
        }
    }

    private let defaultBinding: KeyBinding
    private let bindingLabel = NSTextField(labelWithString: "")
    private let recordButton = NSButton(title: "Record…", target: nil, action: nil)
    private let resetButton = NSButton(title: "Reset Default", target: nil, action: nil)
    private var recordingMonitor: Any?
    private var isRecording = false
    private var captureState = ShortcutCaptureState()

    init(defaultBinding: KeyBinding) {
        self.defaultBinding = defaultBinding
        self.binding = defaultBinding
        super.init(frame: .zero)
        orientation = .horizontal
        alignment = .centerY
        spacing = 8
        setup()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setup() {
        bindingLabel.font = .monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
        bindingLabel.stringValue = binding.description
        bindingLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        bindingLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 120).isActive = true

        recordButton.target = self
        recordButton.action = #selector(toggleRecording(_:))
        recordButton.bezelStyle = .rounded

        resetButton.target = self
        resetButton.action = #selector(resetToDefault(_:))
        resetButton.bezelStyle = .rounded

        addArrangedSubview(bindingLabel)
        addArrangedSubview(recordButton)
        addArrangedSubview(resetButton)
    }

    @objc private func toggleRecording(_ sender: NSButton) {
        isRecording ? stopRecording() : startRecording()
    }

    @objc private func resetToDefault(_ sender: NSButton) {
        binding = defaultBinding
        onChange?(defaultBinding)
    }

    private func startRecording() {
        guard !isRecording else { return }
        captureState = ShortcutCaptureState()
        isRecording = true
        bindingLabel.stringValue = "Press shortcut…"
        recordButton.title = "Cancel"
        onRecordingChange?(true)

        recordingMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) { [weak self] event in
            guard let self else { return event }

            let result: ShortcutCaptureResult
            switch event.type {
            case .keyDown:
                result = self.captureState.handleKeyDown(
                    keyCode: UInt16(event.keyCode),
                    modifiers: Self.modifierMask(from: event.modifierFlags)
                )
            case .flagsChanged:
                result = self.captureState.handleFlagsChanged(keyCode: UInt16(event.keyCode))
                if let pendingModifierKeyCode = self.captureState.pendingModifierKeyCode {
                    self.bindingLabel.stringValue = "Release \(KeyBinding.keyName(for: pendingModifierKeyCode)) to use it, or press another key for a combo…"
                }
            default:
                return event
            }

            return self.handleCaptureResult(result)
        }
    }

    private func stopRecording() {
        let wasRecording = isRecording
        if let monitor = recordingMonitor {
            NSEvent.removeMonitor(monitor)
            recordingMonitor = nil
        }
        captureState = ShortcutCaptureState()
        isRecording = false
        bindingLabel.stringValue = binding.description
        recordButton.title = "Record…"
        if wasRecording {
            onRecordingChange?(false)
        }
    }

    private func handleCaptureResult(_ result: ShortcutCaptureResult) -> NSEvent? {
        switch result {
        case .none:
            return nil
        case .captured(let binding):
            self.binding = binding
            onChange?(binding)
            stopRecording()
            return nil
        case .cancelled:
            stopRecording()
            return nil
        case .invalid:
            NSSound.beep()
            bindingLabel.stringValue = "Use a modifier combo, or press and release one modifier."
            return nil
        }
    }

    private static func modifierMask(from flags: NSEvent.ModifierFlags) -> UInt64 {
        let filtered = flags.intersection(.deviceIndependentFlagsMask)
        var result: UInt64 = 0
        if filtered.contains(.control) { result |= 0x40000 }
        if filtered.contains(.option) { result |= 0x80000 }
        if filtered.contains(.command) { result |= 0x100000 }
        if filtered.contains(.shift) { result |= 0x20000 }
        return result
    }
}
