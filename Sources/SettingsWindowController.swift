import AppKit

@MainActor
final class SettingsWindowController: NSWindowController {
    var onDictationModeChange: ((DictationInteractionMode) -> Void)?
    var onDictationBindingChange: ((KeyBinding) -> Void)?
    var onServerModeChange: ((ServerMode) -> Void)?
    var onServerPortChange: ((UInt16) -> Void)?
    var onSaveRecordingsChange: ((Bool) -> Void)?
    var onChooseRecordingsDirectory: (() -> Void)?
    var onResetRecordingsDirectory: (() -> Void)?
    var onRevealRecordingsDirectory: (() -> Void)?
    var onModelPresetChange: ((Int) -> Void)?
    var onBatchRetranscribeChange: ((Bool) -> Void)?
    var onApplyStreamingModelSpec: ((String) -> Void)?
    var onApplyBatchModelSpec: ((String) -> Void)?
    var onDownloadStreamingModel: ((String) -> Void)?
    var onDownloadBatchModel: ((String) -> Void)?

    private let dictationModeControl = NSSegmentedControl(labels: ["Toggle", "Push to Talk"], trackingMode: .selectOne, target: nil, action: nil)
    private let shortcutRecorder = ShortcutRecorderView(defaultBinding: .ctrlBacktick)

    private let modelPresetPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let batchRetranscribeCheckbox = NSButton(checkboxWithTitle: "Use a final pass after stop for better accuracy", target: nil, action: nil)
    private let streamingModelField = NSTextField(frame: .zero)
    private let applyStreamingModelButton = NSButton(title: "Apply", target: nil, action: nil)
    private let downloadStreamingPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let streamingStatusLabel = NSTextField(labelWithString: "")
    private let batchModelField = NSTextField(frame: .zero)
    private let applyBatchModelButton = NSButton(title: "Apply", target: nil, action: nil)
    private let downloadBatchPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let batchStatusLabel = NSTextField(labelWithString: "")

    private let saveRecordingsCheckbox = NSButton(checkboxWithTitle: "Keep audio files after each dictation", target: nil, action: nil)
    private let recordingsLocationLabel = NSTextField(wrappingLabelWithString: "")
    private let chooseRecordingsButton = NSButton(title: "Choose…", target: nil, action: nil)
    private let resetRecordingsButton = NSButton(title: "Reset Default", target: nil, action: nil)
    private let revealRecordingsButton = NSButton(title: "Reveal in Finder", target: nil, action: nil)

    private let serverModePopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let serverPortField = NSTextField(frame: .zero)
    private let applyPortButton = NSButton(title: "Apply", target: nil, action: nil)

    init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 720, height: 640),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Yuwp Settings"
        window.center()
        window.minSize = NSSize(width: 680, height: 520)
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
        serverMode: ServerMode,
        serverPort: UInt16,
        streamingModel: String,
        batchModel: String,
        batchRetranscribeEnabled: Bool,
        saveRecordings: Bool,
        recordingsDir: URL,
        usingDefaultRecordingsDir: Bool
    ) {
        dictationModeControl.selectedSegment = dictationMode == .toggle ? 0 : 1
        shortcutRecorder.binding = dictationBinding

        if let preset = ModelPreset.current() {
            modelPresetPopup.selectItem(at: ModelPreset.presets.firstIndex { $0.label == preset.label } ?? 0)
        } else {
            modelPresetPopup.selectItem(withTitle: "Custom")
        }

        batchRetranscribeCheckbox.state = batchRetranscribeEnabled ? .on : .off
        streamingModelField.stringValue = streamingModel
        batchModelField.stringValue = batchModel
        streamingStatusLabel.stringValue = modelStatusText(label: "Streaming", spec: streamingModel, enabled: true)
        batchStatusLabel.stringValue = modelStatusText(label: "Final", spec: batchModel, enabled: batchRetranscribeEnabled)
        batchStatusLabel.textColor = batchRetranscribeEnabled ? .secondaryLabelColor : .tertiaryLabelColor

        saveRecordingsCheckbox.state = saveRecordings ? .on : .off
        recordingsLocationLabel.stringValue = recordingsPathText(for: recordingsDir, usingDefault: usingDefaultRecordingsDir)

        if let index = ServerMode.allCases.firstIndex(of: serverMode) {
            serverModePopup.selectItem(at: index)
        }
        serverPortField.stringValue = "\(serverPort)"

        downloadStreamingPopup.selectItem(at: 0)
        downloadBatchPopup.selectItem(at: 0)
    }

    private func buildUI() {
        guard let contentView = window?.contentView else { return }

        dictationModeControl.target = self
        dictationModeControl.action = #selector(dictationModeChanged(_:))

        shortcutRecorder.onChange = { [weak self] binding in
            self?.onDictationBindingChange?(binding)
        }

        modelPresetPopup.addItems(withTitles: ModelPreset.presets.map(\.label) + ["Custom"])
        modelPresetPopup.target = self
        modelPresetPopup.action = #selector(modelPresetChanged(_:))

        batchRetranscribeCheckbox.target = self
        batchRetranscribeCheckbox.action = #selector(batchRetranscribeChanged(_:))

        configureModelField(streamingModelField)
        applyStreamingModelButton.target = self
        applyStreamingModelButton.action = #selector(applyStreamingModel(_:))
        configureDownloadPopup(downloadStreamingPopup, action: #selector(downloadStreamingSelectionChanged(_:)))

        configureModelField(batchModelField)
        applyBatchModelButton.target = self
        applyBatchModelButton.action = #selector(applyBatchModel(_:))
        configureDownloadPopup(downloadBatchPopup, action: #selector(downloadBatchSelectionChanged(_:)))

        streamingStatusLabel.textColor = .secondaryLabelColor
        batchStatusLabel.textColor = .secondaryLabelColor

        saveRecordingsCheckbox.target = self
        saveRecordingsCheckbox.action = #selector(saveRecordingsChanged(_:))
        chooseRecordingsButton.target = self
        chooseRecordingsButton.action = #selector(chooseRecordingsDirectory(_:))
        resetRecordingsButton.target = self
        resetRecordingsButton.action = #selector(resetRecordingsDirectory(_:))
        revealRecordingsButton.target = self
        revealRecordingsButton.action = #selector(revealRecordingsDirectory(_:))
        recordingsLocationLabel.textColor = .secondaryLabelColor
        recordingsLocationLabel.maximumNumberOfLines = 3

        serverModePopup.addItems(withTitles: ServerMode.allCases.map(\.description))
        serverModePopup.target = self
        serverModePopup.action = #selector(serverModeChanged(_:))

        serverPortField.placeholderString = "9748"
        serverPortField.alignment = .right
        serverPortField.font = .monospacedDigitSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
        applyPortButton.target = self
        applyPortButton.action = #selector(applyPort(_:))
        applyPortButton.bezelStyle = .rounded

        let stack = NSStackView(views: [
            makeIntro(),
            makeSection(title: "General", body: [
                makeRow(label: "Dictation Mode", control: dictationModeControl),
                makeRow(label: "Shortcut", control: shortcutRecorder),
            ]),
            makeSection(title: "Models", body: [
                helperLabel("Keep model configuration here. You can paste a Hugging Face repo id or a local model folder path, then click Apply."),
                makeRow(label: "Preset", control: modelPresetPopup),
                makeRow(label: "Final Pass", control: batchRetranscribeCheckbox),
                makeRow(label: "Streaming Model", control: makeModelEditor(
                    field: streamingModelField,
                    applyButton: applyStreamingModelButton,
                    downloadPopup: downloadStreamingPopup,
                    statusLabel: streamingStatusLabel
                )),
                makeRow(label: "Final Model", control: makeModelEditor(
                    field: batchModelField,
                    applyButton: applyBatchModelButton,
                    downloadPopup: downloadBatchPopup,
                    statusLabel: batchStatusLabel
                )),
            ]),
            makeSection(title: "Recordings", body: [
                makeRow(label: "Save Recordings", control: saveRecordingsCheckbox),
                makeRow(label: "Save Location", control: makeRecordingsLocationView()),
            ]),
            makeSection(title: "Server", body: [
                helperLabel("Yuwp always talks to the server over localhost. Exposing 0.0.0.0 only matters if you want LAN clients to connect."),
                makeRow(label: "Server Mode", control: serverModePopup),
                makeRow(label: "Server Port", control: makePortRow()),
            ]),
        ])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 18
        stack.translatesAutoresizingMaskIntoConstraints = false

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

            stack.leadingAnchor.constraint(equalTo: documentView.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: documentView.trailingAnchor, constant: -20),
            stack.topAnchor.constraint(equalTo: documentView.topAnchor, constant: 20),
            stack.bottomAnchor.constraint(equalTo: documentView.bottomAnchor, constant: -20),
        ])
    }

    private func makeIntro() -> NSView {
        let intro = helperLabel("Use the menu bar for quick actions. Use Settings for models, recordings, the dictation shortcut, and server behavior.")
        intro.maximumNumberOfLines = 2
        return intro
    }

    private func makeSection(title: String, body: [NSView]) -> NSView {
        let titleLabel = NSTextField(labelWithString: title)
        titleLabel.font = .boldSystemFont(ofSize: NSFont.systemFontSize)

        let section = NSStackView(views: [titleLabel] + body)
        section.orientation = .vertical
        section.alignment = .leading
        section.spacing = 10
        section.translatesAutoresizingMaskIntoConstraints = false
        return section
    }

    private func makeRow(label title: String, control: NSView) -> NSView {
        let row = NSGridView(views: [[label(title), control]])
        row.translatesAutoresizingMaskIntoConstraints = false
        row.rowSpacing = 8
        row.columnSpacing = 16
        row.xPlacement = .leading
        row.column(at: 0).width = 120
        return row
    }

    private func makeModelEditor(
        field: NSTextField,
        applyButton: NSButton,
        downloadPopup: NSPopUpButton,
        statusLabel: NSTextField
    ) -> NSView {
        let controls = NSStackView()
        controls.orientation = .horizontal
        controls.alignment = .centerY
        controls.spacing = 8
        field.setContentHuggingPriority(.defaultLow, for: .horizontal)
        applyButton.bezelStyle = .rounded
        controls.addArrangedSubview(field)
        controls.addArrangedSubview(applyButton)
        controls.addArrangedSubview(downloadPopup)

        let stack = NSStackView(views: [statusLabel, controls])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 6
        return stack
    }

    private func makeRecordingsLocationView() -> NSView {
        let buttonRow = NSStackView()
        buttonRow.orientation = .horizontal
        buttonRow.alignment = .centerY
        buttonRow.spacing = 8
        buttonRow.addArrangedSubview(chooseRecordingsButton)
        buttonRow.addArrangedSubview(resetRecordingsButton)
        buttonRow.addArrangedSubview(revealRecordingsButton)

        let stack = NSStackView(views: [recordingsLocationLabel, buttonRow])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 8
        return stack
    }

    private func makePortRow() -> NSView {
        let stack = NSStackView()
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.spacing = 8

        serverPortField.widthAnchor.constraint(equalToConstant: 90).isActive = true
        stack.addArrangedSubview(serverPortField)
        stack.addArrangedSubview(applyPortButton)

        let hint = helperLabel("1–65535")
        stack.addArrangedSubview(hint)

        return stack
    }

    private func configureModelField(_ field: NSTextField) {
        field.placeholderString = "mlx-community/Qwen3-ASR-0.6B-4bit or /path/to/model"
        field.font = .monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
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
        return usingDefault ? "Default: \(path)" : path
    }

    private func modelStatusText(label: String, spec: String, enabled: Bool) -> String {
        let name = ModelLocator.displayName(for: spec)
        guard enabled else { return "\(label): \(name) (disabled)" }
        return "\(label): \(name) \(ModelLocator.resolve(spec) != nil ? "✓ installed" : "⚠ missing")"
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

    @objc private func modelPresetChanged(_ sender: NSPopUpButton) {
        let index = sender.indexOfSelectedItem
        guard index >= 0, index < ModelPreset.presets.count else { return }
        onModelPresetChange?(index)
    }

    @objc private func batchRetranscribeChanged(_ sender: NSButton) {
        onBatchRetranscribeChange?(sender.state == .on)
    }

    @objc private func applyStreamingModel(_ sender: NSButton) {
        onApplyStreamingModelSpec?(streamingModelField.stringValue)
    }

    @objc private func applyBatchModel(_ sender: NSButton) {
        onApplyBatchModelSpec?(batchModelField.stringValue)
    }

    @objc private func downloadStreamingSelectionChanged(_ sender: NSPopUpButton) {
        guard sender.indexOfSelectedItem > 0,
              let repoId = sender.selectedItem?.representedObject as? String else { return }
        sender.selectItem(at: 0)
        onDownloadStreamingModel?(repoId)
    }

    @objc private func downloadBatchSelectionChanged(_ sender: NSPopUpButton) {
        guard sender.indexOfSelectedItem > 0,
              let repoId = sender.selectedItem?.representedObject as? String else { return }
        sender.selectItem(at: 0)
        onDownloadBatchModel?(repoId)
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

@MainActor
private final class ShortcutRecorderView: NSStackView {
    var onChange: ((KeyBinding) -> Void)?
    var binding: KeyBinding {
        didSet { bindingLabel.stringValue = binding.description }
    }

    private let defaultBinding: KeyBinding
    private let bindingLabel = NSTextField(labelWithString: "")
    private let recordButton = NSButton(title: "Record…", target: nil, action: nil)
    private let resetButton = NSButton(title: "Reset Default", target: nil, action: nil)
    private var recordingMonitor: Any?
    private var isRecording = false

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
        isRecording = true
        bindingLabel.stringValue = "Press shortcut…"
        recordButton.title = "Cancel"

        recordingMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            if event.keyCode == 53 { // Esc
                self.stopRecording()
                return nil
            }

            let modifiers = Self.modifierMask(from: event.modifierFlags)
            guard modifiers != 0 else {
                NSSound.beep()
                return nil
            }

            let binding = KeyBinding(keyCode: UInt16(event.keyCode), modifiers: modifiers)
            self.binding = binding
            self.stopRecording()
            self.onChange?(binding)
            return nil
        }
    }

    private func stopRecording() {
        if let monitor = recordingMonitor {
            NSEvent.removeMonitor(monitor)
            recordingMonitor = nil
        }
        isRecording = false
        bindingLabel.stringValue = binding.description
        recordButton.title = "Record…"
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
