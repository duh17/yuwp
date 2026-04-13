import AppKit

/// Floating dictation indicator pill.
///
/// Two modes:
///   - **Minimal**: narrow pill with blue animated bars (live injection into terminal/text field)
///   - **Full**: wide pill showing transcript text with border glow (clipboard fallback)
///
/// Draggable — remembers pinned position across sessions.
@MainActor
final class MicPanel: NSObject, NSWindowDelegate {
    var animationConfig: MicPanelAnimationConfig = .default {
        didSet {
            applyAnimationStyle()
        }
    }

    private var panel: NSPanel?
    private var contentView: NSView?
    private var textView: NSTextView?
    private var bars: [CALayer] = []
    private var isMinimal = false

    // Escape key dismiss
    var onDismiss: (() -> Void)?
    private var escapeMonitor: Any?

    // Remembers last user-dragged position
    private var pinnedOrigin: NSPoint?

    // Animation state
    private var animationTimer: Timer?
    private var targetLevel: Float = 0
    private var smoothLevel: Float = 0
    private var barPhase: Float = 0

    // Layout
    private let fullWidth: CGFloat = 320
    private let minimalWidth: CGFloat = 80
    private let pillHeight: CGFloat = 32
    private let cornerRadius: CGFloat = 16 // fully rounded ends
    private let textPadding: CGFloat = 14
    private let verticalPadding: CGFloat = 8
    private let screenEdgePadding: CGFloat = 8
    private let bottomPadding: CGFloat = 36
    private var lastProgrammaticOrigin: NSPoint?

    // Bar waveform
    private let barCount = 5
    private let barWidth: CGFloat = 3.0
    private let barGap: CGFloat = 3.0
    private let barScale: [CGFloat] = [0.5, 0.8, 1.0, 0.8, 0.5]
    private let barPhaseOffset: [Float] = [0, 0.7, 1.4, 2.1, 2.8]
    private let barColor = NSColor(calibratedRed: 0.4, green: 0.6, blue: 1.0, alpha: 1.0) // blue

    // Glow
    private let glowColor = NSColor(calibratedRed: 0.4, green: 0.6, blue: 1.0, alpha: 1.0)

    private var animationTuning: MicPanelAnimationTuning {
        animationConfig.resolvedTuning(reduceMotion: NSWorkspace.shared.accessibilityDisplayShouldReduceMotion)
    }

    // MARK: - Public API

    /// Show the pill at the pinned position, or bottom-center on the active screen.
    /// - `minimal: true` — narrow pill with waveform bars (live injection mode)
    /// - `minimal: false` — wide pill with transcript text (clipboard fallback)
    func show(minimal: Bool = false) {
        isMinimal = minimal
        if panel == nil { createPanel() }
        guard let panel else { return }

        let width = minimal ? minimalWidth : fullWidth
        let size = NSSize(width: width, height: pillHeight)

        panel.setContentSize(size)
        contentView?.frame = NSRect(origin: .zero, size: size)
        textView?.isHidden = minimal
        bars.forEach { $0.isHidden = !minimal }

        if minimal {
            layoutBars(in: width)
        }

        placePanel(at: resolvedOrigin(for: size))

        if !minimal {
            textView?.string = ""
            resizeToFit()
        }

        panel.alphaValue = 1
        panel.orderFront(nil)
        startAnimation()
        startEscapeMonitor()
    }

    func present(_ state: DictationPresentationState) {
        let minimal = state.bubbleStyle == .compact
        if panel?.isVisible != true || isMinimal != minimal {
            show(minimal: minimal)
        }
        if minimal {
            textView?.string = ""
        } else {
            updateTranscript(state.displayText)
        }
    }

    func updateTranscript(_ text: String) {
        guard !isMinimal else { return }
        textView?.string = text
        resizeToFit()
    }

    /// Feed normalized audio level (0.0-1.0).
    func updateAudioLevel(_ level: Float) {
        targetLevel = level
    }

    func hide() {
        stopAnimation()
        stopEscapeMonitor()
        panel?.orderOut(nil)
        targetLevel = 0
        smoothLevel = 0
    }

    // MARK: - Escape Key

    private func startEscapeMonitor() {
        guard escapeMonitor == nil else { return }
        // Use global monitor — local monitor can't see key events because
        // the panel is .nonactivatingPanel (events go to the focused app).
        // Capture the dismiss closure directly instead of touching the
        // @MainActor-isolated MicPanel instance from the monitor callback.
        let dismiss = onDismiss
        escapeMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { event in
            guard event.keyCode == 53 else { return }
            dismiss?()
        }
    }

    private func stopEscapeMonitor() {
        if let monitor = escapeMonitor {
            NSEvent.removeMonitor(monitor)
            escapeMonitor = nil
        }
    }

    // MARK: - Screen Placement

    func windowDidMove(_ notification: Notification) {
        guard let panel else { return }
        let origin = panel.frame.origin
        if let lastProgrammaticOrigin, pointsEqual(origin, lastProgrammaticOrigin) {
            self.lastProgrammaticOrigin = nil
            return
        }

        let midpoint = NSPoint(x: panel.frame.midX, y: panel.frame.midY)
        pinnedOrigin = clampToVisibleFrame(
            origin,
            panelSize: panel.frame.size,
            visibleFrame: visibleFrame(containing: midpoint)
        )
    }

    private func placePanel(at origin: NSPoint) {
        guard let panel else { return }
        lastProgrammaticOrigin = origin
        panel.setFrameOrigin(origin)
    }

    private func resolvedOrigin(for panelSize: NSSize) -> NSPoint {
        if let pinned = pinnedOrigin {
            return clampToVisibleFrame(
                pinned,
                panelSize: panelSize,
                visibleFrame: visibleFrame(containing: pinned)
            )
        }
        return defaultOrigin(for: panelSize)
    }

    private func defaultOrigin(for panelSize: NSSize) -> NSPoint {
        guard let visibleFrame = activeVisibleFrame() ?? NSScreen.main?.visibleFrame else {
            return .zero
        }
        return clampToVisibleFrame(
            NSPoint(
                x: visibleFrame.midX - panelSize.width / 2,
                y: visibleFrame.minY + bottomPadding
            ),
            panelSize: panelSize,
            visibleFrame: visibleFrame
        )
    }

    private func activeVisibleFrame() -> NSRect? {
        visibleFrame(containing: NSEvent.mouseLocation)
    }

    private func visibleFrame(containing point: NSPoint) -> NSRect? {
        NSScreen.screens.first(where: { NSMouseInRect(point, $0.frame, false) })?.visibleFrame
    }

    private func clampToVisibleFrame(
        _ point: NSPoint,
        panelSize: NSSize,
        visibleFrame: NSRect?
    ) -> NSPoint {
        guard let visibleFrame = visibleFrame ?? activeVisibleFrame() ?? NSScreen.main?.visibleFrame else {
            return point
        }
        var x = point.x
        var y = point.y
        if x + panelSize.width > visibleFrame.maxX { x = visibleFrame.maxX - panelSize.width - screenEdgePadding }
        if x < visibleFrame.minX { x = visibleFrame.minX + screenEdgePadding }
        if y + panelSize.height > visibleFrame.maxY { y = visibleFrame.maxY - panelSize.height - screenEdgePadding }
        if y < visibleFrame.minY { y = visibleFrame.minY + screenEdgePadding }
        return NSPoint(x: x, y: y)
    }

    private func pointsEqual(_ lhs: NSPoint, _ rhs: NSPoint, tolerance: CGFloat = 0.5) -> Bool {
        abs(lhs.x - rhs.x) <= tolerance && abs(lhs.y - rhs.y) <= tolerance
    }

    // MARK: - Animation

    private func startAnimation() {
        guard animationTimer == nil else { return }
        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) {
            [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    private func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
    }

    private func tick() {
        let tuning = animationTuning
        let attack = Float(tuning.smoothingAttack)
        let decay = Float(tuning.smoothingDecay)
        let factor = targetLevel > smoothLevel ? attack : decay
        smoothLevel += (targetLevel - smoothLevel) * factor
        barPhase += Float(tuning.phaseStep)

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        // Border glow (both modes)
        let level = CGFloat(smoothLevel)
        let borderWidth = tuning.glowWidthBase + Double(level) * tuning.glowWidthScale
        let borderAlpha = tuning.glowAlphaBase + Double(level) * tuning.glowAlphaScale
        contentView?.layer?.borderWidth = borderWidth
        contentView?.layer?.borderColor = glowColor.withAlphaComponent(borderAlpha).cgColor

        // Bar animation (minimal mode)
        if isMinimal {
            tickBars(tuning: tuning)
        }

        CATransaction.commit()
    }

    private func tickBars(tuning: MicPanelAnimationTuning) {
        let h = pillHeight
        for i in 0..<bars.count {
            let bar = bars[i]
            let levelContrib = CGFloat(smoothLevel) * barScale[i] * (h * 0.5) * tuning.levelBarScale
            let phase = barPhase + barPhaseOffset[i]
            let idle = CGFloat(sin(phase) * 0.5 + 0.5) * tuning.idleBarAmplitude
            let barH = max(3, levelContrib + idle)
            let y = (h - barH) / 2
            bar.frame = CGRect(x: bar.frame.origin.x, y: y, width: barWidth, height: barH)

            let brightness = 0.5 + CGFloat(smoothLevel) * 0.5
            bar.backgroundColor = barColor.withAlphaComponent(brightness).cgColor
        }
    }

    // MARK: - Layout

    private func layoutBars(in width: CGFloat) {
        let totalBarsWidth = CGFloat(barCount) * barWidth + CGFloat(barCount - 1) * barGap
        let startX = (width - totalBarsWidth) / 2
        for i in 0..<bars.count {
            let x = startX + CGFloat(i) * (barWidth + barGap)
            bars[i].frame = CGRect(x: x, y: pillHeight / 2 - 1, width: barWidth, height: 3)
        }
    }

    private func resizeToFit() {
        guard let panel, let textView, let contentView else { return }
        guard !isMinimal else { return }

        let textWidth = fullWidth - textPadding * 2
        let text = textView.string

        var neededHeight = pillHeight
        if !text.isEmpty {
            let storage = NSTextStorage(string: text, attributes: [
                .font: NSFont.systemFont(ofSize: 15, weight: .regular),
            ])
            let container = NSTextContainer(
                size: NSSize(width: textWidth, height: .greatestFiniteMagnitude)
            )
            let lm = NSLayoutManager()
            lm.addTextContainer(container)
            storage.addLayoutManager(lm)
            lm.ensureLayout(for: container)
            let textH = lm.usedRect(for: container).height

            let screenH = NSScreen.main?.visibleFrame.height ?? 800
            let maxHeight = screenH - 100
            neededHeight = min(max(textH + verticalPadding * 2, pillHeight), maxHeight)
        }

        var frame = panel.frame
        frame.size.height = neededHeight
        panel.setFrame(frame, display: true, animate: false)

        contentView.frame = NSRect(x: 0, y: 0, width: fullWidth, height: neededHeight)
        textView.frame = NSRect(
            x: textPadding,
            y: verticalPadding,
            width: textWidth,
            height: neededHeight - verticalPadding * 2
        )
    }

    private func applyAnimationStyle() {
        guard let contentView else { return }
        let tuning = animationTuning
        contentView.layer?.borderWidth = tuning.glowWidthBase
        contentView.layer?.borderColor = glowColor.withAlphaComponent(tuning.glowAlphaBase).cgColor
        if isMinimal {
            layoutBars(in: minimalWidth)
        }
    }

    // MARK: - Panel Creation

    private func createPanel() {
        let p = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: fullWidth, height: pillHeight),
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        p.isFloatingPanel = true
        p.becomesKeyOnlyIfNeeded = true
        p.level = .statusBar
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = true
        p.hidesOnDeactivate = false
        p.titleVisibility = .hidden
        p.titlebarAppearsTransparent = true
        p.isMovableByWindowBackground = true

        let cv = NSView(frame: NSRect(x: 0, y: 0, width: fullWidth, height: pillHeight))
        cv.wantsLayer = true
        cv.layer?.cornerRadius = cornerRadius
        cv.layer?.masksToBounds = true
        cv.layer?.backgroundColor = NSColor(white: 0.10, alpha: 0.92).cgColor
        let tuning = animationTuning
        cv.layer?.borderWidth = tuning.glowWidthBase
        cv.layer?.borderColor = glowColor.withAlphaComponent(tuning.glowAlphaBase).cgColor

        // Waveform bars (hidden in full mode)
        bars = []
        for _ in 0..<barCount {
            let bar = CALayer()
            bar.cornerRadius = barWidth / 2
            bar.backgroundColor = barColor.withAlphaComponent(0.5).cgColor
            bar.isHidden = true
            cv.layer?.addSublayer(bar)
            bars.append(bar)
        }

        // Text view (hidden in minimal mode)
        let textWidth = fullWidth - textPadding * 2
        let tv = NSTextView(frame: NSRect(
            x: textPadding, y: verticalPadding,
            width: textWidth, height: pillHeight - verticalPadding * 2
        ))
        tv.isEditable = false
        tv.isSelectable = false
        tv.drawsBackground = false
        tv.font = .systemFont(ofSize: 15, weight: .regular)
        tv.textColor = NSColor(white: 1, alpha: 0.92)
        tv.isVerticallyResizable = false
        tv.isHorizontallyResizable = false
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.lineFragmentPadding = 0
        cv.addSubview(tv)
        textView = tv

        p.contentView = cv
        p.delegate = self
        contentView = cv
        panel = p
        applyAnimationStyle()
    }
}
