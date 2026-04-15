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
    private let fullWidth: CGFloat = 360
    private let minimalWidth: CGFloat = 80
    private let pillHeight: CGFloat = 32
    private let cornerRadius: CGFloat = 16 // fully rounded ends
    private let textPadding: CGFloat = 18
    private let textTopPadding: CGFloat = 9
    private let textBottomPadding: CGFloat = 12
    private let textContainerVerticalInset: CGFloat = 3
    private let screenEdgePadding: CGFloat = 8
    private let bottomPadding: CGFloat = 36
    private var lastProgrammaticOrigin: NSPoint?

    // Bar waveform
    private let barCount = 5
    private let barWidth: CGFloat = 3.0
    private let barGap: CGFloat = 3.0
    private let barScale: [CGFloat] = [0.5, 0.8, 1.0, 0.8, 0.5]
    private let barPhaseOffset: [Float] = [0, 0.7, 1.4, 2.1, 2.8]
    private let barColor = NSColor(calibratedRed: 0.35, green: 0.72, blue: 1.0, alpha: 1.0)

    // Glow
    private let glowColor = NSColor(calibratedRed: 0.36, green: 0.74, blue: 1.0, alpha: 1.0)

    // Transcript rendering state
    private var currentDisplayText = ""
    private var currentCommittedPrefixLength = 0
    private var correctionHighlightRanges: [NSRange] = []
    private var correctionHighlightText = ""
    private var correctionHighlightToken = UUID()

    private var animationTuning: MicPanelAnimationTuning {
        animationConfig.resolvedTuning(reduceMotion: NSWorkspace.shared.accessibilityDisplayShouldReduceMotion)
    }

    private var transcriptParagraphStyle: NSMutableParagraphStyle {
        let paragraph = NSMutableParagraphStyle()
        paragraph.alignment = .left
        paragraph.lineBreakMode = .byWordWrapping
        paragraph.lineSpacing = 2
        paragraph.hyphenationFactor = 0
        return paragraph
    }

    private var transcriptBaseAttributes: [NSAttributedString.Key: Any] {
        [
            .font: NSFont.systemFont(ofSize: 15, weight: .regular),
            .foregroundColor: NSColor(white: 1, alpha: 0.90),
            .paragraphStyle: transcriptParagraphStyle,
        ]
    }

    private var transcriptCommittedAttributes: [NSAttributedString.Key: Any] {
        [
            .font: NSFont.systemFont(ofSize: 15, weight: .regular),
            .foregroundColor: NSColor(white: 1, alpha: 0.91),
        ]
    }

    private var transcriptActiveAttributes: [NSAttributedString.Key: Any] {
        [
            .font: NSFont.systemFont(ofSize: 15, weight: .semibold),
            .foregroundColor: NSColor(calibratedRed: 0.66, green: 0.84, blue: 1.0, alpha: 1.0),
            .backgroundColor: NSColor(calibratedRed: 0.25, green: 0.45, blue: 0.78, alpha: 0.14),
        ]
    }

    private var correctionUnderlineAttributes: [NSAttributedString.Key: Any] {
        [
            .underlineStyle: NSUnderlineStyle.single.rawValue,
            .underlineColor: NSColor(calibratedRed: 1.0, green: 0.45, blue: 0.45, alpha: 0.95),
        ]
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
            resetTranscriptRenderState()
            renderCurrentTranscript()
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
            resetTranscriptRenderState()
            renderCurrentTranscript()
        } else {
            updateTranscript(state)
        }
    }

    private func updateTranscript(_ state: DictationPresentationState) {
        guard !isMinimal else { return }

        let newText = state.displayText
        let previousCommittedPrefixLength = currentCommittedPrefixLength
        let newCommittedPrefixLength = committedVisiblePrefixLength(
            displayText: newText,
            committedText: state.committedText,
            activeText: state.activeText
        )
        currentCommittedPrefixLength = newCommittedPrefixLength

        // Show correction underline only when text has settled from active (blue)
        // into committed (white), not during in-flight active corrections.
        let settledPrefixAdvanced = newCommittedPrefixLength > previousCommittedPrefixLength
        let correctionRanges = correctionWordRanges(old: currentDisplayText, new: newText)
        if settledPrefixAdvanced, !correctionRanges.isEmpty {
            correctionHighlightText = newText
            correctionHighlightRanges = correctionRanges
            let token = UUID()
            correctionHighlightToken = token
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.9) { [weak self] in
                Task { @MainActor in
                    guard let self, self.correctionHighlightToken == token else { return }
                    self.correctionHighlightRanges = []
                    self.correctionHighlightText = ""
                    self.renderCurrentTranscript()
                }
            }
        } else if correctionHighlightText != newText {
            correctionHighlightRanges = []
            correctionHighlightText = ""
        }

        currentDisplayText = newText
        renderCurrentTranscript()
        resizeToFit()
    }

    private func resetTranscriptRenderState() {
        currentDisplayText = ""
        currentCommittedPrefixLength = 0
        correctionHighlightRanges = []
        correctionHighlightText = ""
        correctionHighlightToken = UUID()
    }

    private func renderCurrentTranscript() {
        textView?.textStorage?.setAttributedString(makeRenderedTranscriptAttributedString())
    }

    private func makeRenderedTranscriptAttributedString() -> NSAttributedString {
        let text = currentDisplayText
        let attributed = NSMutableAttributedString(string: text, attributes: transcriptBaseAttributes)

        let committedPrefix = min(currentCommittedPrefixLength, text.count)
        let committedEnd = text.index(text.startIndex, offsetBy: committedPrefix)

        if committedEnd > text.startIndex {
            let committedRange = NSRange(text.startIndex..<committedEnd, in: text)
            attributed.addAttributes(transcriptCommittedAttributes, range: committedRange)
        }

        var activeStart = committedEnd
        if activeStart < text.endIndex, text[activeStart] == " " {
            activeStart = text.index(after: activeStart)
        }
        if activeStart < text.endIndex {
            let activeRange = NSRange(activeStart..<text.endIndex, in: text)
            attributed.addAttributes(transcriptActiveAttributes, range: activeRange)
        }

        if correctionHighlightText == text {
            let committedBounds = NSRange(
                location: 0,
                length: max(0, min(currentCommittedPrefixLength, (text as NSString).length))
            )
            for range in correctionHighlightRanges {
                let visibleRange = NSIntersectionRange(range, committedBounds)
                if visibleRange.length > 0 {
                    attributed.addAttributes(correctionUnderlineAttributes, range: visibleRange)
                }
            }
        }

        return attributed
    }

    private func committedVisiblePrefixLength(displayText: String, committedText: String, activeText: String) -> Int {
        guard !displayText.isEmpty, !committedText.isEmpty else { return 0 }
        let shared = commonPrefixCount(displayText, committedText)

        // If active text exists and we've fully shown committed text, skip the separator
        // space so active styling starts on the first active character.
        if !activeText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            shared < displayText.count {
            let idx = displayText.index(displayText.startIndex, offsetBy: shared)
            if displayText[idx] == " " {
                return min(displayText.count, shared + 1)
            }
        }
        return min(displayText.count, shared)
    }

    private struct WordToken {
        let text: String
        let range: NSRange
    }

    private func correctionWordRanges(old: String, new: String) -> [NSRange] {
        guard !old.isEmpty, !new.isEmpty else { return [] }

        let oldTokens = tokenizeWords(in: old)
        let newTokens = tokenizeWords(in: new)
        guard !oldTokens.isEmpty, !newTokens.isEmpty else { return [] }

        var prefix = 0
        while prefix < oldTokens.count,
            prefix < newTokens.count,
            oldTokens[prefix].text == newTokens[prefix].text {
            prefix += 1
        }

        // Pure append should not flash correction underline.
        if prefix == oldTokens.count, newTokens.count >= oldTokens.count {
            return []
        }

        var suffix = 0
        while oldTokens.count - suffix - 1 >= prefix,
            newTokens.count - suffix - 1 >= prefix,
            oldTokens[oldTokens.count - suffix - 1].text == newTokens[newTokens.count - suffix - 1].text {
            suffix += 1
        }

        let start = prefix
        let end = newTokens.count - suffix
        guard end > start else { return [] }

        return newTokens[start..<end].map(\.range)
    }

    private func tokenizeWords(in text: String) -> [WordToken] {
        let nsText = text as NSString
        let full = NSRange(location: 0, length: nsText.length)
        var tokens: [WordToken] = []

        text.enumerateSubstrings(in: text.startIndex..<text.endIndex, options: [.byWords, .substringNotRequired]) {
            _, substringRange, _, _ in
            let nsRange = NSRange(substringRange, in: text)
            let token = nsText.substring(with: nsRange)
            tokens.append(WordToken(text: token, range: nsRange))
        }

        // Fallback for scripts where .byWords returns nothing.
        if tokens.isEmpty {
            nsText.enumerateSubstrings(in: full, options: [.byComposedCharacterSequences]) {
                substring, range, _, _ in
                guard let substring else { return }
                if substring.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return }
                tokens.append(WordToken(text: substring, range: range))
            }
        }

        return tokens
    }

    private func commonPrefixCount(_ lhs: String, _ rhs: String) -> Int {
        var count = 0
        var li = lhs.startIndex
        var ri = rhs.startIndex
        while li < lhs.endIndex, ri < rhs.endIndex, lhs[li] == rhs[ri] {
            count += 1
            li = lhs.index(after: li)
            ri = rhs.index(after: ri)
        }
        return count
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

        // Border glow (both modes) — slightly stronger in transcript mode so
        // the bubble feels alive without becoming visually noisy.
        let level = CGFloat(smoothLevel)
        let widthBoost = isMinimal ? 1.20 : 1.45
        let alphaBoost = isMinimal ? 1.22 : 1.55
        let borderWidth = (tuning.glowWidthBase + Double(level) * tuning.glowWidthScale) * widthBoost
        let borderAlpha = min(0.85, (tuning.glowAlphaBase + Double(level) * tuning.glowAlphaScale) * alphaBoost)
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

            let brightness = 0.58 + CGFloat(smoothLevel) * 0.42
            bar.backgroundColor = barColor.withAlphaComponent(min(1.0, brightness)).cgColor
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
        let attributedText = makeRenderedTranscriptAttributedString()

        var neededHeight = pillHeight
        if attributedText.length > 0 {
            let storage = NSTextStorage(attributedString: attributedText)
            let container = NSTextContainer(
                size: NSSize(width: textWidth, height: .greatestFiniteMagnitude)
            )
            let lm = NSLayoutManager()
            lm.addTextContainer(container)
            storage.addLayoutManager(lm)
            lm.ensureLayout(for: container)
            let textH = lm.usedRect(for: container).height

            let contentPadding = textTopPadding + textBottomPadding + (textContainerVerticalInset * 2)
            let screenH = NSScreen.main?.visibleFrame.height ?? 800
            let maxHeight = screenH - 100
            neededHeight = min(max(textH + contentPadding, pillHeight), maxHeight)
        }

        var frame = panel.frame
        frame.size.height = neededHeight
        panel.setFrame(frame, display: true, animate: false)

        contentView.frame = NSRect(x: 0, y: 0, width: fullWidth, height: neededHeight)
        textView.frame = NSRect(
            x: textPadding,
            y: textBottomPadding,
            width: textWidth,
            height: neededHeight - textTopPadding - textBottomPadding
        )
    }

    private func applyAnimationStyle() {
        guard let contentView else { return }
        let tuning = animationTuning
        let widthBoost = isMinimal ? 1.20 : 1.45
        let alphaBoost = isMinimal ? 1.22 : 1.55
        contentView.layer?.borderWidth = tuning.glowWidthBase * widthBoost
        contentView.layer?.borderColor = glowColor.withAlphaComponent(min(0.85, tuning.glowAlphaBase * alphaBoost)).cgColor
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
        cv.layer?.backgroundColor = NSColor(white: 0.11, alpha: 0.95).cgColor
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
            x: textPadding, y: textBottomPadding,
            width: textWidth, height: pillHeight - textTopPadding - textBottomPadding
        ))
        tv.isEditable = false
        tv.isSelectable = false
        tv.drawsBackground = false
        tv.isVerticallyResizable = false
        tv.isHorizontallyResizable = false
        tv.alignment = .left
        tv.textContainerInset = NSSize(width: 0, height: textContainerVerticalInset)
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.lineFragmentPadding = 0
        tv.textContainer?.lineBreakMode = .byWordWrapping
        tv.textStorage?.setAttributedString(NSAttributedString(string: "", attributes: transcriptBaseAttributes))
        cv.addSubview(tv)
        textView = tv

        p.contentView = cv
        p.delegate = self
        contentView = cv
        panel = p
        applyAnimationStyle()
    }
}
