import Foundation

/// Animates text appearing character-by-character when the sidecar sends
/// full-replacement transcript updates every ~2s.
///
/// Computes the delta (new chars appended) and reveals them gradually
/// over ~1.5s, leaving a 0.5s buffer before the next update.
/// If a new update arrives mid-animation, snaps to completion and starts fresh.
@MainActor
final class TypewriterAnimator {
    private(set) var displayText = ""
    private(set) var isAnimating = false

    private var targetText = ""
    private var animationTask: Task<Void, Never>?

    /// Total reveal time for delta chars. Lower = faster typewriter.
    static let animationDurationNs: UInt64 = 800_000_000  // 0.8s (was 1.5s)
    static let minimumIntervalNs: UInt64 = 8_000_000 // ~8ms, one frame at 120Hz

    /// Feed a new full replacement transcript. Animates new text only.
    ///
    /// Corrections (batch retranscription, punctuation changes) snap instantly.
    /// Only genuinely new characters appended to the end get animated.
    func update(fullText: String) {
        commitCurrentAnimation()

        let previousTarget = targetText
        targetText = fullText

        let commonCount = commonPrefixCount(previousTarget, fullText)

        // If the common prefix doesn't cover the old text, something changed
        // mid-text (a correction). Snap the corrected portion immediately
        // and only animate chars beyond the old target length.
        //
        // Exception: if the only removed characters are trailing punctuation
        // (.!?。！？) and there's new text after, treat it as a pure append.
        // The ASR commonly outputs "word." then corrects to "word more text."
        // — the period was speculative, not a real correction.
        let removedCount = previousTarget.count - commonCount
        let isTrailingPunctOnly = removedCount > 0
            && fullText.count > previousTarget.count
            && Self.isOnlyTrailingPunctuation(
                previousTarget, from: commonCount
            )
        let isCorrection = removedCount > 0 && !isTrailingPunctOnly

        if isCorrection {
            // Snap to the end of the old text (corrected), animate only truly new chars
            let snapTo = min(previousTarget.count, fullText.count)
            let snapEnd = fullText.index(fullText.startIndex, offsetBy: snapTo)
            displayText = String(fullText[..<snapEnd])

            // If no new chars beyond old length, we're done
            guard fullText.count > previousTarget.count else { return }
        } else {
            // Pure append (or period-merge) — show common prefix, animate the delta
            let prefixEnd = fullText.index(fullText.startIndex, offsetBy: commonCount)
            displayText = String(fullText[..<prefixEnd])
        }

        // Nothing new to animate?
        guard fullText.count > displayText.count else {
            displayText = fullText
            return
        }

        let animateFrom = fullText.index(fullText.startIndex, offsetBy: displayText.count)
        let deltaCount = fullText.count - displayText.count
        let intervalNs = max(
            Self.minimumIntervalNs,
            Self.animationDurationNs / UInt64(max(1, deltaCount))
        )

        isAnimating = true

        animationTask = Task { [weak self] in
            var currentIndex = animateFrom
            let target = fullText

            while currentIndex < target.endIndex {
                do {
                    try await Task.sleep(nanoseconds: intervalNs)
                } catch { break }
                guard !Task.isCancelled else { break }
                guard let self else { break }

                currentIndex = target.index(after: currentIndex)
                self.displayText = String(target[..<currentIndex])
            }

            guard let self else { return }
            if !Task.isCancelled {
                self.isAnimating = false
            }
        }
    }

    /// Snap to target text immediately.
    func commitCurrentAnimation() {
        animationTask?.cancel()
        animationTask = nil
        displayText = targetText
        isAnimating = false
    }

    func reset() {
        animationTask?.cancel()
        animationTask = nil
        targetText = ""
        displayText = ""
        isAnimating = false
    }

    private static let trailingPunctuation: Set<Character> = [".", "!", "?", "\u{3002}", "\u{FF01}", "\u{FF1F}"]

    /// Check if all characters from `fromIndex` to the end are trailing punctuation.
    private static func isOnlyTrailingPunctuation(_ text: String, from offset: Int) -> Bool {
        var idx = text.index(text.startIndex, offsetBy: offset)
        while idx < text.endIndex {
            let ch = text[idx]
            if !ch.isWhitespace && !trailingPunctuation.contains(ch) {
                return false
            }
            idx = text.index(after: idx)
        }
        return true
    }

    private func commonPrefixCount(_ a: String, _ b: String) -> Int {
        var count = 0
        var ai = a.startIndex
        var bi = b.startIndex
        while ai < a.endIndex, bi < b.endIndex, a[ai] == b[bi] {
            count += 1
            ai = a.index(after: ai)
            bi = b.index(after: bi)
        }
        return count
    }
}
