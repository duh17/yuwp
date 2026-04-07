import Foundation
import Testing
@testable import Yuwp

@Suite("TypewriterAnimator")
@MainActor
struct TypewriterAnimatorTests {

    // MARK: - Basic Operation

    @Test func initialState() {
        let tw = TypewriterAnimator()
        #expect(tw.displayText == "")
        #expect(!tw.isAnimating)
    }

    @Test func updateStartsAnimation() {
        let tw = TypewriterAnimator()
        tw.update(fullText: "Hello")
        #expect(tw.isAnimating)
    }

    @Test func commitSnapsToTarget() {
        let tw = TypewriterAnimator()
        tw.update(fullText: "Hello world")
        tw.commitCurrentAnimation()
        #expect(tw.displayText == "Hello world")
        #expect(!tw.isAnimating)
    }

    @Test func resetClearsEverything() {
        let tw = TypewriterAnimator()
        tw.update(fullText: "Hello")
        tw.reset()
        #expect(tw.displayText == "")
        #expect(!tw.isAnimating)
    }

    // MARK: - Append Behavior (streaming partials)

    @Test func appendOnlyAnimatesNewChars() {
        let tw = TypewriterAnimator()

        // First update: "Hello" — animates from empty
        tw.update(fullText: "Hello")
        tw.commitCurrentAnimation()
        #expect(tw.displayText == "Hello")

        // Second update appends: "Hello world" — only " world" should animate
        tw.update(fullText: "Hello world")
        // displayText should show common prefix immediately
        #expect(tw.displayText.hasPrefix("Hello"))
        // But shouldn't show the full text yet (animating)
        #expect(tw.isAnimating)

        tw.commitCurrentAnimation()
        #expect(tw.displayText == "Hello world")
    }

    // MARK: - Correction Behavior (batch retranscription)

    @Test func correctionSnapsInsteadOfReAnimating() {
        let tw = TypewriterAnimator()

        // Streaming partial: "hello im testing this"
        tw.update(fullText: "hello im testing this")
        tw.commitCurrentAnimation()
        #expect(tw.displayText == "hello im testing this")

        // Batch correction: "Hello, I'm testing this."
        // The common prefix is just "Hello" (case change breaks at char 5)
        // This should snap the correction, not re-animate the whole thing
        tw.update(fullText: "Hello, I'm testing this.")

        // The corrected text up to the old length should snap immediately
        let display = tw.displayText
        #expect(display.count >= "hello im testing this".count,
                "Correction should snap to at least old text length, got: \(display)")
    }

    @Test func correctionWithShorterTextSnaps() {
        let tw = TypewriterAnimator()

        tw.update(fullText: "Hello world testing")
        tw.commitCurrentAnimation()

        // Batch correction removes words
        tw.update(fullText: "Hello world.")
        #expect(tw.displayText == "Hello world.")
        #expect(!tw.isAnimating)
    }

    @Test func correctionWithSameLengthSnaps() {
        let tw = TypewriterAnimator()

        tw.update(fullText: "hello world")
        tw.commitCurrentAnimation()

        // Capitalization correction, same length
        tw.update(fullText: "Hello World")
        #expect(tw.displayText == "Hello World")
    }

    @Test func correctionWithNewAppendedText() {
        let tw = TypewriterAnimator()

        tw.update(fullText: "hello testing")
        tw.commitCurrentAnimation()

        // Batch correction + new text: corrects "hello" to "Hello," and adds more
        tw.update(fullText: "Hello, testing this now.")
        tw.commitCurrentAnimation()
        #expect(tw.displayText == "Hello, testing this now.")
    }

    // MARK: - Edge Cases

    @Test func emptyUpdateSnaps() {
        let tw = TypewriterAnimator()
        tw.update(fullText: "Hello")
        tw.commitCurrentAnimation()
        tw.update(fullText: "")
        #expect(tw.displayText == "")
    }

    @Test func identicalUpdateIsNoOp() {
        let tw = TypewriterAnimator()
        tw.update(fullText: "Hello")
        tw.commitCurrentAnimation()
        tw.update(fullText: "Hello")
        #expect(tw.displayText == "Hello")
        #expect(!tw.isAnimating)
    }

    @Test func rapidUpdatesCommitPrevious() {
        let tw = TypewriterAnimator()
        tw.update(fullText: "First")
        tw.update(fullText: "First and second")
        // First update should be committed, second should be animating
        #expect(tw.displayText.hasPrefix("First"))
        tw.commitCurrentAnimation()
        #expect(tw.displayText == "First and second")
    }

    @Test func commitAfterUpdateSnapsToFull() {
        let tw = TypewriterAnimator()
        tw.update(fullText: "Hi there")
        #expect(tw.isAnimating)
        // Commit should snap all remaining chars
        tw.commitCurrentAnimation()
        #expect(tw.displayText == "Hi there")
        #expect(!tw.isAnimating)
    }
}
