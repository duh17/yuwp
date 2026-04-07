import Foundation
import Testing
@testable import Yuwp

@Suite("CGEventInjector diff algorithm")
struct CGEventInjectorTests {

    @Test func emptyToText() {
        let (bs, suffix) = CGEventInjector.diff(old: "", new: "hello")
        #expect(bs == 0)
        #expect(suffix == "hello")
    }

    @Test func appendOnly() {
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "hello world")
        #expect(bs == 0)
        #expect(suffix == " world")
    }

    @Test func fullReplacement() {
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "world")
        #expect(bs == 5)
        #expect(suffix == "world")
    }

    @Test func partialReplacement() {
        // "hello" → "help": common prefix "hel" (3), removes "lo" (2 backspaces), types "p"
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "help")
        #expect(bs == 2)
        #expect(suffix == "p")
    }

    @Test func identical() {
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "hello")
        #expect(bs == 0)
        #expect(suffix == "")
    }

    @Test func deleteAll() {
        let (bs, suffix) = CGEventInjector.diff(old: "hello", new: "")
        #expect(bs == 5)
        #expect(suffix == "")
    }

    @Test func bothEmpty() {
        let (bs, suffix) = CGEventInjector.diff(old: "", new: "")
        #expect(bs == 0)
        #expect(suffix == "")
    }

    @Test func streamingPartials() {
        // Simulates incremental streaming: "" → "Hel" → "Hell" → "Hello"
        var current = ""

        let (bs1, s1) = CGEventInjector.diff(old: current, new: "Hel")
        #expect(bs1 == 0)
        #expect(s1 == "Hel")
        current = "Hel"

        let (bs2, s2) = CGEventInjector.diff(old: current, new: "Hell")
        #expect(bs2 == 0)
        #expect(s2 == "l")
        current = "Hell"

        let (bs3, s3) = CGEventInjector.diff(old: current, new: "Hello")
        #expect(bs3 == 0)
        #expect(s3 == "o")
    }

    @Test func batchCorrectionTypo() {
        // "helo wrld" → "Hello world": case difference on first char kills common prefix
        let (bs, suffix) = CGEventInjector.diff(old: "helo wrld", new: "Hello world")
        #expect(bs == 9)
        #expect(suffix == "Hello world")
    }

    @Test func periodSnapCorrection() {
        // Streaming: "hello." → "hello world" (period removed, word added)
        let (bs, suffix) = CGEventInjector.diff(old: "hello.", new: "hello world")
        #expect(bs == 1)   // delete "."
        #expect(suffix == " world")
    }

    @Test func unicodeAppend() {
        let (bs, suffix) = CGEventInjector.diff(old: "你好", new: "你好世界")
        #expect(bs == 0)
        #expect(suffix == "世界")
    }

    @Test func unicodeFullReplacement() {
        // No common prefix between Chinese and ASCII
        let (bs, suffix) = CGEventInjector.diff(old: "你好世界", new: "Hello")
        #expect(bs == 4)
        #expect(suffix == "Hello")
    }

    @Test func singleCharAppend() {
        let (bs, suffix) = CGEventInjector.diff(old: "a", new: "ab")
        #expect(bs == 0)
        #expect(suffix == "b")
    }

    @Test func singleCharReplace() {
        let (bs, suffix) = CGEventInjector.diff(old: "a", new: "b")
        #expect(bs == 1)
        #expect(suffix == "b")
    }

    @Test func commitAfterStreamDiffers() {
        // Full session: stream "helo", then commit "Hello"
        var state = ""

        let (bs1, s1) = CGEventInjector.diff(old: state, new: "helo")
        state = "helo"
        #expect(bs1 == 0)
        #expect(s1 == "helo")

        // Commit with batch-corrected text
        let (bs2, s2) = CGEventInjector.diff(old: state, new: "Hello")
        #expect(bs2 == 4)   // backspace all of "helo" (no common prefix — 'h' != 'H')
        #expect(s2 == "Hello")
    }

    @Test func commitAfterStreamIdentical() {
        var state = ""

        let (bs1, s1) = CGEventInjector.diff(old: state, new: "hello")
        state = "hello"
        #expect(bs1 == 0)
        #expect(s1 == "hello")

        // Batch correction returns same text
        let (bs2, s2) = CGEventInjector.diff(old: state, new: "hello")
        #expect(bs2 == 0)
        #expect(s2 == "")
    }
}
