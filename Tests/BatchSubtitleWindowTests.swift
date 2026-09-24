import Foundation
import Testing
@testable import NativeASR

@Suite("Batch subtitle windows")
struct BatchSubtitleWindowTests {
    private let rate = ASRAudio.sampleRate

    @Test func subtitlesTranscribeAndAlignTheSameHardCappedAudio() throws {
        let service = WindowService(text: "こんにちは世界。", language: "Japanese")
        let audio = [Float](repeating: 0, count: 121 * rate)
        var aligned: [(Int, String)] = []
        let result = try BatchTranscriptionPipeline.subtitle(
            using: service, audio: audio, transcript: nil, language: nil, temperature: 0,
            vad: nil, chunking: .automatic,
            alignItems: { window, text, _, _ in
                aligned.append((window.count, text))
                return (text, "Japanese", [
                    ForcedAlignItem(text: text, startTime: 0, endTime: Double(window.count) / Double(rate))
                ])
            }
        )
        let clips = try #require(result.debug?.chunks)
        #expect(clips.count > 4)
        #expect(service.transcribeLengths == aligned.map(\.0))
        #expect(aligned.map(\.1) == clips.map(\.transcript))
        #expect(aligned.allSatisfy { $0.0 <= 30 * rate && $0.1 == "こんにちは世界。" })
        #expect(clips.first?.start == 0 && clips.last?.end == 121)
        #expect(zip(clips, clips.dropFirst()).allSatisfy { abs($0.end - $1.start) < 0.0001 })
        #expect(result.transcript == String(repeating: "こんにちは世界。", count: clips.count))
        #expect(result.language == "Japanese")
        #expect(result.debug?.chunkingMode == "energy-pre-asr-cap")
        #expect(result.items.allSatisfy { $0.endTime > $0.startTime && $0.endTime <= 121 })
    }

    @Test func subtitleDefaultPrefersVADOnLongAudioButPlainTranscriptionDoesNot() throws {
        #expect(BatchTranscriptionPipeline.subtitleChunkingMode(.automatic, audioDuration: 3951, hasVAD: true) == .vad)
        #expect(BatchTranscriptionPipeline.subtitleChunkingMode(.automatic, audioDuration: 3951, hasVAD: false) == .energy)
        #expect(BatchTranscriptionPipeline.subtitleChunkingMode(.energy, audioDuration: 3951, hasVAD: true) == .energy)
        #expect(BatchTranscriptionPipeline.subtitleChunkingMode(.vad, audioDuration: 3951, hasVAD: false) == .energy)
        #expect(BatchChunkingMode.automatic.resolved(audioDuration: 3951, hasVAD: true) == .energy)

        let service = WindowService(text: "chunk", language: "English")
        let result = try BatchTranscriptionPipeline.transcribe(
            using: service, audio: [Float](repeating: 0, count: 121 * rate),
            language: nil, temperature: 0, vad: nil, chunking: .energy
        )
        #expect(service.transcribeLengths == [1840800, 95200]) // Original 120s energy policy.
        #expect(result.text == "chunk chunk")
    }

    @Test func strictEnergyCutNeverExceedsThirtySecondsOrLeavesSubsecondTail() {
        var audio = [Float](repeating: 0.2, count: rate * 30 + rate / 2)
        for index in (29 * rate) ..< (29 * rate + rate / 5) { audio[index] = 0 }
        let chunks = chunkAudioByEnergy(audio, sampleRate: rate,
                                        config: BatchTranscriptionDefaults.subtitleAlignmentConfig,
                                        strictMaxDuration: true)
        #expect(chunks.count == 2)
        #expect(chunks.allSatisfy { $0.audio.count <= 30 * rate && $0.duration >= 1 })
        #expect(chunks[0].audio + chunks[1].audio == audio)
        #expect(chunks[0].endTime == chunks[1].startTime)
        #expect(chunks[1].endTime == 30.5)
    }

    @Test func suppliedTranscriptStaysIntactAndDoesNotTriggerASRForEmptySplits() throws {
        let service = WindowService(text: "unexpected", language: "English")
        let text = "こんにちは世界。"
        var aligned: [(Int, String)] = []
        let result = try BatchTranscriptionPipeline.subtitle(
            using: service, audio: [Float](repeating: 0, count: 65 * rate),
            transcript: text, language: "Japanese", temperature: 0, vad: nil, chunking: .energy,
            alignItems: { window, part, _, _ in
                aligned.append((window.count, part))
                return (part, "Japanese", [])
            }
        )
        #expect(service.transcribeLengths.isEmpty)
        #expect(aligned.allSatisfy { $0.0 <= 30 * rate && !$0.1.isEmpty })
        #expect(aligned.map(\.1).joined() == text)
        #expect(result.debug?.chunks.map(\.transcript).joined() == text)
        #expect(result.transcript == text)
    }

    @Test func r2t2TerminatorIsNotGivenToTheAlignerButQwenLiteralPipeRemains() throws {
        for r2t2 in [true, false] {
            let service = WindowService(text: r2t2 ? "language Japaneseこんにちは|extra" : "こんにちは|extra",
                                        language: nil, hasR2T2BatchDelimiter: r2t2)
            var aligned: [String] = []
            let result = try BatchTranscriptionPipeline.subtitle(
                using: service, audio: [Float](repeating: 0, count: 3 * rate),
                transcript: nil, language: "Japanese", temperature: 0, vad: nil,
                alignItems: { _, text, _, _ in
                    aligned.append(text)
                    return (text, "Japanese", [ForcedAlignItem(text: text, startTime: 0, endTime: 1)])
                }
            )
            let expected = r2t2 ? "こんにちは" : "こんにちは|extra"
            #expect(result.transcript == expected)
            #expect(aligned == [expected])
            #expect(result.debug?.chunks.first?.transcript == expected)
            #expect(result.language == "Japanese")
        }
    }

    @Test func autoJapaneseAndSuppliedTextDoNotDefaultToEnglish() throws {
        let service = WindowService(text: "こんにちは。", language: nil)
        var alignedLanguages: [String?] = []
        let audio = [Float](repeating: 0, count: 31 * rate)
        let result = try BatchTranscriptionPipeline.subtitle(
            using: service, audio: audio, transcript: nil, language: nil,
            temperature: 0, vad: nil,
            alignItems: { _, text, language, _ in
                alignedLanguages.append(language)
                return (text, language ?? "English", [ForcedAlignItem(text: text, startTime: 0, endTime: 1)])
            }
        )
        #expect(result.language == "Japanese")
        #expect(alignedLanguages.allSatisfy { $0 == "Japanese" })

        alignedLanguages.removeAll()
        let supplied = try BatchTranscriptionPipeline.subtitle(
            using: service, audio: audio, transcript: "こんにちは。", language: nil,
            temperature: 0, vad: nil,
            alignItems: { _, text, language, _ in
                alignedLanguages.append(language)
                return (text, language ?? "English", [])
            }
        )
        #expect(supplied.language == "Japanese")
        #expect(alignedLanguages.allSatisfy { $0 == "Japanese" })
    }

    @Test func laterEnglishClipDoesNotRestyleJapaneseFile() throws {
        let service = WindowService(text: "hello", language: nil,
                                    detectedLanguages: ["Japanese", "English"])
        var alignedLanguages: [String?] = []
        let result = try BatchTranscriptionPipeline.subtitle(
            using: service, audio: [Float](repeating: 0, count: 31 * rate),
            transcript: nil, language: nil, temperature: 0, vad: nil,
            alignItems: { _, text, language, _ in
                alignedLanguages.append(language)
                return (text, language ?? "English", [ForcedAlignItem(text: text, startTime: 0, endTime: 1)])
            }
        )
        #expect(service.transcribeLengths.count == 2)
        #expect(alignedLanguages == ["Japanese", "English"])
        #expect(result.language == "Japanese")
    }

    @Test func emptyASRDoesNotAlignOrRetranscribe() throws {
        let service = WindowService(text: "|extra", language: "Japanese", hasR2T2BatchDelimiter: true)
        let result = try BatchTranscriptionPipeline.subtitle(
            using: service, audio: [Float](repeating: 0, count: 2 * rate), transcript: nil,
            language: nil, temperature: 0, vad: nil,
            alignItems: { _, _, _, _ in Issue.record("Empty ASR text must not reach the aligner"); return ("", "", []) }
        )
        #expect(service.transcribeLengths == [2 * rate])
        #expect(result.transcript.isEmpty && result.items.isEmpty)
    }

    @Test func zeroDurationSuffixIsFoldedIntoTimedCueWithoutLosingText() {
        let items = [
            ForcedAlignItem(text: "音が似てく", startTime: 1170, endTime: 1171.68),
            ForcedAlignItem(text: "る。", startTime: 1171.688, endTime: 1171.688),
        ]
        let cues = groupSubtitles(items, language: "ja")
        #expect(cues.count == 1)
        #expect(cues[0].text == "音が似てくる。")
        #expect(cues[0].start == 1170 && cues[0].end == 1171.68)
        #expect(formatSRT(cues).contains("音が似てくる。"))
        #expect(cues.allSatisfy { $0.end > $0.start })
    }

    @Test func untimedWordsAttachBeforeGroupingWithoutExtendingCueTimes() {
        let items = [
            ForcedAlignItem(text: "前", startTime: 0, endTime: 0),
            ForcedAlignItem(text: "半。", startTime: 0.4, endTime: 1.2),
            ForcedAlignItem(text: "done.", startTime: 2, endTime: 3),
            ForcedAlignItem(text: "Next。", startTime: 3.15, endTime: 3.15),
            ForcedAlignItem(text: "scene.", startTime: 3.15, endTime: 4),
        ]
        let cues = groupSubtitles(items, language: "ja")
        #expect(cues.map(\.text) == ["前半。", "done.", "Next。 scene."])
        #expect(cues.map(\.start) == [0.4, 2, 3.15])
        #expect(cues.map(\.end) == [1.2, 3, 4])
        #expect(cues.map(\.index) == [1, 2, 3])
        #expect(cues.allSatisfy { $0.end > $0.start })
        #expect(groupSubtitles([ForcedAlignItem(text: "untimed", startTime: 0, endTime: 0)], language: "ja").isEmpty)
    }

    @Test func sharedPositiveIntervalBecomesOnePhraseWithoutInventedWordTimes() {
        let pinned = [
            ForcedAlignItem(text: "長", startTime: 2, endTime: 4, alignText: "長"),
            ForcedAlignItem(text: "短い", startTime: 2, endTime: 4, alignText: "短い"),
        ]
        let collapsed = BatchTranscriptionPipeline.collapsePinnedItems(pinned, duration: 12)
        #expect(collapsed.count == 1)
        #expect(collapsed[0].text == "長短い")
        #expect(collapsed[0].startTime == 2 && collapsed[0].endTime == 4)

        let invalid = pinned.map { ForcedAlignItem(text: $0.text, startTime: 999, endTime: 999) }
        let unchanged = BatchTranscriptionPipeline.collapsePinnedItems(invalid, duration: 12)
        #expect(unchanged.count == 2)
        let bounded = BatchTranscriptionPipeline.boundedAlignmentItems(unchanged, duration: 12)
        #expect(groupSubtitles(bounded, language: "ja").isEmpty)
    }
}

private final class WindowService: BatchTranscriptionServing, @unchecked Sendable {
    let text: String
    let language: String?
    let hasR2T2BatchDelimiter: Bool
    let detectedLanguages: [String?]?
    var transcribeLengths: [Int] = []

    init(text: String, language: String?, hasR2T2BatchDelimiter: Bool = false,
         detectedLanguages: [String?]? = nil) {
        self.text = text
        self.language = language
        self.hasR2T2BatchDelimiter = hasR2T2BatchDelimiter
        self.detectedLanguages = detectedLanguages
    }

    func transcribeChunk(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult {
        let index = transcribeLengths.count
        transcribeLengths.append(audio.count)
        let detected = detectedLanguages.flatMap { index < $0.count ? $0[index] : nil } ?? self.language
        return TranscriptionResult(text: text, language: detected,
                                   audioDuration: Double(audio.count) / Double(ASRAudio.sampleRate), processingTime: 0)
    }

    func subtitleItems(audio: [Float], transcript: String?, language: String?, temperature: Float,
                       aligner: ForcedAligner) throws -> (transcript: String, language: String, items: [ForcedAlignItem]) {
        Issue.record("Use injected alignment closure")
        return ("", "", [])
    }
}
