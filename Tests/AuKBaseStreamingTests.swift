import Foundation
import Testing
@testable import NativeTTS

@Suite("AuK Base sampling, detection, long-form TTS chunking, stream events")
struct AuKBaseStreamingTests {
    @Test func baseSamplingDefaultsMatchUpstreamMLX() {
        let resolved = AuKSampling.resolve(variant: .base)
        #expect(resolved.nfe == 32)
        #expect(resolved.cfgStrength == 2.0)
        #expect(resolved.sway == -1.0)
        #expect(resolved.usesCFG)
        #expect(resolved.tGrid.count == 33)
        #expect(abs(resolved.tGrid.first! ) < 1e-7)
        #expect(abs(resolved.tGrid.last! - 1) < 1e-6)
        #expect(abs(resolved.tGrid[1] - 0.0012045438) < 1e-6)
        #expect(abs(resolved.tGrid[8] - 0.0761204675) < 1e-6)
        #expect(abs(resolved.tGrid[16] - 0.2928932188) < 1e-6)
        #expect(abs(resolved.tGrid[24] - 0.6173165676) < 1e-6)
        for (prev, next) in zip(resolved.tGrid, resolved.tGrid.dropFirst()) {
            #expect(next > prev)
        }
    }

    @Test func baseSwayZeroKeepsLinspaceGrid() {
        let resolved = AuKSampling.resolve(variant: .base, nfe: 4, cfgStrength: 0, sway: 0)
        #expect(resolved.nfe == 4)
        #expect(resolved.cfgStrength == 0)
        #expect(resolved.usesCFG == false)
        #expect(resolved.tGrid.count == 5)
        #expect(abs(resolved.tGrid[1] - 0.25) < 1e-6)
        #expect(abs(resolved.tGrid[2] - 0.5) < 1e-6)
    }

    @Test func baseExplicitTGridSkipsSwayWarp() {
        let resolved = AuKSampling.resolve(
            variant: .base,
            nfe: 8,
            cfgStrength: 1.5,
            sway: -1,
            tGrid: [0, 1]
        )
        #expect(resolved.nfe == 1)
        #expect(resolved.cfgStrength == 1.5)
        #expect(resolved.tGrid == [0, 1])
    }

    @Test func flashStillIgnoresBaseCallerSampling() {
        let resolved = AuKSampling.resolve(variant: .flash, nfe: 32, cfgStrength: 2, sway: -1)
        #expect(resolved == AuKSampling.flash)
        #expect(resolved.nfe == 4)
        #expect(resolved.cfgStrength == 0)
        #expect(resolved.sway == nil)
        #expect(resolved.tGrid == AuKFlashConfig.tGrid)
    }

    @Test func directoryDetectionDistinguishesBaseFlashAndQwen() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("auk-base-detect-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try "model:\n  name: AuK\n".write(to: root.appendingPathComponent("config.yaml"), atomically: true, encoding: .utf8)
        #expect(inspectAuKModelDirectory(root) == .pytorchSource(variant: .base))
        #expect(detectTTSBackend(at: root) == .aukBase)

        try Data().write(to: root.appendingPathComponent("auk_base.safetensors"))
        #expect(inspectAuKModelDirectory(root) == .pytorchSource(variant: .base))
        #expect(detectTTSBackend(at: root) != .qwen3TTS)

        try Data().write(to: root.appendingPathComponent("dit_base.safetensors"))
        try Data().write(to: root.appendingPathComponent("fusion_base.safetensors"))
        try Data().write(to: root.appendingPathComponent("vae.safetensors"))
        #expect(inspectAuKModelDirectory(root) == .converted(variant: .base))
        #expect(detectTTSBackend(at: root) == .aukBase)
        #expect(AuKVariant.base.backendString == "auk-base")
        #expect(AuKVariant.base.modelName == "AuK")
    }

    @Test func aukBasePytorchFileIsNotClassifiedAsQwen() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("auk-base-file-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data().write(to: root.appendingPathComponent("auk_base.safetensors"))
        #expect(inspectAuKModelDirectory(root) == .pytorchSource(variant: .base))
        #expect(detectTTSBackend(at: root) == .aukBase)
    }

    @Test func qwenDirectoryStillDetectsAsQwen() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("qwen-not-auk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try #"{"tts_model_type":"custom_voice"}"#.write(
            to: root.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )
        #expect(inspectAuKModelDirectory(root) == .unknown)
        #expect(detectTTSBackend(at: root) == .qwen3TTS)
        #expect(detectTTSBackend(at: root).isAuK == false)
    }

    @Test func longFormTTSPreservesSentenceOrderAndInsertsPause() {
        let text = [
            "This is the first sentence and it is deliberately a little longer than usual.",
            "Here is the second sentence, which should stay attached to its own natural boundary.",
            "This third sentence pushes the total over the target chunk size so a split should happen here.",
            "Finally, the fourth sentence should land in a later chunk instead of being forced into the first one.",
        ].joined(separator: " ")
        let result = AuKSpeechPlanner.plan(
            AuKSpeechPlanRequest(
                task: "tts",
                input: text,
                chunkTargetCharacters: 140,
                chunkHardCharacterLimit: 180,
                genSeconds: 10
            )
        )
        guard case .plan(let plan) = result else {
            Issue.record("expected TTS plan")
            return
        }
        #expect(plan.kind == .tts)
        #expect(plan.isMultiChunk)
        #expect(plan.pauseSeconds == 0.2)
        #expect(plan.chunks.allSatisfy { $0.instruction.contains("This") || $0.instruction.contains("Here") || $0.instruction.contains("Finally") || $0.instruction.contains("third") })
        let joined = plan.chunks.map(\.instruction).joined(separator: " ")
        #expect(joined == text)
        #expect(plan.chunks.allSatisfy { $0.genSeconds != nil })
        let seconds = plan.chunks.compactMap(\.genSeconds)
        #expect(abs(seconds.reduce(0, +) - 10) < 0.3)

        let audio = aukConcatenateTTSChunks(
            [[Float](repeating: 1, count: 10), [Float](repeating: 2, count: 10)],
            sampleRate: 24_000,
            pauseSeconds: 0.2
        )
        #expect(audio.count == 20 + Int((0.2 * 24_000).rounded()))
        #expect(audio[10] == 0)
        #expect(audio.last == 2)
    }

    @Test func zeroShotTemplateChunksOnlyTargetText() {
        let spoken = [
            "Ladies and gentlemen, it is an honor to have the opportunity to address such a distinguished audience today.",
            "We should remember that clarity matters more than volume when the room is already listening.",
            "Thank you for your time, and please enjoy the rest of the program this evening with our remaining speakers.",
        ].joined(separator: " ")
        let instruction = "Say the following with the same voice: '\(spoken)'"
        let result = AuKSpeechPlanner.plan(
            AuKSpeechPlanRequest(
                instruction: instruction,
                hasSourceAudio: true,
                chunkTargetCharacters: 120,
                chunkHardCharacterLimit: 160
            )
        )
        guard case .plan(let plan) = result else {
            Issue.record("expected TTS plan from zero-shot template")
            return
        }
        #expect(plan.isMultiChunk)
        #expect(plan.chunks.allSatisfy { $0.instruction.hasPrefix("Say the following with the same voice: '") })
        #expect(plan.chunks.allSatisfy { $0.instruction.hasSuffix("'") })
        let extracted = plan.chunks.map { chunk in
            String(chunk.instruction.dropFirst("Say the following with the same voice: '".count).dropLast())
        }
        #expect(extracted.joined(separator: " ") == spoken)
    }

    @Test func instructTemplateKeepsVoiceDescriptionAndChunksContent() {
        let content = [
            "Welcome home, how was work today my love I missed you so much this afternoon.",
            "Dinner is almost ready and the tea is already waiting on the table by the window.",
        ].joined(separator: " ")
        let instruction = "Based on the following description: \"a warm female voice\", generate speech content \"\(content)\"."
        let result = AuKSpeechPlanner.plan(
            AuKSpeechPlanRequest(
                instruction: instruction,
                chunkTargetCharacters: 80,
                chunkHardCharacterLimit: 120
            )
        )
        guard case .plan(let plan) = result else {
            Issue.record("expected TTS plan from instruct template")
            return
        }
        #expect(plan.isMultiChunk)
        #expect(plan.chunks.allSatisfy { $0.instruction.contains("a warm female voice") })
        #expect(plan.chunks.allSatisfy { $0.instruction.contains("generate speech content") })
    }

    @Test func openAIInputPlusStyleIsTTSEvenWithoutTask() {
        let input = [
            "The resident server correctness harness is checking cloned voice quality with a longer script.",
            "Audio should begin quickly, remain intelligible, and keep a stable speaker identity while generation continues.",
        ].joined(separator: " ")
        let result = AuKSpeechPlanner.plan(
            AuKSpeechPlanRequest(
                input: input,
                instructions: "a calm male narrator",
                hasSourceAudio: true,
                chunkTargetCharacters: 90,
                chunkHardCharacterLimit: 130
            )
        )
        guard case .plan(let plan) = result else {
            Issue.record("expected TTS plan from input+style")
            return
        }
        #expect(plan.kind == .tts)
        #expect(plan.isMultiChunk)
        #expect(plan.chunks.allSatisfy { $0.instruction.contains("Say the following with the same voice:") })
    }

    @Test func editInstructionWithSourceAudioStaysSingleTask() {
        let result = AuKSpeechPlanner.plan(
            AuKSpeechPlanRequest(
                instruction: "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.",
                hasSourceAudio: true,
                genSeconds: 7
            )
        )
        guard case .plan(let plan) = result else {
            Issue.record("expected single-task edit plan")
            return
        }
        #expect(plan.kind == .singleTask)
        #expect(plan.chunks.count == 1)
        #expect(plan.chunks[0].instruction.contains("Replace"))
    }

    @Test func requestingAutoChunkOnEditFailsClosedWithStructuredOptions() {
        let result = AuKSpeechPlanner.plan(
            AuKSpeechPlanRequest(
                instruction: "Increase the volume by 10 dB.",
                hasSourceAudio: true,
                autoChunk: true
            )
        )
        guard case .rejected(let rejection) = result else {
            Issue.record("expected auto-chunk rejection")
            return
        }
        #expect(rejection.code == "auto_chunk_unsupported")
        #expect(rejection.options["task"] == "tts")
        #expect(rejection.jsonObject["code"] as? String == "auto_chunk_unsupported")
    }

    @Test func enhancementAndSeparationAutoChunkAlsoFailClosed() {
        for instruction in [
            "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.",
            "Keep only the first speaker to start talking and remove all other speakers.",
            "Raise the pitch by 2 semitones.",
        ] {
            let result = AuKSpeechPlanner.plan(
                AuKSpeechPlanRequest(
                    instruction: instruction,
                    hasSourceAudio: true,
                    chunkTargetCharacters: 80
                )
            )
            guard case .rejected(let rejection) = result else {
                Issue.record("expected rejection for \(instruction)")
                continue
            }
            #expect(rejection.code == "auto_chunk_unsupported")
        }
    }

    @Test func autoChunkFalseKeepsLongTTSAsOneChunk() {
        let text = String(repeating: "This is a long sentence for synthesis. ", count: 20)
        let result = AuKSpeechPlanner.plan(
            AuKSpeechPlanRequest(
                task: "tts",
                input: text,
                autoChunk: false
            )
        )
        guard case .plan(let plan) = result else {
            Issue.record("expected plan")
            return
        }
        #expect(plan.chunks.count == 1)
    }

    @Test func streamEmitsMetadataThenFirstAudioBeforeLaterChunksGenerate() throws {
        let plan = AuKSpeechPlan(
            kind: .tts,
            chunks: [
                AuKPreparedChunk(instruction: "one", genSeconds: 1),
                AuKPreparedChunk(instruction: "two", genSeconds: 1),
                AuKPreparedChunk(instruction: "three", genSeconds: 1),
            ],
            pauseSeconds: 0.2
        )
        var log: [String] = []
        var clock: TimeInterval = 0
        try AuKStreamSession.run(
            sampleRate: 24_000,
            backend: "auk-base",
            variant: "base",
            plan: plan,
            generate: { chunk in
                log.append("generate:\(chunk.instruction)")
                clock += 1
                return [Float](repeating: 0.1, count: 8)
            },
            now: { clock },
            emit: { event in
                switch event {
                case .metadata(let metadata):
                    log.append("emit:metadata")
                    #expect(metadata.format == "pcm_s16le")
                    #expect(metadata.encoding == "base64")
                    #expect(metadata.backend == "auk-base")
                    #expect(metadata.variant == "base")
                    #expect(metadata.sampleRate == 24_000)
                    #expect(metadata.channels == 1)
                case .audio(let audio):
                    log.append("emit:audio:\(audio.chunk)")
                    if audio.chunk < 3 {
                        #expect(audio.samples == 8 + Int((0.2 * 24_000).rounded()))
                    } else {
                        #expect(audio.samples == 8)
                    }
                case .done(let done):
                    log.append("emit:done")
                    #expect(done.chunks == 3)
                    #expect(done.firstAudioSeconds < done.wallSeconds)
                    #expect(done.firstAudioSeconds >= 0)
                case .error:
                    Issue.record("unexpected error event")
                }
            }
        )
        #expect(
            log == [
                "emit:metadata",
                "generate:one",
                "emit:audio:1",
                "generate:two",
                "emit:audio:2",
                "generate:three",
                "emit:audio:3",
                "emit:done",
            ]
        )
    }

    @Test func streamJSONShapeMatchesContract() {
        let metadata = aukSpeechStreamJSON(
            .metadata(AuKStreamMetadataEvent(sampleRate: 24_000, backend: "auk-flash", variant: "flash"))
        )
        #expect(metadata["event"] as? String == "metadata")
        #expect(metadata["format"] as? String == "pcm_s16le")
        #expect(metadata["encoding"] as? String == "base64")
        #expect(metadata["backend"] as? String == "auk-flash")
        #expect(metadata["variant"] as? String == "flash")

        let audio = aukSpeechStreamJSON(
            .audio(AuKStreamAudioEvent(chunk: 1, samples: 12, seconds: 0.5, elapsedSeconds: 1.2, pcm: [0])),
            audioBase64: "AAA="
        )
        #expect(audio["event"] as? String == "audio")
        #expect(audio["chunk"] as? Int == 1)
        #expect(audio["samples"] as? Int == 12)
        #expect(audio["seconds"] as? Double == 0.5)
        #expect(audio["elapsed_seconds"] as? Double == 1.2)
        #expect(audio["audio"] as? String == "AAA=")

        let done = aukSpeechStreamJSON(
            .done(AuKStreamDoneEvent(firstAudioSeconds: 0.4, audioDurationSeconds: 2, wallSeconds: 3, chunks: 2))
        )
        #expect(done["event"] as? String == "done")
        #expect(done["first_audio_seconds"] as? Double == 0.4)
        #expect(done["audio_duration_seconds"] as? Double == 2)
        #expect(done["wall_seconds"] as? Double == 3)
        #expect(done["chunks"] as? Int == 2)
    }

    @Test func apiCompatibilityDecodesLegacyAndExtendedSpeechJSON() throws {
        let legacy = try JSONDecoder().decode(AuKSpeechJSON.self, from: Data(#"{"input":"hello"}"#.utf8))
        #expect(legacy.input == "hello")
        #expect(legacy.nfe == nil)
        #expect(legacy.task == nil)

        let extended = try JSONDecoder().decode(
            AuKSpeechJSON.self,
            from: Data(
                """
                {"input":"hi","instruction":"warm voice","task":"tts","nfe":16,"cfg":1.5,"sway":-0.5,"auto_chunk":true,"chunk_pause_ms":100,"gen_seconds":3,"ref_audio":"/tmp/a.wav"}
                """.utf8
            )
        )
        #expect(extended.task == "tts")
        #expect(extended.nfe == 16)
        #expect(extended.resolvedCFG == 1.5)
        #expect(extended.sway == -0.5)
        #expect(extended.resolvedPauseSeconds == 0.1)
        #expect(extended.planRequest(hasSourceAudio: false).hasSourceAudio)
    }

    @Test func chineseInstructTemplateIsTTS() {
        let instruction = "请基于下面的描述: \"温柔女声\",生成语音内容\"欢迎回家，今天工作顺利吗？还想再听一段更长的问候好让分块发生。还要再补一句。\"."
        let result = AuKSpeechPlanner.plan(
            AuKSpeechPlanRequest(
                instruction: instruction,
                chunkTargetCharacters: 20,
                chunkHardCharacterLimit: 30
            )
        )
        guard case .plan(let plan) = result else {
            Issue.record("expected CN instruct TTS plan")
            return
        }
        #expect(plan.kind == .tts)
        #expect(plan.chunks.allSatisfy { $0.instruction.contains("温柔女声") })
    }
}
