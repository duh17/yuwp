import Foundation
@preconcurrency import MLX
import Testing
@testable import NativeTTS

@Suite("Drama 3 conversion and prompt rules", .serialized)
struct Drama3Tests {
    @Test func e4m3DecodeCoversNormalSubnormalSignAndNaN() {
        let table = Drama3Convert.e4m3Values
        #expect(table[0x38] == 1.0)
        #expect(table[0xB8] == -1.0)
        #expect(table[0x7E] == 448.0) // largest finite E4M3
        #expect(table[0x01] == Float(pow(2.0, -9.0))) // smallest subnormal
        #expect(table[0x08] == Float(pow(2.0, -6.0))) // smallest normal
        #expect(table[0x7F].isNaN)
        #expect(table[0xFF].isNaN)
    }

    @Test func fp8DequantizationAppliesEachBlockScaleAndCropsPartialBlocks() {
        // 130×3 weight of 1.0 (0x38): rows 0..<128 use scale row 0, rows 128..<130 scale row 1.
        Stream.withNewDefaultStream(device: .cpu) {
            let weight = MLXArray([UInt8](repeating: 0x38, count: 130 * 3), [130, 3])
            let scale = MLXArray([Float(2), Float(-0.5)], [2, 1])
            let out = Drama3Convert.dequantizeFP8(weight, scaleInverse: scale).asType(.float32)
            #expect(out.shape == [130, 3])
            #expect(out[0, 0].item(Float.self) == 2)
            #expect(out[127, 2].item(Float.self) == 2)
            #expect(out[128, 0].item(Float.self) == -0.5)
        }
    }

    @Test func zeroCenteredNormsAreShiftedButGatedDeltaNormIsNot() throws {
        #expect(Drama3Convert.isZeroCenteredNorm(try #require(try Drama3Convert.mapKey("model.language_model.norm.weight"))))
        #expect(Drama3Convert.isZeroCenteredNorm(try #require(try Drama3Convert.mapKey(
            "audio_decoder_mtp.layers.3.transformer_layer.attention.q_norm.weight"))))
        #expect(!Drama3Convert.isZeroCenteredNorm(try #require(try Drama3Convert.mapKey(
            "model.language_model.layers.0.linear_attn.norm.weight"))))
        #expect(throws: Drama3Error.self) { try Drama3Convert.mapKey("vision_tower.patch.weight") }
    }

    @Test func callerTextAddsDefaultSpeakerUnlessALeadingTagIsPresent() {
        #expect(Drama3Engine.withSpeakerTag("Hello") == "<|speaker:0|> Hello")
        #expect(Drama3Engine.withSpeakerTag("  <|speaker:3|> Hi") == "  <|speaker:3|> Hi")
        #expect(Drama3Engine.withSpeakerTag("<|speaker:16|> Hi") == "<|speaker:0|> <|speaker:16|> Hi")
    }

    @Test func callerTextRejectsStructuralTokensAndMalformedSpeakerTags() throws {
        try Drama3Engine.validateCallerText("<|speaker:15|> fine [laughs]", location: "input")
        #expect(throws: Drama3Error.self) { try Drama3Engine.validateCallerText("  \n", location: "input") }
        #expect(throws: Drama3Error.self) { try Drama3Engine.validateCallerText("hi <|im_end|>", location: "input") }
        #expect(throws: Drama3Error.self) { try Drama3Engine.validateCallerText("<|speaker:07|> hi", location: "input") }
        #expect(throws: Drama3Error.self) { try Drama3Engine.validateCallerText("<|speaker:16|> hi", location: "input") }
    }

    @Test func scriptChunksReopenWithTheActiveSpeakerTag() {
        let long = String(repeating: "Hagrid kept talking about dragons. ", count: 12)
        let chunks = Drama3ScriptChunker.chunk(
            "Narration first. <|speaker:1|> \(long)<|speaker:2|> Short reply.",
            targetCharacters: 200,
            hardCharacterLimit: 300
        )
        #expect(chunks.count == 4)
        #expect(chunks[0] == "Narration first.")
        // Every later chunk opens with the speaker who is talking there, not the default speaker 0.
        for chunk in chunks.dropFirst() {
            #expect(chunk.hasPrefix("<|speaker:1|>") || chunk.hasPrefix("<|speaker:2|>"))
        }
        #expect(chunks.last?.hasSuffix("<|speaker:2|> Short reply.") == true)
        // A turn split across chunks does not repeat its tag inside one chunk.
        #expect(chunks.allSatisfy { $0.components(separatedBy: "<|speaker:1|>").count <= 2 })
    }

    @Test func frameBudgetCoversSlowReadingButStopsRunaways() {
        let framesPerSecond = 44_100.0 / 2_048.0
        func seconds(_ script: String) -> Double { Double(Drama3Sampling.frameBudget(forScript: script)) / framesPerSecond }
        // Observed: this two-voice line ends in 54-95 frames (2.5-4.4 s); a runaway runs to 2,048.
        let line = "<|speaker:0|> The hall was quiet. <|speaker:1|> [whispering] Did you hear that?"
        #expect((8 ... 15).contains(seconds(line)))
        // About 100 Chinese characters take ~22 s at a relaxed pace; Latin-rate counting would cut it off.
        let chinese = String(repeating: "今天天气很好，我们去公园散步吧。", count: 7)
        #expect(seconds(chinese) > 40)
        // Point tags add time; scoped-tag markup does not, but its words do.
        #expect(abs(seconds("Well [sighs] fine. [pause] Okay.") - seconds("Well fine. Okay.") - 3) < 0.1)
        #expect(seconds("Promise me <whisper>you won't tell</whisper>.") == seconds("Promise me you won't tell."))
    }

    @Test func shortScriptStaysVerbatimIncludingLineBreaks() {
        let script = "<|speaker:0|> [laughs] Okay, fine. <|speaker:1|> [sighs] You always say that."
        #expect(Drama3ScriptChunker.chunk(script) == [script])
        let note = "Direction: icy calm, as if you already accepted it.\nLine: So this is goodbye."
        #expect(Drama3ScriptChunker.chunk("\n\(note)\n") == [note])
        #expect(Drama3ScriptChunker.chunk("  \n ") == [])
    }

    @Test(.enabled(if: FileManager.default.fileExists(atPath: Drama3Tests.codecPath)))
    func streamingCodecChunksMatchFullDecode() throws {
        let codec = try FishCodec.load(url: URL(fileURLWithPath: Self.codecPath))
        var codes: [Int32] = []
        codes.reserveCapacity(10 * 9)
        var next: UInt32 = 1
        for _ in 0 ..< (10 * 9) {
            next = next &* 1_103_515_245 &+ 12_345
            codes.append(Int32(next % 1024))
        }
        let all = MLXArray(codes, [10, 9])
        let batched = codec.decode(codes: all)
        codec.resetStreaming()
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: 9, by: 3) {
            chunks.append(codec.streamingStep(codes: all[0..., start ..< (start + 3)]))
        }
        let streamed = concatenated(chunks)
        eval(batched, streamed)
        #expect(batched.shape == streamed.shape)
        let diff = abs(batched - streamed).max().item(Float.self)
        #expect(diff < 2e-3)
    }

    private static var codecPath: String {
        NSHomeDirectory() + "/Library/Application Support/Yuwp/models/drama3-mlx-4bit/codec.safetensors"
    }
}
