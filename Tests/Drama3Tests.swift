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
}
