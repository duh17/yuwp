import Foundation
import Testing
@testable import NativeTTS

@Suite("AuK-Flash config, mapping, rotary, audio, sampling")
struct AuKFlashTests {
    @Test func flashConfigUsesFixedHopAndSampleRate() {
        #expect(AuKFlashConfig.name == "AuK-Flash")
        #expect(AuKFlashConfig.sampleRate == 24_000)
        #expect(AuKFlashConfig.thinkerSampleRate == 16_000)
        #expect(AuKFlashConfig.downsampleRate == 480)
        #expect(AuKFlashConfig.latentDim == 64)
        #expect(AuKFlashConfig.hopSize == 480)
        #expect(AuKFlashConfig.ditLayers == 10)
        #expect(AuKFlashConfig.ditSingleLayers == 20)
        #expect(AuKFlashConfig.thinkerLayers == 36)
        #expect(AuKFlashConfig.thinkerHeads == 16)
        #expect(AuKFlashConfig.thinkerKVHeads == 2)
        #expect(AuKFlashConfig.convPreIsCausal == false)
    }

    @Test func flashSamplingIgnoresCallerNFEAndCFG() {
        let resolved = AuKSampling.resolve(
            variant: .flash,
            nfe: 32,
            cfgStrength: 2.0,
            sway: -1.0,
            tGrid: [0, 0.5, 1]
        )
        #expect(resolved.nfe == 4)
        #expect(resolved.cfgStrength == 0)
        #expect(resolved.sway == nil)
        #expect(resolved.tGrid == AuKFlashConfig.tGrid)
        #expect(resolved.tGrid.count == 5)
        #expect(resolved.usesCFG == false)
        #expect(resolved.tGrid.first == 0)
        #expect(resolved.tGrid.last == 1)
        for (prev, next) in zip(resolved.tGrid, resolved.tGrid.dropFirst()) {
            #expect(next > prev)
        }
    }

    @Test func latentFrameCountMatchesHop() {
        #expect(AuKFlashConfig.latentFrames(seconds: 4) == 200)
        #expect(AuKFlashConfig.latentFrames(seconds: 0.01) == 1)
        #expect(AuKFlashConfig.latentFrames(seconds: 1.5) == 75)
    }

    @Test func yamlNameParserReadsAuKFlashAndAuKBase() throws {
        let flash = """
        model:
          name: AuK-Flash
          vae_name: BigVGANFlowVAE
        """
        #expect(aukModelName(fromYAML: flash) == "AuK-Flash")
        #expect(try aukParseVariant(named: "AuK-Flash") == .flash)
        #expect(try aukParseVariant(named: "AuK") == .base)
        #expect(try aukParseVariant(named: "base") == .base)
        #expect(throws: AuKError.self) {
            _ = try aukParseVariant(named: "Qwen3-TTS")
        }
    }

    @Test func directoryDetectionPrefersConvertedFlashThenPyTorchSource() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("auk-detect-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try "model:\n  name: AuK-Flash\n".write(to: root.appendingPathComponent("config.yaml"), atomically: true, encoding: .utf8)
        #expect(inspectAuKModelDirectory(root) == .pytorchSource(variant: .flash))

        try Data().write(to: root.appendingPathComponent("auk_flash.safetensors"))
        #expect(inspectAuKModelDirectory(root) == .pytorchSource(variant: .flash))

        try Data().write(to: root.appendingPathComponent("dit_flash.safetensors"))
        try Data().write(to: root.appendingPathComponent("fusion_flash.safetensors"))
        try Data().write(to: root.appendingPathComponent("vae.safetensors"))
        #expect(inspectAuKModelDirectory(root) == .converted(variant: .flash))
        #expect(detectTTSBackend(at: root) == .aukFlash)
    }

    @Test func qwen3TTSDirectoryIsNotClassifiedAsAuK() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("qwen3-detect-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try """
        {"tts_model_type":"custom_voice"}
        """.write(to: root.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        #expect(inspectAuKModelDirectory(root) == .unknown)
        #expect(detectTTSBackend(at: root) == .qwen3TTS)
    }

    @Test func ditAndThinkerKeyRemapsMatchOfficialConverter() {
        #expect(
            remapDiTKey("transformer.time_embed.time_mlp.2.weight")
                == .dit("time_embed.time_mlp.1.weight")
        )
        #expect(
            remapDiTKey("transformer.audio_embed.conv_pos_embed.conv1d.2.weight")
                == .dit("audio_embed.conv_pos_embed.conv1d.1.weight")
        )
        #expect(remapDiTKey("transformer.rotary_embed.inv_freq") == .fusion("inv_freq"))
        #expect(remapDiTKey("layer_weights") == .fusion("layer_weights"))
        #expect(remapDiTKey("layer_scale") == .fusion("layer_scale"))
        #expect(remapDiTKey("text_encoder.foo") == .skip)
        #expect(
            remapThinkerKey("thinker.model.layers.0.self_attn.q_proj.weight")
                == .thinker("layers.0.self_attn.q_proj.weight")
        )
        #expect(remapThinkerKey("thinker.model.embed_tokens.weight") == .thinker("embed_tokens.weight"))
        #expect(remapThinkerKey("thinker.audio_tower.conv1.weight") == .thinker("audio_tower.conv1.weight"))
        #expect(remapThinkerKey("thinker.visual.patch_embed.weight") == .skip)
        #expect(remapThinkerKey("thinker.lm_head.weight") == .skip)
        #expect(remapThinkerKey("thinker.audio_tower.audio_bos_eos_token.weight") == .skip)
        #expect(remapThinkerKey("talker.model.layers.0.weight") == .skip)
    }

    @Test func vaeKeyRemapsMatchOfficialConverter() {
        #expect(remapVAEKey("audio_encoder.generator.0.layer.weight") == "audio_encoder.pre.weight")
        #expect(remapVAEKey("audio_encoder.generator.2.layer.weight") == "audio_encoder.stages.0.down.weight")
        #expect(
            remapVAEKey("audio_encoder.generator.3.layers.0.1.weight")
                == "audio_encoder.stages.0.stack.layers.0.conv1.weight"
        )
        #expect(
            remapVAEKey("audio_encoder.generator.3.layers.0.3.bias")
                == "audio_encoder.stages.0.stack.layers.0.conv2.bias"
        )
        #expect(remapVAEKey("audio_encoder.generator.20.layer.weight") == "audio_encoder.post.weight")
        #expect(remapVAEKey("conv_pre.weight") == "decoder.conv_pre.weight")
        #expect(remapVAEKey("conv_post.weight") == "decoder.conv_post.weight")
        #expect(remapVAEKey("ups.0.0.weight") == "decoder.ups.0.weight")
        #expect(remapVAEKey("resblocks.2.convs1.1.bias") == "decoder.resblocks.2.convs1.1.bias")
        #expect(remapVAEKey("activation_post.act.alpha") == "decoder.activation_post.act.alpha")
        #expect(remapVAEKey("global_mean") == "global_mean")
        #expect(remapVAEKey("flow.foo") == nil)
        #expect(remapVAEKey("activation_post.upsample.filter") == nil)
        #expect(remapVAEKey("resblocks.0.activations.0.upsample.filter") == nil)
        #expect(remapVAEKey("resblocks.0.activations.0.downsample.lowpass.filter") == nil)
    }

    @Test func convLayoutsTransposeLikeOfficialMLXConverter() {
        let conv = Array(0 ..< 24).map(Float.init) // (O=2, I=3, K=4)
        let convMLX = transposeConv1dWeightCPU(conv, outChannels: 2, inChannels: 3, kernel: 4)
        #expect(convMLX.count == 24)
        #expect(convMLX[0] == 0)
        // src [o,i,k]=[0,0,1] -> dst [o,k,i]=[0,1,0] index (0*4+1)*3+0 = 3
        #expect(convMLX[3] == conv[1])

        let convT = Array(0 ..< 24).map(Float.init) // (I=3, O=2, K=4)
        let convTMLX = transposeConvTranspose1dWeightCPU(convT, inChannels: 3, outChannels: 2, kernel: 4)
        #expect(convTMLX.count == 24)
        #expect(convTMLX[0] == convT[0])
        // src [i,o,k]=[0,1,2] -> dst [o,k,i]=[1,2,0] index (1*4+2)*3+0 = 18
        #expect(convTMLX[18] == convT[(0 * 2 + 1) * 4 + 2])
    }

    @Test func weightNormFoldsOverNonLeadingAxes() {
        let folded = foldWeightNormCPU(g: [2, 2], v: [3, 0, 0, 4], rows: 2)
        #expect(abs(folded[0] - 2) < 1e-5)
        #expect(abs(folded[1]) < 1e-5)
        #expect(abs(folded[2]) < 1e-5)
        #expect(abs(folded[3] - 2) < 1e-5)
    }

    @Test func analyticInvFreqIsNotTheCheckpointFriendlyThreeQuarter() {
        let inv = analyticRoPEInvFreq(dimHead: 64, base: 10_000)
        #expect(inv.count == 32)
        #expect(abs(inv[0] - 1) < 1e-7)
        #expect(abs(inv[1] - 0.749894) < 1e-4)
        #expect(abs(inv[1] - 0.75) > 5e-5)
    }

    @Test func interleavedAndHalfSplitRotaryConventionsDiffer() {
        let x: [Float] = [1, 2, 3, 4]
        let cos: [Float] = [0, 0, 0, 0]
        let sin: [Float] = [1, 1, 1, 1]
        let interleaved = applyInterleavedRoPECPU(x, cos: cos, sin: sin)
        let half = applyHalfSplitRoPECPU(x, cos: cos, sin: sin)
        #expect(interleaved == [-2, 1, -4, 3])
        #expect(half == [-3, -4, 1, 2])
        #expect(interleaved != half)
    }

    @Test func kaiserSincFilterIsNormalizedSymmetricAndZeroAtCutoffZero() {
        let zeros = aukKaiserSincFilter1d(cutoff: 0, halfWidth: 0.3, kernelSize: 12)
        #expect(zeros.allSatisfy { abs($0) < 1e-12 })

        let filt = aukKaiserSincFilter1d(cutoff: 0.25, halfWidth: 0.3, kernelSize: 12)
        #expect(filt.count == 12)
        let sum = filt.reduce(0, +)
        #expect(abs(sum - 1) < 1e-5)
        for i in 0..<6 {
            #expect(abs(filt[i] - filt[11 - i]) < 1e-6)
        }
        #expect(abs(aukModifiedBesselI0(0) - 1) < 1e-12)
    }

    @Test func chatTemplateMatchesQwen25OmniDefaultSystemAndAudioPlaceholder() {
        let textOnly = aukApplyChatTemplate(instruction: "Say hello", hasAudio: false)
        #expect(
            textOnly == """
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            Say hello|<no_prompt_audio>|<|im_end|>
            <|im_start|>assistant

            """
        )
        let withAudio = aukApplyChatTemplate(instruction: "Clone this", hasAudio: true)
        #expect(
            withAudio == """
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            Clone this<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>
            <|im_start|>assistant

            """
        )
        #expect(throws: AuKError.self) {
            _ = try aukRejectUnsupportedMultimodal(contentTypes: ["video"])
        }
    }

    @Test func audioTokenCountFollowsWindowedConvThenAvgPool() {
        #expect(aukAudioTowerTokenCount(melFrames: 200) == 50)
        #expect(aukAudioTowerTokenCount(melFrames: 400) == 100)
        #expect(aukAudioTowerTokenCount(melFrames: 250) == 62)
        #expect(aukAudioTowerTokenCount(melFrames: 0) == 0)
        #expect(aukWhisperMelFrameCount(sampleCount: 16_000) == 100)
        #expect(aukWhisperMelFrameCount(sampleCount: 1_600) == 10)
    }

    @Test func audioPrepDownmixesBeforeResample() {
        let left: [Float] = [1, 1, 1, 1]
        let right: [Float] = [-1, -1, -1, -1]
        let mono = aukDownmixToMono(channels: [left, right])
        #expect(mono == [0, 0, 0, 0])
        #expect(aukDownmixToMono(channels: [left]) == left)
    }

    @Test func expandAudioPlaceholdersRepeatsAudioToken() {
        let ids = [10, AuKFlashConfig.audioTokenId, 20]
        #expect(aukExpandAudioTokens(ids, count: 3) == [10, 151_646, 151_646, 151_646, 20])
        #expect(aukExpandAudioTokens([1, 2, 3], count: 4) == [1, 2, 3])
    }

    @Test func convertedWeightGraphsFailClosedOnMissingOrExtraKeys() throws {
        #expect(throws: AuKError.self) {
            try aukRequireExactKeys(fileKeys: ["a"], modelKeys: ["a", "b"], label: "VAE")
        }
        #expect(throws: AuKError.self) {
            try aukRequireExactKeys(fileKeys: ["a", "c"], modelKeys: ["a", "b"], label: "DiT")
        }
        try aukRequireExactKeys(fileKeys: ["a", "b"], modelKeys: ["a", "b"], label: "Thinker")
    }

    @Test func incompleteVAEConversionFailsClosed() {
        #expect(aukExpectedVAETensorCount() == 611)
        #expect(throws: AuKError.self) {
            try aukRequireConvertedGraph(
                fileKeys: ["global_mean", "decoder.conv_pre.weight"],
                requiredKeys: ["audio_encoder.pre.weight", "decoder.conv_pre.weight"],
                expectedCount: aukExpectedVAETensorCount(),
                label: "VAE"
            )
        }
    }

    @Test func periodicHannUsesLengthNNotNMinusOne() {
        let periodic = aukPeriodicHann(size: 8)
        #expect(periodic.count == 8)
        #expect(abs(periodic[0]) < 1e-6)
        let lastPeriodic = 0.5 - 0.5 * cos(2 * Float.pi * 7 / 8)
        #expect(abs(periodic[7] - lastPeriodic) < 1e-6)
        #expect(abs(periodic[7]) > 1e-3)

        let symmetricLast = 0.5 * (1 - cos(2 * Float.pi * 7 / 7))
        #expect(abs(symmetricLast) < 1e-6)
        #expect(abs(periodic[7] - symmetricLast) > 1e-3)
    }

    @Test func perFrameDCRemovalSubtractsTheFrameMean() {
        #expect(aukRemoveFrameDC([3, 3, 3]) == [0, 0, 0])
        let centered = aukRemoveFrameDC([1, 2, 3])
        #expect(abs(centered[0] + 1) < 1e-6)
        #expect(abs(centered[1]) < 1e-6)
        #expect(abs(centered[2] - 1) < 1e-6)
        #expect(aukRemoveFrameDC([]) == [])
    }

    @Test func instructionResolutionPreservesOpenAIInputBeforeDeliveryInstructions() {
        #expect(aukResolveInstruction(instruction: "A", instructions: "B", input: "C") == "A")
        #expect(aukResolveInstruction(instruction: "  ", instructions: "B", input: "C") == "C")
        #expect(aukResolveInstruction(instruction: nil, instructions: "B", input: "  ") == "B")
        #expect(aukResolveInstruction(instruction: nil, instructions: "  ", input: "  ") == nil)
    }

    @Test func genSecondsValidationFailsClosedWithoutAudio() throws {
        #expect(throws: AuKError.self) {
            _ = try aukResolveGenSeconds(nil, hasReferenceAudio: false)
        }
        #expect(throws: AuKError.self) {
            _ = try aukResolveGenSeconds(0, hasReferenceAudio: true)
        }
        #expect(throws: AuKError.self) {
            _ = try aukResolveGenSeconds(-1, hasReferenceAudio: false)
        }
        #expect(try aukResolveGenSeconds(3, hasReferenceAudio: false) == 3)
        #expect(try aukResolveGenSeconds(nil, hasReferenceAudio: true) == nil)
    }

    @Test func omittedBitsResolvesToFP32EvenIfQuantizedFilesExist() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("auk-bits-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data().write(to: root.appendingPathComponent("dit_flash.safetensors"))
        try Data().write(to: root.appendingPathComponent("dit_flash.q8.safetensors"))
        try Data().write(to: root.appendingPathComponent("dit_flash.q4.safetensors"))
        let resolved = resolveAuKWeightFile(directory: root, baseName: "dit_flash", bits: nil)
        #expect(resolved.url.lastPathComponent == "dit_flash.safetensors")
        #expect(resolved.bits == nil)
        let q8 = resolveAuKWeightFile(directory: root, baseName: "dit_flash", bits: 8)
        #expect(q8.url.lastPathComponent == "dit_flash.q8.safetensors")
        #expect(q8.bits == 8)
    }
}
