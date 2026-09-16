import MLX
import Testing
@testable import NativeASR

// Run separately from the server/VAD suite: inference is deliberately serial.
@Suite(.serialized) struct DraftVerificationModelTests {
    @Test(arguments: [true, false])
    func causalBlockMatchesSerialLogitsAndRewind(tied: Bool) {
        Stream.withNewDefaultStream(device: .cpu) {
            let model = Qwen3ASRModel(config: Qwen3ASRConfig(
                audioConfig: AudioEncoderConfig(
                    encoderLayers: 0, encoderAttentionHeads: 2, encoderFfnDim: 64,
                    dModel: 32, outputDim: 32, downsampleHiddenSize: 8
                ),
                textConfig: TextDecoderConfig(
                    vocabSize: 64, hiddenSize: 32, intermediateSize: 64,
                    numHiddenLayers: 2, numAttentionHeads: 4, numKeyValueHeads: 2,
                    headDim: 8, tieWordEmbeddings: tied
                )
            ))
            model.train(false)
            func ids(_ values: [Int32]) -> MLXArray {
                MLXArray(values).expandedDimensions(axis: 0)
            }
            let prompt = ids([1, 2, 3, 4, 5])
            let (initial, blockCache) = model(inputIds: prompt)
            let (_, serialCache) = model(inputIds: prompt)
            #expect(initial.shape == [1, 1, 64])
            let block: [Int32] = [6, 7, 8, 9]
            let (blockLogits, _) = model(
                inputIds: ids(block), cache: blockCache, logitPositions: block.count
            )
            eval(blockLogits)
            #expect(blockLogits.shape == [1, 4, 64])
            var serial: [MLXArray] = []
            for token in block {
                let (logits, _) = model(inputIds: ids([token]), cache: serialCache)
                eval(logits)
                serial.append(logits)
            }
            let expected = MLX.concatenated(serial, axis: 1)
            #expect(MLX.max(MLX.abs(blockLogits - expected)).item(Float.self) < 0.0001)

            // Discard two proposed positions, then process a different correction.
            for cache in blockCache { cache.trim(n: 2) }
            let (_, freshCache) = model(inputIds: ids([1, 2, 3, 4, 5, 6, 7]))
            let (rewound, _) = model(inputIds: ids([10]), cache: blockCache)
            let (fresh, _) = model(inputIds: ids([10]), cache: freshCache)
            #expect(MLX.max(MLX.abs(rewound - fresh)).item(Float.self) < 0.0001)
            #expect(blockCache.allSatisfy { $0.offset == 8 })
        }
    }
}
