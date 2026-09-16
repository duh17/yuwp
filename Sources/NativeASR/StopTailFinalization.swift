import Foundation

/// Whether stop can skip the unexposed tail streaming decode.
enum StopTailPlan: Equatable, Sendable {
    case finalizeOnly
    case processThenFinalize
    case skipProvisionalDecode
}

enum StopTailExecution: Equatable, Sendable {
    case finalizeOnly
    case processedThenFinalized
    case skippedProvisionalDecode
    case skippedThenFallbackDecoded
}

struct StopTailPlanInputs: Equatable, Sendable {
    var pendingSampleCount: Int
    var batchRetranscribe: Bool
    var wouldProvisionallyDecode: Bool
    var strategy: StopBatchStrategy
    var sessionAudioSampleCountAfterIngest: Int
    var activeAudioSampleCountAfterIngest: Int
    var maxSessionContextSamples: Int
    var maxLiveBatchSegmentSamples: Int
    /// After ingesting the stop tail. Skip is unsafe when this is false and the
    /// active segment is already past the live-batch cap: fallback decode then
    /// hits `shouldAppendActiveTextFallback`, which drops energy-only recovery
    /// words that original `finalize()` keeps via the long-segment streaming path.
    var hasEnoughSpeechAfterIngest: Bool = true
}

public struct StopFinishResult: Sendable {
    public let text: String
    public let skippedProvisionalDecode: Bool
    public let usedFallbackDecode: Bool
    public let pendingSampleCount: Int
    public let sessionSampleCount: Int
}

enum StopTailExecutor {
    static func plan(_ inputs: StopTailPlanInputs) -> StopTailPlan {
        guard inputs.pendingSampleCount > 0 else { return .finalizeOnly }
        guard inputs.batchRetranscribe, inputs.wouldProvisionallyDecode else {
            return .processThenFinalize
        }
        switch inputs.strategy {
        case .none:
            return .processThenFinalize
        case .fullSession:
            return .skipProvisionalDecode
        case .activeSegmentOnly:
            // First-segment energy recovery can be `.activeSegmentOnly` while
            // `hasEnoughSpeech` is still false. Skip-path fallback then uses
            // `appendActiveTextFallback`, which drops those streaming words;
            // original `finalize()` keeps them via the long-segment path.
            if !inputs.hasEnoughSpeechAfterIngest
                && inputs.activeAudioSampleCountAfterIngest > inputs.maxLiveBatchSegmentSamples
            {
                return .processThenFinalize
            }
            if inputs.sessionAudioSampleCountAfterIngest > inputs.maxSessionContextSamples
                && inputs.activeAudioSampleCountAfterIngest > inputs.maxLiveBatchSegmentSamples
            {
                return .processThenFinalize
            }
            return .skipProvisionalDecode
        }
    }

    static func isUsableBatchText(_ text: String) -> Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    static func execute(
        plan: StopTailPlan,
        processThenFinalize: () -> String,
        finalizeOnly: () -> String,
        ingestWithoutDecode: () -> Void,
        finalizeBatch: () -> String?,
        fallbackDecodeThenText: () -> String
    ) -> (text: String, execution: StopTailExecution) {
        switch plan {
        case .finalizeOnly:
            return (finalizeOnly(), .finalizeOnly)
        case .processThenFinalize:
            return (processThenFinalize(), .processedThenFinalized)
        case .skipProvisionalDecode:
            ingestWithoutDecode()
            if let text = finalizeBatch(), isUsableBatchText(text) {
                return (text, .skippedProvisionalDecode)
            }
            return (fallbackDecodeThenText(), .skippedThenFallbackDecoded)
        }
    }
}
