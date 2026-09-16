import Testing
@testable import NativeASR

@Suite struct TranscriptDraftTests {
    @Test func draftsFollowTheLongestAlreadyVerifiedSuffix() {
        let draft = TranscriptDraft(tokens: [1, 2, 3, 4, 2, 5, 6])
        #expect(draft.continuation(after: [90, 1, 2], limit: 3) == [3, 4, 2])
        #expect(draft.continuation(after: [4, 2], limit: 8) == [5, 6])
        #expect(draft.continuation(after: [99], limit: 8).isEmpty)
        #expect(draft.continuation(after: [Qwen3ASRTokenizer.asrText], limit: 2) == [1, 2])
        #expect(draft.continuation(after: [1], limit: 0).isEmpty)
    }

    private func run(
        expected: [Int], draft: [Int], cap: Int = 100
    ) -> (tokens: [Int], calls: Int, rewound: Int, validCache: Bool) {
        var cache: [Int] = []
        var calls = 0
        var rewound = 0
        let tokens = TranscriptDraft.greedyDecode(
            firstToken: expected.first ?? 99,
            maxTokens: cap,
            eosTokens: [99],
            draft: TranscriptDraft(tokens: draft),
            verify: { inputs in
                calls += 1
                return inputs.map { token in
                    cache.append(token)
                    // Predictions after a wrong draft prefix must not leak into output.
                    guard cache == Array(expected.prefix(cache.count)) else { return 98 }
                    return cache.count < expected.count ? expected[cache.count] : 99
                }
            },
            rewind: { count in
                rewound += count
                cache.removeLast(count)
            }
        )
        return (tokens, calls, rewound, cache == Array(expected.prefix(cache.count)))
    }

    @Test func exactDraftProducesGreedyOutputWithFewerForwardPasses() {
        let expected = Array(1...30)
        let result = run(expected: expected, draft: expected)
        #expect(result.tokens == expected)
        #expect(result.calls <= 5)
        #expect(result.rewound == 0)
        #expect(result.validCache)
    }

    @Test func mismatchRollsBackUnverifiedInputsBeforeResuming() {
        let expected = Array(1...18)
        for mismatch in 1..<8 {
            var draft = expected
            draft[mismatch] = 97
            let result = run(expected: expected, draft: draft)
            #expect(result.tokens == expected)
            #expect(result.rewound > 0)
            #expect(result.validCache)
        }
    }

    @Test func divergentOrEmptyDraftNeverChangesGreedyChoices() {
        for draft in [[], [76, 77], [1, 98, 97, 96], [2, 1, 3, 1, 9]] {
            let result = run(expected: Array(1...12), draft: draft)
            #expect(result.tokens == Array(1...12))
            #expect(result.validCache)
        }
    }

    @Test func eosInsideBlockAndFirstTokenAreNeverEmitted() {
        #expect(run(expected: [1, 2, 99, 3], draft: [1, 2, 99, 3]).tokens == [1, 2])
        let firstEOS = run(expected: [99], draft: [1, 2])
        #expect(firstEOS.tokens.isEmpty)
        #expect(firstEOS.calls == 0)
    }

    @Test func capsApplyInsideBlocksWithoutAnExtraForwardPass() {
        for cap in 0...12 {
            let result = run(expected: Array(1...20), draft: Array(1...20), cap: cap)
            #expect(result.tokens == Array((1...20).prefix(cap)))
        }
        #expect(run(expected: [1], draft: [1], cap: 1).calls == 0)
    }

    @Test func repetitionTerminationMatchesTheSerialBatchDecoder() {
        let repeats = Array(repeating: 3, count: 20)
        let result = run(expected: repeats, draft: repeats)
        #expect(result.tokens == Array(repeating: 3, count: 10))
    }
}
