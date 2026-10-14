import Foundation

/// Splits a Drama 3 script into generation-sized pieces for long narration.
///
/// One Drama 3 call is capped by `Drama3Sampling.maxFrames` (~95 s), and every call re-sends the
/// same references, so a long script is generated chunk by chunk. A turn that spans a chunk
/// boundary keeps its voice because each chunk opens with the `<|speaker:N|>` tag active there.
public enum Drama3ScriptChunker {
    public static func chunk(
        _ script: String,
        targetCharacters: Int = 600,
        hardCharacterLimit: Int = 900
    ) -> [String] {
        // A script that fits stays verbatim: line breaks carry meaning (`Direction:` / `Line:` notes).
        let trimmedScript = script.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedScript.count <= hardCharacterLimit {
            return trimmedScript.isEmpty ? [] : [trimmedScript]
        }
        var chunks: [String] = []
        var current = ""
        var currentTag: String?

        func flush() {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { chunks.append(trimmed) }
            current = ""
            currentTag = nil
        }

        for turn in turns(script) {
            let pieces = LongFormTTSChunker.chunk(
                turn.body,
                targetCharacters: targetCharacters,
                hardCharacterLimit: hardCharacterLimit
            )
            for piece in pieces {
                // Repeat the tag only when the speaker changes inside a chunk or a chunk starts.
                let needsTag = turn.tag != nil && (current.isEmpty || turn.tag != currentTag)
                let segment = needsTag ? "\(turn.tag ?? "") \(piece)" : piece
                if !current.isEmpty, current.count + 1 + segment.count > targetCharacters {
                    flush()
                    current = turn.tag.map { "\($0) \(piece)" } ?? piece
                } else {
                    current = current.isEmpty ? segment : "\(current) \(segment)"
                }
                if turn.tag != nil { currentTag = turn.tag }
            }
        }
        flush()
        return chunks
    }

    /// Script text split at each valid `<|speaker:N|>` tag. Text before the first tag has no tag.
    static func turns(_ script: String) -> [(tag: String?, body: String)] {
        var result: [(tag: String?, body: String)] = []
        var tag: String?
        var body = ""
        var index = script.startIndex
        while index < script.endIndex {
            if script[index] == "<", let next = Drama3Engine.leadingSpeakerTag(String(script[index...])) {
                result.append((tag, body))
                tag = next
                body = ""
                index = script.index(index, offsetBy: next.count)
                continue
            }
            body.append(script[index])
            index = script.index(after: index)
        }
        result.append((tag, body))
        return result.filter { !$0.body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }
}
