import Foundation

public enum LongFormTTSChunker {
    public static func chunk(
        _ text: String,
        targetCharacters: Int = 220,
        hardCharacterLimit: Int = 320
    ) -> [String] {
        let normalized = normalizeWhitespace(text)
        guard normalized.count > hardCharacterLimit else { return normalized.isEmpty ? [] : [normalized] }

        let sentences = splitSentences(normalized)
        guard !sentences.isEmpty else { return splitLongSentence(normalized, hardCharacterLimit: hardCharacterLimit) }

        var chunks: [String] = []
        var current = ""

        func flushCurrent() {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                chunks.append(trimmed)
            }
            current = ""
        }

        for sentence in sentences {
            let trimmedSentence = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedSentence.isEmpty else { continue }

            if trimmedSentence.count > hardCharacterLimit {
                flushCurrent()
                chunks.append(contentsOf: splitLongSentence(trimmedSentence, hardCharacterLimit: hardCharacterLimit))
                continue
            }

            let candidate = current.isEmpty ? trimmedSentence : current + " " + trimmedSentence
            if candidate.count <= targetCharacters || current.isEmpty {
                current = candidate
                continue
            }

            flushCurrent()
            current = trimmedSentence
        }

        flushCurrent()
        return chunks
    }

    private static func normalizeWhitespace(_ text: String) -> String {
        text
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func splitSentences(_ text: String) -> [String] {
        var result: [String] = []
        var current = ""
        let sentenceEnders: Set<Character> = [".", "!", "?", "。", "！", "？"]

        for character in text {
            current.append(character)
            if sentenceEnders.contains(character) {
                let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    result.append(trimmed)
                }
                current = ""
            }
        }

        let trailing = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trailing.isEmpty {
            result.append(trailing)
        }
        return result
    }

    private static func splitLongSentence(_ text: String, hardCharacterLimit: Int) -> [String] {
        let words = text.split(separator: " ")
        guard !words.isEmpty else { return [] }

        var chunks: [String] = []
        var current = ""

        func flushCurrent() {
            guard !current.isEmpty else { return }
            chunks.append(current)
            current = ""
        }

        for word in words {
            let wordString = String(word)

            if wordString.count > hardCharacterLimit {
                flushCurrent()
                chunks.append(contentsOf: splitOversizedWord(wordString, hardCharacterLimit: hardCharacterLimit))
                continue
            }

            let candidate = current.isEmpty ? wordString : current + " " + wordString
            if candidate.count <= hardCharacterLimit {
                current = candidate
            } else {
                flushCurrent()
                current = wordString
            }
        }

        flushCurrent()
        return chunks
    }

    private static func splitOversizedWord(_ word: String, hardCharacterLimit: Int) -> [String] {
        guard hardCharacterLimit > 0 else { return [word] }

        var chunks: [String] = []
        var current = ""
        current.reserveCapacity(min(word.count, hardCharacterLimit))

        for character in word {
            current.append(character)
            if current.count == hardCharacterLimit {
                chunks.append(current)
                current = ""
            }
        }

        if !current.isEmpty {
            chunks.append(current)
        }
        return chunks
    }
}
