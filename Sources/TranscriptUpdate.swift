import Foundation

/// The semantic kind of a transcript update delivered to clients.
enum TranscriptUpdateKind: String, Sendable, Codable, Equatable {
    case partial = "partial"
    case segmentCommit = "segment_commit"
    case final = "final"

    /// Preview behavior: only final updates should force an immediate settle.
    /// Segment commits still go through the typewriter path so append-heavy
    /// speech feels continuous instead of jumping a whole sentence at once.
    var settlesPreviewImmediately: Bool {
        self == .final
    }

    /// Default target behavior: only final updates should commit/finish the target surface.
    var commitsTargetText: Bool {
        self == .final
    }
}

/// A client-facing transcript update.
///
/// `text` is always the full visible transcript. When available, the server may
/// also provide the explicit `committedText` / `activeText` split. Until then,
/// the client can derive a best-effort split from prior segment commits.
struct TranscriptUpdate: Sendable, Codable, Equatable {
    let kind: TranscriptUpdateKind
    let text: String
    let committedText: String?
    let activeText: String?

    init(
        kind: TranscriptUpdateKind,
        text: String,
        committedText: String? = nil,
        activeText: String? = nil
    ) {
        self.kind = kind
        self.text = text
        self.committedText = committedText
        self.activeText = activeText
    }
}

/// Client-side resolved transcript state used by desktop injectors/UI.
struct TranscriptState: Sendable, Equatable {
    var committedText: String
    var activeText: String

    static let empty = TranscriptState(committedText: "", activeText: "")

    var fullText: String {
        Self.appendSegment(committedText, activeText)
    }

    func applying(_ update: TranscriptUpdate) -> TranscriptState {
        switch update.kind {
        case .partial:
            if let explicit = Self.explicitState(from: update) {
                return explicit
            }

            let fullText = update.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if let split = Self.splitActiveText(from: fullText, committedText: committedText) {
                return TranscriptState(committedText: committedText, activeText: split)
            }
            return TranscriptState(committedText: "", activeText: fullText)

        case .segmentCommit, .final:
            return TranscriptState(
                committedText: update.text.trimmingCharacters(in: .whitespacesAndNewlines),
                activeText: ""
            )
        }
    }

    static func appendSegment(_ committed: String, _ active: String) -> String {
        let trimmedActive = active.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedActive.isEmpty { return committed }
        if committed.isEmpty { return trimmedActive }
        return committed + " " + trimmedActive
    }

    private static func explicitState(from update: TranscriptUpdate) -> TranscriptState? {
        guard update.committedText != nil || update.activeText != nil else { return nil }
        let committed = update.committedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let active = update.activeText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return TranscriptState(committedText: committed, activeText: active)
    }

    private static func splitActiveText(from fullText: String, committedText: String) -> String? {
        guard !committedText.isEmpty else { return fullText }
        if fullText == committedText { return "" }

        let separator = committedText + " "
        guard fullText.hasPrefix(separator) else { return nil }
        return String(fullText.dropFirst(separator.count))
    }
}
