"""Transparent transcript normalization and corpus-level WER/CER scoring."""

from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence, TypeVar

T = TypeVar("T")


def _spoken_text(text: str, *, strip_annotations: bool) -> str:
    if strip_annotations:
        # Reference annotations and speaker labels are metadata, not spoken words.
        text = re.sub(r"\[[^\]]*\]", " ", text)
    # AMI writes spoken initialisms as X_M_L_; ASR systems normally emit XML.
    return re.sub(
        r"(?<![A-Za-z])(?:[A-Za-z]_){2,}[A-Za-z]?(?![A-Za-z])",
        lambda match: match.group(0).replace("_", ""),
        text,
    )


def _normalized_characters(
    text: str,
    *,
    punctuation_as_space: bool,
    strip_annotations: bool,
) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        _spoken_text(text, strip_annotations=strip_annotations),
    ).casefold()
    # Apostrophes inside words are orthography, not spoken word boundaries.
    normalized = re.sub(r"(?<=\w)['’](?=\w)", "", normalized)

    output: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if category[0] in {"P", "S"}:
            if punctuation_as_space:
                output.append(" ")
            continue
        output.append(character)
    return "".join(output)


def normalize_for_wer(text: str, *, strip_annotations: bool = True) -> str:
    """Normalize case, compatibility forms, punctuation, and whitespace for WER."""
    return " ".join(
        _normalized_characters(
            text,
            punctuation_as_space=True,
            strip_annotations=strip_annotations,
        ).split()
    )


def normalize_for_cer(text: str, *, strip_annotations: bool = True) -> str:
    """Normalize as for WER, then remove all whitespace for CJK-friendly CER."""
    return "".join(
        _normalized_characters(
            text,
            punctuation_as_space=False,
            strip_annotations=strip_annotations,
        ).split()
    )


def levenshtein_distance(reference: Sequence[T], hypothesis: Sequence[T]) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class EditCounts:
    correct: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def _last_edit_row(reference: Sequence[T], hypothesis: Sequence[T]) -> list[int]:
    """Return the final Levenshtein row using linear memory."""
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous


def _add_counts(left: EditCounts, right: EditCounts) -> EditCounts:
    return EditCounts(
        correct=left.correct + right.correct,
        substitutions=left.substitutions + right.substitutions,
        deletions=left.deletions + right.deletions,
        insertions=left.insertions + right.insertions,
    )


def edit_counts(reference: Sequence[T], hypothesis: Sequence[T]) -> EditCounts:
    """Return deterministic Levenshtein C/S/D/I counts with linear memory."""
    if not reference:
        return EditCounts(insertions=len(hypothesis))
    if not hypothesis:
        return EditCounts(deletions=len(reference))

    # Hirschberg alignment keeps long-form transcripts from allocating an
    # O(reference × hypothesis) Python integer matrix.
    if len(hypothesis) > len(reference):
        reversed_counts = edit_counts(hypothesis, reference)
        return EditCounts(
            correct=reversed_counts.correct,
            substitutions=reversed_counts.substitutions,
            deletions=reversed_counts.insertions,
            insertions=reversed_counts.deletions,
        )

    if len(reference) == 1:
        if reference[0] in hypothesis:
            return EditCounts(correct=1, insertions=len(hypothesis) - 1)
        return EditCounts(substitutions=1, insertions=len(hypothesis) - 1)
    if len(hypothesis) == 1:
        if hypothesis[0] in reference:
            return EditCounts(correct=1, deletions=len(reference) - 1)
        return EditCounts(substitutions=1, deletions=len(reference) - 1)

    midpoint = len(reference) // 2
    left_costs = _last_edit_row(reference[:midpoint], hypothesis)
    right_costs = _last_edit_row(reference[midpoint:][::-1], hypothesis[::-1])
    split = min(
        range(len(hypothesis) + 1),
        key=lambda index: left_costs[index] + right_costs[len(hypothesis) - index],
    )
    return _add_counts(
        edit_counts(reference[:midpoint], hypothesis[:split]),
        edit_counts(reference[midpoint:], hypothesis[split:]),
    )


@dataclass
class ErrorRateAccumulator:
    """Accumulate edit counts for a micro-averaged corpus WER or CER."""

    metric: str
    errors: int = 0
    reference_units: int = 0
    correct: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    utterance_rates: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.metric not in {"wer", "cer"}:
            raise ValueError(f"unsupported error-rate metric: {self.metric}")

    def add(self, reference: str, hypothesis: str) -> None:
        if self.metric == "wer":
            reference_units: Sequence[str] = normalize_for_wer(reference, strip_annotations=True).split()
            hypothesis_units: Sequence[str] = normalize_for_wer(hypothesis, strip_annotations=False).split()
        else:
            reference_units = normalize_for_cer(reference, strip_annotations=True)
            hypothesis_units = normalize_for_cer(hypothesis, strip_annotations=False)

        counts = edit_counts(reference_units, hypothesis_units)
        unit_count = len(reference_units)
        self.errors += counts.errors
        self.reference_units += unit_count
        self.correct += counts.correct
        self.substitutions += counts.substitutions
        self.deletions += counts.deletions
        self.insertions += counts.insertions
        self.utterance_rates.append(counts.errors / max(unit_count, 1))

    def summary(self) -> dict[str, int | float]:
        return {
            "errors": self.errors,
            "reference_units": self.reference_units,
            "correct": self.correct,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "error_rate": self.errors / max(self.reference_units, 1),
            "macro_error_rate": statistics.mean(self.utterance_rates) if self.utterance_rates else 0.0,
            "utterances": len(self.utterance_rates),
        }
