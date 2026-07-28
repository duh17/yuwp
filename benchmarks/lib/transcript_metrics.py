"""Transparent transcript normalization and corpus-level WER/CER scoring."""

from __future__ import annotations

import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence, TypeVar

T = TypeVar("T")


def _normalized_characters(text: str, *, punctuation_as_space: bool) -> str:
    output: list[str] = []
    for character in unicodedata.normalize("NFKC", text).casefold():
        category = unicodedata.category(character)
        if category[0] in {"P", "S"}:
            if punctuation_as_space:
                output.append(" ")
            continue
        output.append(character)
    return "".join(output)


def normalize_for_wer(text: str) -> str:
    """Normalize case, compatibility forms, punctuation, and whitespace for WER."""
    return " ".join(_normalized_characters(text, punctuation_as_space=True).split())


def normalize_for_cer(text: str) -> str:
    """Normalize as for WER, then remove all whitespace for CJK-friendly CER."""
    return "".join(_normalized_characters(text, punctuation_as_space=False).split())


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


@dataclass
class ErrorRateAccumulator:
    """Accumulate edit counts for a micro-averaged corpus WER or CER."""

    metric: str
    errors: int = 0
    reference_units: int = 0
    utterance_rates: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.metric not in {"wer", "cer"}:
            raise ValueError(f"unsupported error-rate metric: {self.metric}")

    def add(self, reference: str, hypothesis: str) -> None:
        if self.metric == "wer":
            reference_units: Sequence[str] = normalize_for_wer(reference).split()
            hypothesis_units: Sequence[str] = normalize_for_wer(hypothesis).split()
        else:
            reference_units = normalize_for_cer(reference)
            hypothesis_units = normalize_for_cer(hypothesis)

        errors = levenshtein_distance(reference_units, hypothesis_units)
        unit_count = len(reference_units)
        self.errors += errors
        self.reference_units += unit_count
        self.utterance_rates.append(errors / max(unit_count, 1))

    def summary(self) -> dict[str, int | float]:
        return {
            "errors": self.errors,
            "reference_units": self.reference_units,
            "error_rate": self.errors / max(self.reference_units, 1),
            "macro_error_rate": statistics.mean(self.utterance_rates) if self.utterance_rates else 0.0,
            "utterances": len(self.utterance_rates),
        }
