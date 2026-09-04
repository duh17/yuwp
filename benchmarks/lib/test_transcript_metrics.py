from __future__ import annotations

import unittest
from itertools import product

from benchmarks.lib.transcript_metrics import (
    ErrorRateAccumulator,
    edit_counts,
    levenshtein_distance,
    normalize_for_cer,
    normalize_for_wer,
)


class TranscriptNormalizationTests(unittest.TestCase):
    def test_wer_normalizes_unicode_case_punctuation_and_whitespace(self) -> None:
        self.assertEqual(
            normalize_for_wer("  ＨＥＬＬＯ，  World!  "),
            "hello world",
        )

    def test_wer_ignores_reference_annotations_and_joins_spelled_acronyms(self) -> None:
        self.assertEqual(normalize_for_wer("[A] X_M_L_ [laugh]"), "xml")

    def test_wer_treats_apostrophes_as_orthography_not_extra_words(self) -> None:
        self.assertEqual(normalize_for_wer("it's we're"), "its were")

    def test_cer_removes_cjk_spacing_and_punctuation(self) -> None:
        self.assertEqual(
            normalize_for_cer("你 好，世 界！"),
            "你好世界",
        )

    def test_cer_preserves_letters_from_all_target_scripts(self) -> None:
        self.assertEqual(
            normalize_for_cer("ABC 東京 カナ"),
            "abc東京カナ",
        )


class ErrorRateAccumulatorTests(unittest.TestCase):
    def test_uses_micro_average_over_reference_units(self) -> None:
        metric = ErrorRateAccumulator("wer")
        metric.add("one two three four", "one two three four")
        metric.add("five", "wrong")

        summary = metric.summary()

        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["reference_units"], 5)
        self.assertAlmostEqual(summary["error_rate"], 0.2)
        self.assertAlmostEqual(summary["macro_error_rate"], 0.5)

    def test_reports_substitutions_deletions_and_insertions(self) -> None:
        metric = ErrorRateAccumulator("wer")
        metric.add("one two", "one too")
        metric.add("three four", "three")
        metric.add("five", "five six")

        summary = metric.summary()

        self.assertEqual(summary["substitutions"], 1)
        self.assertEqual(summary["deletions"], 1)
        self.assertEqual(summary["insertions"], 1)
        self.assertEqual(summary["correct"], 3)
        self.assertEqual(summary["errors"], 3)

    def test_ignores_reference_annotations_but_scores_hypothesis_annotations(self) -> None:
        metric = ErrorRateAccumulator("wer")
        metric.add("[laugh] hello", "[laugh] hello")

        summary = metric.summary()

        self.assertEqual(summary["insertions"], 1)
        self.assertEqual(summary["errors"], 1)

    def test_empty_reference_with_hypothesis_counts_insertions(self) -> None:
        metric = ErrorRateAccumulator("cer")
        metric.add("", "幻覚")

        summary = metric.summary()

        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["insertions"], 2)
        self.assertEqual(summary["reference_units"], 0)
        self.assertEqual(summary["error_rate"], 2.0)

    def test_linear_memory_edit_counts_match_levenshtein_distance(self) -> None:
        sequences = [tuple(bits) for length in range(5) for bits in product("ab", repeat=length)]
        for reference in sequences:
            for hypothesis in sequences:
                with self.subTest(reference=reference, hypothesis=hypothesis):
                    counts = edit_counts(reference, hypothesis)
                    self.assertEqual(counts.errors, levenshtein_distance(reference, hypothesis))
                    self.assertEqual(counts.correct + counts.substitutions + counts.deletions, len(reference))
                    self.assertEqual(counts.correct + counts.substitutions + counts.insertions, len(hypothesis))

    def test_rejects_unknown_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "metric"):
            ErrorRateAccumulator("bleu")


if __name__ == "__main__":
    unittest.main()
