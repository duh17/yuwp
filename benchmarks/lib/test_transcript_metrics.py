from __future__ import annotations

import unittest

from benchmarks.lib.transcript_metrics import (
    ErrorRateAccumulator,
    normalize_for_cer,
    normalize_for_wer,
)


class TranscriptNormalizationTests(unittest.TestCase):
    def test_wer_normalizes_unicode_case_punctuation_and_whitespace(self) -> None:
        self.assertEqual(
            normalize_for_wer("  ＨＥＬＬＯ，  World!  "),
            "hello world",
        )

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

    def test_empty_reference_with_hypothesis_counts_insertions(self) -> None:
        metric = ErrorRateAccumulator("cer")
        metric.add("", "幻覚")

        summary = metric.summary()

        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["reference_units"], 0)
        self.assertEqual(summary["error_rate"], 2.0)

    def test_rejects_unknown_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "metric"):
            ErrorRateAccumulator("bleu")


if __name__ == "__main__":
    unittest.main()
