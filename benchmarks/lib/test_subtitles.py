from __future__ import annotations

import unittest

from benchmarks.lib.subtitles import analyze_subtitles


class SubtitleAnalysisTests(unittest.TestCase):
    def test_reports_standard_rtf_and_reference_free_readability_violations(self) -> None:
        items = [
            {"text": "x" * 43, "start": 0.0, "end": 1.0},
            {"text": "one\ntwo\nthree", "start": 1.05, "end": 8.10},
        ]

        result = analyze_subtitles(items, wall_seconds=2.0, gap_warn_sec=2.0)

        self.assertAlmostEqual(result["rtf"], 2.0 / 8.1)
        self.assertAlmostEqual(result["speed_multiplier"], 8.1 / 2.0)
        self.assertEqual(result["characters_per_line_violation_count"], 1)
        self.assertEqual(result["line_count_violation_count"], 1)
        self.assertEqual(result["reading_speed_violation_count"], 1)
        self.assertEqual(result["max_duration_violation_count"], 1)
        self.assertEqual(result["min_gap_violation_count"], 1)

    def test_reports_short_single_word_cues(self) -> None:
        result = analyze_subtitles(
            [{"text": "hello", "start": 0.0, "end": 0.5}],
            wall_seconds=1.0,
            gap_warn_sec=2.0,
        )

        self.assertEqual(result["min_duration_violation_count"], 1)
        self.assertEqual(result["single_word_cue_count"], 1)


if __name__ == "__main__":
    unittest.main()
