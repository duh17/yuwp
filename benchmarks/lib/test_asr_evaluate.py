from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.lib.asr_evaluate import EvaluationMeasurement, load_manifest, summarize


class ManifestTests(unittest.TestCase):
    def test_loads_jsonl_and_resolves_relative_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.wav").touch()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "clip",
                        "audio": "clip.wav",
                        "language": "en",
                        "metric": "wer",
                        "reference": "hello world",
                    }
                )
                + "\n"
            )

            cases = load_manifest(manifest)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].audio, (root / "clip.wav").resolve())
            self.assertEqual(cases[0].metric, "wer")

    def test_rejects_invalid_language_metric_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "audio": "missing.wav",
                        "language": "en",
                        "metric": "cer",
                        "reference": "text",
                    }
                )
                + "\n"
            )

            with self.assertRaisesRegex(ValueError, "English"):
                load_manifest(manifest)


class SummaryTests(unittest.TestCase):
    def test_reports_micro_accuracy_and_standard_rtf_by_language(self) -> None:
        rows = [
            EvaluationMeasurement("model", "a", "en", "wer", 10.0, 1.0, "one two", "one two", 0, 2),
            EvaluationMeasurement("model", "b", "en", "wer", 20.0, 4.0, "three", "wrong", 1, 1),
        ]

        result = summarize(rows)
        english = result["model"]["en"]

        self.assertEqual(english["errors"], 1)
        self.assertEqual(english["reference_units"], 3)
        self.assertAlmostEqual(english["error_rate"], 1 / 3)
        self.assertAlmostEqual(english["corpus_rtf"], 5 / 30)
        self.assertAlmostEqual(english["median_utterance_rtf"], 0.15)


if __name__ == "__main__":
    unittest.main()
