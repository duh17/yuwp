from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from benchmarks.lib.batch_compare import multipart_request, run_yuwp_server, run_yuwp_server_with_duration
from benchmarks.lib.asr_evaluate import (
    EvaluationCase,
    EvaluationMeasurement,
    load_manifest,
    run_yuwp,
    summarize,
)


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


class MultipartRequestTests(unittest.TestCase):
    def test_omits_stale_public_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "clip.wav"
            audio.write_bytes(b"audio")
            response = MagicMock()
            response.__enter__.return_value.read.return_value = b"{}"

            with patch("benchmarks.lib.batch_compare.urlopen", return_value=response) as open_url:
                multipart_request("http://127.0.0.1/transcriptions", audio, language="English")

        body = open_url.call_args.args[0].data
        self.assertNotIn(b'name="model"', body)
        self.assertIn(b'name="response_format"', body)
        self.assertIn(b'name="language"', body)
        self.assertIn(b"English", body)
        self.assertIn(b'name="file"', body)


class YuwpEvaluationTests(unittest.TestCase):
    def test_reads_decoded_duration_from_yuwp_server_response(self) -> None:
        server = SimpleNamespace(port=4321)
        with (
            patch(
                "benchmarks.lib.batch_compare.multipart_request",
                return_value=b'{"text":"hello","duration":3953.423625}',
            ) as request,
            patch("benchmarks.lib.batch_compare.time.perf_counter", side_effect=[10.0, 12.5]),
        ):
            result = run_yuwp_server_with_duration(
                Path("clip.mp3"),
                server,
                language="English",
            )

        self.assertEqual(result, ("hello", 2.5, 3953.423625))
        request.assert_called_once_with(
            "http://127.0.0.1:4321/v1/audio/transcriptions",
            Path("clip.mp3"),
            language="English",
        )

    def test_pins_the_manifest_language_for_each_case(self) -> None:
        case = EvaluationCase("clip", Path("clip.wav"), "en", "wer", "hello")
        server = SimpleNamespace(process=object(), startup_s=0.1, log_path="/tmp/server.log")

        with (
            patch(
                "benchmarks.lib.asr_evaluate.start_yuwp_server",
                return_value=server,
            ) as start_server,
            patch(
                "benchmarks.lib.asr_evaluate.run_yuwp_server_with_duration",
                return_value=("hello", 0.2, 12.5),
            ) as run,
            patch(
                "benchmarks.lib.asr_evaluate.audio_duration_seconds",
                side_effect=AssertionError("Yuwp evaluation must use server duration"),
            ),
            patch("benchmarks.lib.asr_evaluate.stop_process"),
        ):
            rows, metadata = run_yuwp(
                "yuwp/test",
                Path("model"),
                [case],
                1,
                None,
                batch_chunking="energy",
            )

        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list,
            [
                call(case.audio, server, language="English"),
                call(case.audio, server, language="English"),
            ],
        )
        self.assertEqual(rows[0].errors, 0)
        self.assertEqual(rows[0].correct, 1)
        self.assertEqual(rows[0].duration_s, 12.5)
        self.assertAlmostEqual(rows[0].rtf, 0.2 / 12.5)
        self.assertEqual(metadata["chunking"], "energy")
        self.assertEqual(metadata["duration_source"], "server_decoded")
        start_server.assert_called_once_with(Path("model"), batch_chunking="energy")

    def test_batch_compare_wrapper_keeps_container_duration_behavior(self) -> None:
        server = SimpleNamespace(port=1234)
        with (
            patch(
                "benchmarks.lib.batch_compare._run_yuwp_server_request",
                return_value=({"text": "hello", "duration": 12.5}, 2.0),
            ),
            patch("benchmarks.lib.batch_compare.audio_duration_seconds", return_value=20.0),
        ):
            result = run_yuwp_server(Path("clip.mp3"), server)

        self.assertEqual(result, ("hello", 2.0, 10.0))


class SummaryTests(unittest.TestCase):
    def test_reports_micro_accuracy_and_standard_rtf_by_language(self) -> None:
        rows = [
            EvaluationMeasurement(
                "model", "a", "en", "wer", 10.0, 1.0, "one two", "one two", 0, 2,
                correct=2,
            ),
            EvaluationMeasurement(
                "model", "b", "en", "wer", 20.0, 4.0, "three", "wrong", 1, 1,
                substitutions=1,
            ),
        ]

        result = summarize(rows)
        english = result["model"]["en"]

        self.assertEqual(english["errors"], 1)
        self.assertEqual(english["reference_units"], 3)
        self.assertEqual(english["correct"], 2)
        self.assertEqual(english["substitutions"], 1)
        self.assertEqual(english["deletions"], 0)
        self.assertEqual(english["insertions"], 0)
        self.assertAlmostEqual(english["error_rate"], 1 / 3)
        self.assertAlmostEqual(english["corpus_rtf"], 5 / 30)
        self.assertAlmostEqual(english["median_utterance_rtf"], 0.15)


if __name__ == "__main__":
    unittest.main()
