from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from benchmarks.lib.asr_evaluate import load_manifest
from benchmarks.lib.subtitles import load_manifest as load_subtitle_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "benchmarks" / "fixtures" / "subtitle-long"
CASE_ROOT = FIXTURE_ROOT / "ami-en2002b-d-0765-0945"
TIMESTAMP = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")


def parse_timestamp(value: str) -> float:
    match = TIMESTAMP.match(value)
    if match is None:
        raise ValueError(f"invalid SRT timestamp: {value}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


class LongSubtitleGoldenFixtureTests(unittest.TestCase):
    def test_audio_hash_and_reference_counts_match_fixture_metadata(self) -> None:
        fixture = json.loads((CASE_ROOT / "fixture.json").read_text())
        audio = CASE_ROOT / fixture["audio"]["file"]
        segments = json.loads((CASE_ROOT / fixture["references"]["human_segments"]).read_text())[
            "segments"
        ]

        self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), fixture["audio"]["sha256"])
        self.assertEqual(len(segments), fixture["references"]["segment_count"])
        self.assertEqual(fixture["duration_seconds"], 180.0)
        self.assertEqual(fixture["validation"]["valid_for"], ["wer-smoke", "subtitle-structure-smoke"])
        self.assertIn("timestamp-accuracy", fixture["validation"]["not_valid_for"])
        self.assertIn("subtitle-segmentation-quality", fixture["validation"]["not_valid_for"])
        self.assertLess(audio.stat().st_size, 2_000_000)
        reference = (CASE_ROOT / fixture["references"]["normalized_transcript"]).read_text()
        self.assertNotRegex(reference, r"\[[^\]]*\]")
        self.assertNotIn("[D]", (CASE_ROOT / fixture["references"]["srt"]).read_text())

    def test_captured_smoke_baseline_is_internally_consistent(self) -> None:
        baseline = json.loads((FIXTURE_ROOT / "baseline.json").read_text())
        asr = baseline["asr"]

        self.assertEqual(baseline["scope"], "smoke-only")
        self.assertEqual(
            asr["errors_per_run"],
            asr["substitutions_per_run"] + asr["deletions_per_run"] + asr["insertions_per_run"],
        )
        self.assertAlmostEqual(asr["wer"], asr["errors_per_run"] / asr["reference_units_per_run"])
        self.assertEqual(baseline["subtitles"]["non_monotonic_count"], 0)
        self.assertGreater(baseline["subtitles"]["reading_speed_violation_count"], 0)

    def test_srt_cues_have_valid_ranges_within_the_audio(self) -> None:
        fixture = json.loads((CASE_ROOT / "fixture.json").read_text())
        blocks = (CASE_ROOT / fixture["references"]["srt"]).read_text().strip().split("\n\n")

        self.assertEqual(len(blocks), fixture["references"]["segment_count"])
        for index, block in enumerate(blocks, start=1):
            lines = block.splitlines()
            self.assertEqual(lines[0], str(index))
            start_text, end_text = lines[1].split(" --> ")
            start = parse_timestamp(start_text)
            end = parse_timestamp(end_text)
            self.assertGreaterEqual(start, 0)
            self.assertGreater(end, start)
            self.assertLessEqual(end, fixture["duration_seconds"])
            self.assertTrue(" ".join(lines[2:]).strip())

    def test_benchmark_manifests_load_the_single_fixture(self) -> None:
        asr_cases = load_manifest(FIXTURE_ROOT / "asr-eval.jsonl")
        subtitle_cases = load_subtitle_manifest(FIXTURE_ROOT / "subtitles-manifest.json")

        self.assertEqual([case.id for case in asr_cases], ["ami-en2002b-d-0765-0945"])
        self.assertEqual(len(subtitle_cases), 1)
        self.assertEqual(subtitle_cases[0]["name"], "ami-en2002b-d-0765-0945")


if __name__ == "__main__":
    unittest.main()
