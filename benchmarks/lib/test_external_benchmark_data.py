from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.lib.external_benchmark_data import (
    build_earnings_manifest,
    load_benchmark_plan,
    parse_earnings_nlp,
    verify_asset,
)


class EarningsReferenceTests(unittest.TestCase):
    def test_parses_spoken_tokens_and_excludes_nonlexical_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.nlp"
            path.write_text(
                "token|speaker|ts|endTs|punctuation|case|tags|wer_tags\n"
                "Good|0||||UC|[]|[]\n"
                "<inaudible>|0||||LC|[]|[]\n"
                "morning|0|||.|LC|[]|[]\n"
            )

            parsed = parse_earnings_nlp(path)

        self.assertEqual(parsed.text, "Good morning")
        self.assertEqual(parsed.spoken_tokens, 2)
        self.assertEqual(parsed.excluded_tokens, {"<inaudible>": 1})

    def test_rejects_an_unexpected_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.nlp"
            path.write_text("word|speaker\nhello|0\n")

            with self.assertRaisesRegex(ValueError, "header"):
                parse_earnings_nlp(path)


class EarningsMaterializationTests(unittest.TestCase):
    def test_verifies_asset_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"audio")
            digest = hashlib.sha256(b"audio").hexdigest()

            verify_asset(path, expected_size=5, expected_sha256=digest)

            with self.assertRaisesRegex(ValueError, "sha256"):
                verify_asset(path, expected_size=5, expected_sha256="0" * 64)

    def test_builds_disjoint_dev_and_heldout_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio").mkdir()
            (root / "references").mkdir()
            for case_id in ("dev-one", "test-one"):
                (root / "audio" / f"{case_id}.mp3").write_bytes(b"audio")
                (root / "references" / f"{case_id}.txt").write_text(f"reference {case_id}\n")

            plan = {
                "source": {"dataset": "Earnings-21", "revision": "abc123"},
                "cases": [
                    {"id": "dev-one", "split": "dev", "duration_seconds": 60.0},
                    {"id": "test-one", "split": "heldout", "duration_seconds": 90.0},
                ],
            }

            summary = build_earnings_manifest(plan, root)

            dev = [json.loads(line) for line in (root / "asr-dev.jsonl").read_text().splitlines()]
            heldout = [json.loads(line) for line in (root / "asr-heldout.jsonl").read_text().splitlines()]

        self.assertEqual([case["id"] for case in dev], ["earnings21-dev-one"])
        self.assertEqual([case["id"] for case in heldout], ["earnings21-test-one"])
        self.assertEqual(dev[0]["reference"], "reference dev-one")
        self.assertEqual(summary["splits"]["dev"]["duration_seconds"], 60.0)
        self.assertEqual(summary["splits"]["heldout"]["duration_seconds"], 90.0)
        self.assertTrue(set(summary["splits"]["dev"]["case_ids"]).isdisjoint(
            summary["splits"]["heldout"]["case_ids"]
        ))

    def test_checked_in_plan_has_disjoint_pinned_earnings_cases(self) -> None:
        plan_path = Path("benchmarks/fixtures/subtitle-external/benchmark-plan.json")
        plan = load_benchmark_plan(plan_path)["earnings21"]
        dev = {case["id"] for case in plan["cases"] if case["split"] == "dev"}
        heldout = {case["id"] for case in plan["cases"] if case["split"] == "heldout"}

        self.assertEqual(len(dev), 2)
        self.assertEqual(len(heldout), 2)
        self.assertTrue(dev.isdisjoint(heldout))
        self.assertRegex(plan["source"]["revision"], r"^[0-9a-f]{40}$")
        for case in plan["cases"]:
            self.assertRegex(case["audio"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(case["reference"]["sha256"], r"^[0-9a-f]{64}$")

    def test_checked_in_dev_baseline_is_consistent_and_does_not_open_heldout(self) -> None:
        baseline = json.loads(Path(
            "benchmarks/fixtures/subtitle-external/earnings21-baseline.json"
        ).read_text())
        quality = baseline["quality_per_repeat"]

        self.assertFalse(baseline["heldout_evaluated"])
        self.assertEqual(
            quality["errors"],
            quality["substitutions"] + quality["deletions"] + quality["insertions"],
        )
        self.assertAlmostEqual(quality["wer"], quality["errors"] / quality["reference_units"])
        self.assertTrue(quality["deterministic_across_repeats"])
        self.assertLess(
            baseline["performance"]["case_rtf_range"][0],
            baseline["performance"]["case_rtf_range"][1],
        )


if __name__ == "__main__":
    unittest.main()
