from __future__ import annotations

import unittest

from benchmarks.lib.stream_quality import summarize_chunk_timings


class SummarizeChunkTimingsTests(unittest.TestCase):
    def test_aggregates_only_speech_active_chunks(self) -> None:
        reports = [
            {
                "streaming": {
                    "chunks": [
                        {
                            "speechActive": False,
                            "timing": {
                                "totalMs": 1.0,
                                "encodeMs": 0.0,
                                "prefillMs": 0.0,
                                "decodeMs": 0.0,
                            },
                        },
                        {
                            "speechActive": True,
                            "timing": {
                                "totalMs": 10.0,
                                "encodeMs": 2.0,
                                "prefillMs": 3.0,
                                "decodeMs": 5.0,
                            },
                        },
                        {
                            "speechActive": True,
                            "timing": {
                                "totalMs": 30.0,
                                "encodeMs": 4.0,
                                "prefillMs": 9.0,
                                "decodeMs": 17.0,
                            },
                        },
                    ]
                }
            }
        ]

        summary = summarize_chunk_timings(reports)

        self.assertEqual(summary["speech_active_chunk_count"], 2)
        self.assertEqual(summary["total_ms"]["mean"], 20.0)
        self.assertEqual(summary["total_ms"]["median"], 20.0)
        self.assertEqual(summary["total_ms"]["p95"], 29.0)
        self.assertEqual(summary["total_ms"]["max"], 30.0)
        self.assertEqual(summary["total_ms"]["mad"], 10.0)
        self.assertEqual(summary["decode_ms"]["mean"], 11.0)

    def test_returns_empty_metrics_when_no_speech_chunks_exist(self) -> None:
        reports = [{"streaming": {"chunks": []}}]

        summary = summarize_chunk_timings(reports)

        self.assertEqual(summary["speech_active_chunk_count"], 0)
        self.assertIsNone(summary["total_ms"]["mean"])
        self.assertIsNone(summary["total_ms"]["p95"])


if __name__ == "__main__":
    unittest.main()
