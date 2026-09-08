from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from benchmarks.lib.asr_evaluate import EvaluationCase
from benchmarks.lib.english_parity import (
    IPCClient,
    analyze_stream,
    execute,
    load_dev_cases,
    main,
    ordered_blocks,
    run_stream,
    run_trial,
    summarize_trials,
)


class ReproductionSafetyTests(unittest.TestCase):
    def run_audio_staging(self, *, existing_root, checksum_exit, block_index=0):
        readme = Path(__file__).resolve().parents[1] / "fixtures/english-parity/reproduction/README.md"
        snippet = readme.read_text().split("```bash\n")[block_index + 1].split("```", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            calls = Path(directory) / "calls"
            if existing_root:
                root.mkdir()
                (root / "sentinel").write_text("preserve")
                for child in ("data/audio", "models", "FluidAudio", "adapter", "adapter-readiness-repair", "repair-assets"):
                    (root / child).mkdir(parents=True)
            snippet = snippet.replace("/tmp/yuwp-english-parity", str(root))
            mocks = '''
                curl() { printf 'download\\n' >> "$CALLS"; }
                shasum() { printf 'checksum\\n' >> "$CALLS"; return "$CHECKSUM_EXIT"; }
                tar() { printf 'extract\\n' >> "$CALLS"; }
                uv() { printf 'uv\\n' >> "$CALLS"; }
                git() { printf 'git\\n' >> "$CALLS"; }
                cp() { printf 'copy\\n' >> "$CALLS"; }
                patch() { printf 'patch\\n' >> "$CALLS"; }
                bash() { printf 'build\\n' >> "$CALLS"; }
            '''
            result = subprocess.run(["/bin/bash", "-c", mocks + snippet], capture_output=True,
                                    text=True, timeout=3, env={**os.environ, "CALLS": str(calls),
                                                             "CHECKSUM_EXIT": str(checksum_exit)})
            if existing_root:
                self.assertEqual((root / "sentinel").read_text(), "preserve")
            return result.returncode, calls.read_text().splitlines() if calls.exists() else []

    def test_existing_staging_root_stops_before_download_or_extraction(self):
        code, calls = self.run_audio_staging(existing_root=True, checksum_exit=0)
        self.assertNotEqual(code, 0)
        self.assertEqual(calls, [])

    def test_later_staging_blocks_reject_existing_destinations(self):
        for index in range(1, 5):
            with self.subTest(block=index):
                code, calls = self.run_audio_staging(existing_root=True, checksum_exit=0, block_index=index)
                self.assertNotEqual(code, 0)
                self.assertEqual(calls, [])

    def test_checksum_failure_stops_before_extraction(self):
        code, calls = self.run_audio_staging(existing_root=False, checksum_exit=1)
        self.assertNotEqual(code, 0)
        self.assertEqual(calls, ["download", "checksum"])


class CLIExecutionTests(unittest.TestCase):
    def test_explicit_under_load_runs_diagnostics_without_claiming_clearance(self):
        with tempfile.TemporaryDirectory() as directory:
            commands = Path(directory) / "commands.json"
            commands.write_text(json.dumps({name: ["fake-peer"] for name in ("Qwen", "Nemotron", "Parakeet")}))
            argv = ["english-parity", "--commands", str(commands), "--output", str(Path(directory) / "run.jsonl"),
                    "--execute", "--load-status", "diagnostic-under-load"]
            with patch.object(sys, "argv", argv), patch("benchmarks.lib.english_parity.load_dev_cases", return_value=[]), \
                    patch("benchmarks.lib.english_parity.execute", new_callable=AsyncMock, return_value=0) as run:
                self.assertEqual(main(), 0)
            args, _, _, receipt = run.call_args.args
            self.assertEqual(args.load_status, "diagnostic-under-load")
            self.assertTrue(receipt["no_load_control"])
            self.assertEqual(receipt["label"], "NON-ACCEPTANCE")


class Clock:
    def __init__(self):
        self.value = 10.0
        self.waiters = []

    def now(self):
        return self.value

    async def sleep_until(self, deadline):
        while self.value < deadline:
            future = asyncio.get_running_loop().create_future()
            self.waiters.append((deadline, future))
            await future

    def advance(self, value):
        self.value = value
        for deadline, future in self.waiters:
            if deadline <= value and not future.done():
                future.set_result(None)


def visible(text, at, *, final=False, committed=""):
    return {
        "command": "stop" if final else "feed",
        "reply_end": at,
        "write_start": at - 0.01,
        "response": {"text": text, "committed_text": committed,
                     "active_text": text if not committed else "", "is_final": final},
    }


class MetricTests(unittest.TestCase):
    def analyze(self, events, reference="one two three four"):
        return analyze_stream(events, reference, 0.0, 1.0)

    def test_nonempty_wrong_annotations_are_not_useful(self):
        events = [visible("", .1), visible("[one two three] wrong", .2),
                  visible("wrong", .3), visible("one two three", .4),
                  visible("one two three four", 1.2, final=True)]
        result = self.analyze(events)
        self.assertEqual(result["first_nonempty_s"], .2)
        self.assertEqual(result["first_useful_s"], .2)
        self.assertEqual(result["stable_useful_s"], .4)
        self.assertAlmostEqual(result["stop_to_final_s"], .2)
        self.assertAlmostEqual(result["stop_request_rtt_s"], .01)
        self.assertGreater(result["rollback_words"], 0)
        self.assertEqual(result["scores"]["errors"], 0)

    def test_annotation_hallucination_not_removed(self):
        result = self.analyze([visible("[noise] one two three", .2),
                               visible("one two three", 1.1, final=True)])
        self.assertIsNone(result["first_useful_s"])
        self.assertIsNone(result["stable_useful_s"])

    def test_empty_success_has_deletions_but_missing_useful(self):
        result = self.analyze([visible("", .2), visible("", 1.1, final=True)])
        self.assertEqual(result["scores"]["deletions"], 4)
        self.assertIsNone(result["first_nonempty_s"])
        self.assertIsNone(result["first_useful_s"])

    def test_final_only_visibility_is_nonempty_but_not_useful(self):
        result = self.analyze([visible("one two three", 1.1, final=True)])
        self.assertEqual(result["first_nonempty_s"], 1.1)
        self.assertIsNone(result["first_useful_s"])
        self.assertIsNone(result["stable_useful_s"])

    def test_cutoff_and_silence_have_no_lexical_scores(self):
        events = [visible("invented", .2), visible("", 1.1, final=True)]
        for kind in ("cutoff", "silence-only", "noise-only"):
            result = analyze_stream(events, "", 0, 1, kind=kind)
            self.assertIsNone(result["scores"])
            self.assertIsNone(result["tail_errors"])
            if kind != "cutoff":
                self.assertEqual(result["hallucination_events"], 1)
                self.assertTrue(result["hallucination_complete"])

    def test_paired_quality_deltas_and_missing_lexical_endpoints(self):
        metric = self.analyze([visible("one two three four", .2),
                               visible("one two three four", 1.1, final=True)])
        rows = [{"system": system, "case_id": "a", "status": "ok", "metrics": metric}
                for system in ("Qwen", "Nemotron", "Parakeet")]
        summary = summarize_trials(rows)
        self.assertEqual(summary["paired"]["Nemotron"]["csdi_delta"]["correct"], 0)
        self.assertEqual(summary["paired"]["Nemotron"]["micro_wer_delta"], 0)
        self.assertEqual(summary["paired"]["Nemotron"]["invariant_deltas"]["tail_errors"]["delta"], 0)
        self.assertEqual(summary["systems"]["Qwen"]["aggregate_rollback_rate"], 0)
        self.assertEqual(summary["paired"]["Nemotron"]["latencies"]["first_useful_s"]["p95_delta_s"], 0)

    def test_missing_final_never_scores_empty_fallback(self):
        result = self.analyze([visible("one two three", .2)])
        self.assertIsNone(result["scores"])
        self.assertIsNone(result["stable_useful_s"])
        self.assertIn("missing-final", result["violations"])

    def test_commit_change_and_space_join_are_violations(self):
        events = [visible("one", .2, committed="one"),
                  visible("two", .3, committed="two"),
                  visible("two", 1.1, final=True, committed="two")]
        self.assertIn("committed-prefix-changed", self.analyze(events)["violations"])
        events[0]["response"]["active_text"] = "more"
        self.assertIn("text-reconstruction", self.analyze(events)["violations"])

    def test_missing_pairs_are_counted_and_no_confidence(self):
        rows = [{"system": "Qwen", "case_id": "a", "status": "failed", "metrics": {}}]
        summary = summarize_trials(rows)
        self.assertEqual(summary["label"], "NON-ACCEPTANCE")
        self.assertEqual(summary["systems"]["Qwen"]["latencies"]["first_useful_s"]["missing"], 1)
        self.assertEqual(summary["paired"]["Nemotron"]["missing_pairs"], 1)
        self.assertNotIn("confidence", json.dumps(summary))


class ManifestTests(unittest.TestCase):
    def test_dev_rejection_and_diagnostic_filter_before_audio_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.wav").touch()
            dev = {"id": "a", "audio": "clip.wav", "language": "en", "metric": "wer",
                   "reference": "one", "split": "dev", "kind": "short"}
            heldout = dict(dev, id="heldout", split="heldout", audio="do-not-open.wav")
            manifest = root / "dev.jsonl"
            diagnostics = root / "diagnostics.jsonl"
            manifest.write_text(json.dumps(dev) + "\n")
            diagnostics.write_text(json.dumps(heldout) + "\n" + json.dumps(dict(dev, id="b", kind="cutoff")))
            cases = load_dev_cases(manifest, diagnostics)
            self.assertEqual([case.id for case, _ in cases], ["a", "b"])
            manifest.write_text(json.dumps(heldout))
            with self.assertRaisesRegex(ValueError, "non-dev"):
                load_dev_cases(manifest, diagnostics)

    def test_fixed_permutations_and_order_are_reproducible(self):
        a = ordered_blocks(["a", "b", "c"])
        self.assertEqual(a, ordered_blocks(["c", "a", "b"]))
        self.assertEqual({x for _, systems in a for x in systems}, {"Qwen", "Nemotron", "Parakeet"})


class ClockTests(unittest.IsolatedAsyncioTestCase):
    async def test_end_deadlines_backpressure_and_stop_intent(self):
        clock = Clock()
        events = []
        first_reply = asyncio.Event()
        calls = []

        class Client:
            async def request(self, command, binary=b"", **kwargs):
                calls.append((command, clock.now(), len(binary)))
                event = {"command": command, "write_start": clock.now()}
                if command == "feed" and len(calls) == 2:
                    await first_reply.wait()
                response = {"session_id": "s"} if command == "create" else {
                    "text": "one two three", "committed_text": "", "active_text": "one two three",
                    "is_final": command == "stop"}
                event.update(response=response, reply_end=clock.now(), write_end=event["write_start"])
                events.append(event)
                return response

        task = asyncio.create_task(run_stream(Client(), b"\0" * 8000, events, deadline=100, clock=clock))
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual([x[0] for x in calls], ["create"])
        clock.advance(10.1)
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual(calls[-1], ("feed", 10.1, 3200))
        clock.advance(10.25)
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertTrue(any(e.get("type") == "stop-intent" and e["scheduled"] == 10.25 for e in events))
        self.assertEqual(len(calls), 2)
        clock.advance(10.5)
        first_reply.set()
        result = await task
        self.assertEqual(result["t0"], 10.0)
        self.assertEqual(result["t_stop"], 10.25)
        self.assertEqual([x[2] for x in calls if x[0] == "feed"], [3200, 3200, 1600])
        self.assertEqual(calls[-1][:2], ("stop", 10.5))
        self.assertEqual(result["end_backlog_s"], .25)


# An actual binary stdio peer, not a model. No wall sleeps or external dependencies.
FAKE = r'''
import json, struct, sys
mode = sys.argv[1]
session = 0
while True:
    header = sys.stdin.buffer.read(8)
    if not header: break
    n, b = struct.unpack(">II", header)
    req = json.loads(sys.stdin.buffer.read(n))
    sys.stdin.buffer.read(b)
    if req['command'] == 'create': session += 1
    out = {"id": req["id"], "ok": True, "status": "ready", "session_id": "s",
           "sample_rate": 16000, "final_accuracy_pass_enabled": True}
    if mode in ('stream', 'missing-final', 'measured-missing-final', 'omitted-echo'):
        out['session_id'] = str(session)
        out.update(text='one two three', committed_text='', active_text='one two three',
                   is_final=req['command'] == 'stop' and mode != 'missing-final'
                   and not (mode == 'measured-missing-final' and session == 2))
    if mode == 'omitted-echo' and req['command'] in ('feed', 'stop'):
        del out['session_id']
    if mode == 'stall': sys.stdin.buffer.read()
    if mode == "wrong-id": out["id"] += 1
    if mode == "wrong-session": out["session_id"] = "other"
    if mode == "error": out.update(ok=False, error="deliberate")
    if mode == "malformed":
        sys.stdout.buffer.write(struct.pack(">II", 1, 0) + b"{")
    elif mode == "oversize":
        sys.stdout.buffer.write(struct.pack(">II", 1048577, 0))
    elif mode == "eof": break
    else:
        data = json.dumps(out).encode()
        sys.stdout.buffer.write(struct.pack(">II", len(data), 0) + data)
    sys.stdout.buffer.flush()
'''


class IPCTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_fake_process_framing_and_cleanup(self):
        events = []
        client = await IPCClient.spawn([sys.executable, "-c", FAKE, "ok"], events)
        try:
            response = await client.request("info", deadline=client.clock.now() + 2)
            self.assertEqual(response["status"], "ready")
            self.assertIn("raw_metadata_base64", events[0])
        finally:
            await client.close()
        self.assertIsNotNone(client.process.returncode)

    async def test_malformed_wrong_ids_errors_and_eof_are_not_success(self):
        for mode in ["wrong-id", "wrong-session", "error", "malformed", "oversize", "eof"]:
            with self.subTest(mode=mode):
                events = []
                client = await IPCClient.spawn([sys.executable, "-c", FAKE, mode], events)
                try:
                    with self.assertRaises((ValueError, RuntimeError, asyncio.IncompleteReadError)):
                        await client.request("feed", session_id="s", deadline=client.clock.now() + 2)
                    self.assertIn("error", events[0])
                finally:
                    await client.close()

    async def test_fresh_process_warmup_then_clean_measured_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / "tiny.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0")
            case = EvaluationCase("a", wav, "en", "wer", "one two three")
            metadata = {"split": "dev", "kind": "short", "samples": 1}
            for mode in ("stream", "omitted-echo", "missing-final", "measured-missing-final"):
                row = await run_trial("Qwen", [sys.executable, "-c", FAKE, mode], case, metadata,
                                      (case, metadata), deadline=asyncio.get_running_loop().time() + 3,
                                      stderr_path=root / f"{mode}.stderr", load_status="uncontrolled")
                if mode in ("missing-final", "measured-missing-final"):
                    self.assertEqual(row["status"], "failed")
                    self.assertIn("missing final", row["error"])
                    if mode == "missing-final":
                        self.assertEqual(row["metrics"], {})
                    else:
                        self.assertIsNone(row["metrics"]["scores"])
                        self.assertIn("missing-final", row["metrics"]["violations"])
                else:
                    self.assertEqual(row["status"], "ok", row.get("error"))
                    self.assertEqual(row["metrics"]["scores"]["errors"], 0)
                    creates = [e for e in row["events"] if e.get("command") == "create"]
                    self.assertEqual([e["phase"] for e in creates], ["warmup", "measured"])
                    self.assertNotEqual(creates[0]["response"]["session_id"], creates[1]["response"]["session_id"])
                self.assertTrue(row["no_load_control"])
                self.assertEqual(row["events"][-1]["type"], "process-exit")

    async def test_global_budget_retains_every_missing_trial_without_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(max_seconds=2, case_limit=1, output=root / "result.jsonl",
                                   load_status="uncontrolled")
            case = EvaluationCase("a", root / "not-read.wav", "en", "wer", "one")
            cases = [(case, {"kind": "short", "diagnostic": False})]
            with redirect_stdout(StringIO()):
                code = await execute(args, cases, {}, {"label": "NON-ACCEPTANCE"})
            self.assertEqual(code, 1)
            records = [json.loads(line) for line in args.output.read_text().splitlines()]
            self.assertEqual([r["status"] for r in records if r["type"] == "trial"], ["not-run"] * 3)
            self.assertFalse(records[-1]["complete_dev_pass"])
            self.assertEqual(records[-1]["scope"], "transport-smoke")
            self.assertEqual(records[-1]["paired"]["Nemotron"]["missing_pairs"], 1)
            with self.assertRaises(FileExistsError):
                await execute(args, cases, {}, {"label": "NON-ACCEPTANCE"})

    async def test_unresponsive_peer_times_out_and_is_cleaned(self):
        events = []
        client = await IPCClient.spawn([sys.executable, "-c", FAKE, "stall"], events)
        try:
            with self.assertRaises(TimeoutError):
                await client.request("info", deadline=client.clock.now() + .05)
            self.assertIn("TimeoutError", events[0]["error"])
        finally:
            await client.close()
        self.assertIsNotNone(client.process.returncode)

    async def test_expired_deadline_captured(self):
        events = []
        client = await IPCClient.spawn([sys.executable, "-c", FAKE, "ok"], events)
        try:
            with self.assertRaises(TimeoutError):
                await client.request("info", deadline=client.clock.now() - 1)
            self.assertIn("error", events[0])
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
