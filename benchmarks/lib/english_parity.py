#!/usr/bin/env -S uv run --python 3.14 --script
"""Bounded dev-only ASRIPC diagnostics; never an acceptance/parity gate.

Scoring and manifest semantics come from asr_evaluate/transcript_metrics.
Framing follows scripts/benchmark_asr_transport.py, with async write/read
budgets so backpressure cannot bypass a request or global deadline.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import itertools
import json
import os
import platform
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from .asr_evaluate import EvaluationCase, load_manifest, percentile, score
from .transcript_metrics import edit_counts, normalize_for_wer

LABEL = "NON-ACCEPTANCE"
SYSTEMS = ("Qwen", "Nemotron", "Parakeet")
UNSCORED = {"cutoff", "silence-only", "noise-only"}
ENDPOINTS = (
    "first_nonempty_s", "first_useful_s", "stable_useful_s", "first_commit_s",
    "stop_to_final_s", "stop_request_rtt_s", "startup_inclusive_useful_s",
    "spawn_to_ready_s", "create_s", "final_pass_runtime_s", "end_backlog_s", "packet_lateness_max_s",
    "packet_lateness_p50_s", "packet_lateness_p95_s",
)


class Clock:
    now = staticmethod(time.monotonic)

    async def sleep_until(self, deadline: float) -> None:
        await asyncio.sleep(max(0, deadline - self.now()))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_dev_cases(dev: Path, diagnostics: Path) -> list[tuple[EvaluationCase, dict]]:
    """Filter diagnostic metadata BEFORE invoking the existing audio-path loader."""
    selected = []
    seen = set()
    for path, diagnostic, maximum in ((dev, False, 12), (diagnostics, True, 15)):
        rows = []
        for raw in path.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if diagnostic and row.get("split") == "heldout":
                continue
            if row.get("split") != "dev":
                raise ValueError(f"non-dev record rejected in {path}")
            if row.get("language") != "en" or row.get("metric") != "wer":
                raise ValueError("English WER records required")
            if row["id"] in seen:
                raise ValueError(f"duplicate case: {row['id']}")
            seen.add(row["id"])
            row = dict(row, diagnostic=diagnostic)
            audio = Path(row["audio"]).expanduser()
            row["audio"] = str(audio if audio.is_absolute() else path.resolve().parent / audio)
            rows.append(row)
        if len(rows) > maximum:
            raise ValueError(f"dev budget exceeded: {len(rows)} > {maximum}")
        if not rows:
            continue
        # Keep one authoritative manifest parser without touching heldout audio.
        with tempfile.TemporaryDirectory(prefix="english-parity-manifest-") as directory:
            filtered = Path(directory) / "dev.jsonl"
            filtered.write_text("\n".join(json.dumps(row) for row in rows))
            selected.extend(zip(load_manifest(filtered), rows))
    if not selected:
        raise ValueError("no dev cases")
    return selected


def ordered_blocks(case_ids: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    phase, repeat = "dev", 1
    permutations = sorted(itertools.permutations(SYSTEMS))
    ordered = sorted(case_ids, key=lambda cid: sha256(
        f"english-parity-order-v1:{phase}:{repeat}:{cid}".encode()))
    return [(cid, permutations[int.from_bytes(hashlib.sha256(
        f"english-parity-system-v1:{phase}:{repeat}:{cid}".encode()).digest()[:8], "big") % 6])
        for cid in ordered]


def read_pcm(case: EvaluationCase, metadata: dict) -> bytes:
    data = case.audio.read_bytes()
    if metadata.get("audio_sha256") and sha256(data) != metadata["audio_sha256"]:
        raise ValueError("audio SHA-256 mismatch")
    with wave.open(str(case.audio), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("frozen mono s16le 16000 Hz WAV required; no resampling")
        pcm = audio.readframes(audio.getnframes())
        if len(pcm) != audio.getnframes() * 2:
            raise ValueError("truncated WAV")
    if not pcm or len(pcm) > 45 * 16000 * 2:
        raise ValueError("audio outside (0, 45s] cap")
    if "samples" in metadata and len(pcm) // 2 != metadata["samples"]:
        raise ValueError("sample count mismatch")
    if metadata.get("pcm_sha256") and sha256(pcm) != metadata["pcm_sha256"]:
        raise ValueError("PCM SHA-256 mismatch")
    return pcm


class IPCClient:
    def __init__(self, process, events: list[dict], clock=None):
        self.process, self.events = process, events
        self.clock = clock or Clock()
        self.request_id = 0
        self.phase = "readiness"
        self.inflight = False

    @classmethod
    async def spawn(cls, command: list[str], events: list[dict], stderr=None):
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=stderr if stderr is not None else asyncio.subprocess.DEVNULL,
            limit=1_048_576,
        )
        return cls(process, events)

    async def request(self, command: str, binary: bytes = b"", *, deadline: float,
                      session_id: str | None = None, context: dict | None = None) -> dict:
        if self.inflight:
            raise RuntimeError("concurrent IPC operation forbidden")
        self.inflight = True
        self.request_id += 1
        request: dict[str, Any] = {"id": self.request_id, "command": command}
        if session_id is not None:
            request["session_id"] = session_id
        if command == "create":
            request["language"] = "English"
        metadata = json.dumps(request, separators=(",", ":")).encode()
        event = dict(context or {}, command=command, phase=self.phase, request=request,
                     request_binary_base64=base64.b64encode(binary).decode(),
                     write_start=self.clock.now())
        self.events.append(event)
        try:
            budget = min(deadline - self.clock.now(), 180 if command == "info" else 30)
            if budget <= 0:
                raise TimeoutError("request/global deadline exhausted")
            async with asyncio.timeout(budget):
                self.process.stdin.write(struct.pack(">II", len(metadata), len(binary)) + metadata + binary)
                await self.process.stdin.drain()
                event["write_end"] = self.clock.now()
                header = await self.process.stdout.readexactly(8)
                event["raw_header_base64"] = base64.b64encode(header).decode()
                n, b = struct.unpack(">II", header)
                if n > 1_048_576 or b > 16_777_216:
                    raise ValueError("oversized IPC frame")
                raw = await self.process.stdout.readexactly(n)
                event["raw_metadata_base64"] = base64.b64encode(raw).decode()
                binary_reply = await self.process.stdout.readexactly(b)
                event["raw_binary_base64"] = base64.b64encode(binary_reply).decode()
                event["read_end"] = self.clock.now()
                response = json.loads(raw)
                event["reply_end"] = self.clock.now()
                event["response"] = response
                if b:
                    raise ValueError("unexpected response binary")
                if not isinstance(response, dict):
                    raise ValueError("IPC response is not an object")
                if type(response.get("id")) is not int or response["id"] != request["id"]:
                    raise ValueError("IPC response id mismatch")
                if response.get("ok") is not True or response.get("error") is not None:
                    raise RuntimeError(f"IPC error: {response.get('error', 'ok not true')}")
                if session_id is not None and "session_id" in response and response["session_id"] != session_id:
                    raise ValueError("IPC session echo mismatch")
                return response
        except BaseException as error:
            event["error"] = f"{type(error).__name__}: {error}"
            event["error_time"] = self.clock.now()
            if isinstance(error, asyncio.IncompleteReadError):
                event["partial_base64"] = base64.b64encode(error.partial).decode()
            raise
        finally:
            self.inflight = False

    async def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), 1)
            except TimeoutError:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                await asyncio.wait_for(self.process.wait(), 1)
        self.events.append({"type": "process-exit", "at": self.clock.now(),
                            "returncode": self.process.returncode})


def validate_visible(response: dict, *, final: bool) -> None:
    for key in ("text", "committed_text", "active_text"):
        if not isinstance(response.get(key), str):
            raise ValueError(f"missing/invalid {key}")
    if type(response.get("is_final")) is not bool:
        raise ValueError("missing/invalid is_final")
    if response["is_final"] != final:
        raise ValueError("missing final or unexpected feed final")
    if "batch_corrected" in response and type(response["batch_corrected"]) is not bool:
        raise ValueError("invalid batch_corrected")
    if "update_kind" in response and not isinstance(response["update_kind"], str):
        raise ValueError("invalid update_kind")


async def run_stream(client, pcm: bytes, events: list[dict], *, deadline: float,
                     clock=None, timing: dict | None = None) -> dict:
    """Independent release observer logs wake lateness before the serial feed slot."""
    clock = clock or Clock()
    result = timing if timing is not None else {}
    create_start = clock.now()
    created = await client.request("create", deadline=deadline)
    session_id = created.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("missing session_id")
    t0 = clock.now()
    duration = len(pcm) / 32000
    if not pcm or len(pcm) % 2 or duration > 45:
        raise ValueError("invalid bounded PCM")
    t_stop = t0 + duration
    deadline = min(deadline, t_stop + 60)
    result.update(t0=t0, t_stop=t_stop, create_s=t0 - create_start, session_id=session_id)
    queue: asyncio.Queue = asyncio.Queue()
    packets = [(end, pcm[start:end]) for start in range(0, len(pcm), 3200)
               for end in [min(start + 3200, len(pcm))]]

    async def release():
        for end, chunk in packets:
            scheduled = t0 + end / 32000
            await clock.sleep_until(scheduled)
            wake = clock.now()
            event = {"type": "packet-release", "scheduled": scheduled, "wake": wake,
                     "client_wake_lateness_s": max(0, wake - scheduled),
                     "cumulative_samples": end // 2, "samples": len(chunk) // 2}
            events.append(event)
            queue.put_nowait((chunk, event))
        events.append({"type": "stop-intent", "scheduled": t_stop, "observed": clock.now()})

    producer = asyncio.create_task(release())
    lateness = []
    try:
        for _ in packets:
            remaining = deadline - clock.now()
            if remaining <= 0:
                raise TimeoutError("stream/global deadline exhausted")
            chunk, release_event = await asyncio.wait_for(queue.get(), remaining)
            now = clock.now()
            late = max(0, now - release_event["scheduled"])
            lateness.append(late)
            pending = max(0, min(duration, now - t0) -
                          (release_event["cumulative_samples"] - release_event["samples"]) / 16000)
            context = dict(release_event, packet_lateness_s=late, pending_audio_s=pending)
            response = await client.request("feed", binary=chunk, session_id=session_id,
                                            deadline=deadline, context=context)
            validate_visible(response, final=False)
        result["end_backlog_s"] = max(0, clock.now() - t_stop)
        result["stop_request"] = clock.now()
        response = await client.request("stop", session_id=session_id, deadline=deadline,
                                        context={"stop_intent": t_stop})
        validate_visible(response, final=True)
        result.update(packet_lateness_max_s=max(lateness),
                      packet_lateness_p50_s=percentile(lateness, .5),
                      packet_lateness_p95_s=percentile(lateness, .95))
        return result
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


def words(text: str, *, reference=False) -> list[str]:
    return normalize_for_wer(text, strip_annotations=reference).split()


def analyze_stream(events: list[dict], reference: str, t0: float, t_stop: float,
                   *, kind: str = "short") -> dict:
    ref = words(reference, reference=True)
    prefix = ref[:3]
    visible = []
    violations = []
    committed = ""
    previous: list[str] = []
    rollback = 0
    first_commit = None
    batch_corrections = 0
    final = None
    for event in events:
        if event.get("command") not in {"feed", "stop"} or "reply_end" not in event or event.get("error"):
            continue
        response = event.get("response", {})
        try:
            validate_visible(response, final=event["command"] == "stop")
        except (ValueError, AttributeError) as error:
            violations.append(str(error))
            continue
        batch_corrections += int(response.get("batch_corrected", False))
        text = response["text"]
        new_commit = response["committed_text"]
        if not new_commit.startswith(committed):
            violations.append("committed-prefix-changed")
        if not text.startswith(committed):
            violations.append("committed-text-lost")
        if text != " ".join(part for part in (new_commit, response["active_text"]) if part):
            violations.append("text-reconstruction")
        committed = new_commit
        if committed and first_commit is None:
            first_commit = event["reply_end"] - t0
        hyp = words(text)
        common = 0
        for a, b in zip(previous, hyp):
            if a != b:
                break
            common += 1
        rollback += len(previous) - common
        previous = hyp
        visible.append((event["reply_end"], hyp))
        if event["command"] == "stop":
            final = event
    if final is None:
        violations.append("missing-final")
    useful = [bool(prefix) and hyp[:len(prefix)] == prefix for _, hyp in visible]
    pre_stop = [at < t_stop for at, _ in visible]
    first_nonempty = next((at - t0 for at, hyp in visible if hyp), None)
    first_useful = next((at - t0 for i, (at, _) in enumerate(visible) if pre_stop[i] and useful[i]), None)
    stable = next((at - t0 for i, (at, _) in enumerate(visible)
                   if pre_stop[i] and useful[i] and all(useful[i:])), None) if final else None
    hypothesis = final["response"]["text"] if final else None
    lexical = kind not in UNSCORED and bool(ref)
    scores = score("wer", reference, hypothesis) if hypothesis is not None and lexical else None
    tail = ref[-5:]
    hyp_words = words(hypothesis) if hypothesis is not None else []
    tail_counts = edit_counts(tail, hyp_words[-5:]) if scores else None

    def occurrences(sequence, phrase):
        return sum(sequence[i:i + len(phrase)] == phrase for i in range(len(sequence) - len(phrase) + 1)) if phrase else 0

    silence = kind in {"silence-only", "noise-only"}
    return {
        "first_nonempty_s": first_nonempty, "first_useful_s": first_useful,
        "stable_useful_s": stable, "first_commit_s": first_commit,
        "stop_to_final_s": final["reply_end"] - t_stop if final else None,
        "stop_request_rtt_s": final["reply_end"] - final["write_start"] if final else None,
        "final_pass_runtime_s": None,  # Not supplied by common ASRIPC; do not equate with RTT.
        "scores": scores, "lexical_eligible": lexical, "hypothesis": hypothesis,
        "normalized_reference": " ".join(ref),
        "normalized_hypothesis": " ".join(hyp_words) if hypothesis is not None else None,
        "rollback_words": rollback, "rollback_rate": rollback / len(ref) if lexical else None,
        "tail_errors": tail_counts.errors if tail_counts else None,
        "tail_missing_words": tail_counts.deletions if tail_counts else None,
        "tail_excess_repetitions": max(0, occurrences(hyp_words, tail) - occurrences(ref, tail)) if scores else None,
        "hallucination_events": sum(bool(hyp) for _, hyp in visible) if silence else None,
        "hallucination_words": sum(len(hyp) for _, hyp in visible) if silence else None,
        "hallucination_complete": final is not None if silence else None,
        "batch_correction_events": batch_corrections,
        "violations": sorted(set(violations)),
    }


def distribution(values: list[float | None]) -> dict:
    present = [value for value in values if value is not None]
    return {"n": len(present), "missing": len(values) - len(present),
            "p50": percentile(present, .5) if present else None,
            "p95": percentile(present, .95) if present else None}


def quality(rows: list[dict]) -> dict:
    scores = [row["metrics"]["scores"] for row in rows
              if row.get("status") == "ok" and row.get("metrics", {}).get("scores") is not None]
    counts = {key: sum(s[key] for s in scores) for key in
              ("correct", "substitutions", "deletions", "insertions", "errors", "reference_units")}
    units = counts["reference_units"]
    return dict(counts, n=len(scores), micro_wer=counts["errors"] / units if units else None,
                macro_wer=sum(s["errors"] / s["reference_units"] for s in scores) / len(scores) if scores else None)


def summarize_trials(rows: list[dict]) -> dict:
    result = {"label": LABEL, "interpretation": "descriptive dev observations only; no promotion",
              "systems": {}, "paired": {}}
    by_system = {system: [row for row in rows if row["system"] == system] for system in SYSTEMS}
    for system, trials in by_system.items():
        lexical_trials = [row for row in trials if row.get("metrics", {}).get("scores") is not None]
        rollback_units = sum(row["metrics"]["scores"]["reference_units"] for row in lexical_trials)
        result["systems"][system] = {
            "aggregate_rollback_rate": (sum(row["metrics"]["rollback_words"] for row in lexical_trials)
                                        / rollback_units if rollback_units else None),
            "trials": len(trials), "failed": sum(row["status"] != "ok" for row in trials),
            "quality": quality(trials),
            "latencies": {key: distribution([row.get("metrics", {}).get(key) if row["status"] == "ok" else None
                                            for row in trials]) for key in ENDPOINTS},
            "invariants": {key: sum(row.get("metrics", {}).get(key) or 0 for row in trials)
                           for key in ("tail_errors", "tail_missing_words", "tail_excess_repetitions",
                                       "rollback_words", "hallucination_events", "hallucination_words")},
            "violations": sum(len(row.get("metrics", {}).get("violations", [])) for row in trials),
            "worst_rollback_rate": max((row.get("metrics", {}).get("rollback_rate") or 0 for row in trials), default=None),
        }
    baseline = {row["case_id"]: row for row in by_system["Qwen"]}
    for system in ("Nemotron", "Parakeet"):
        candidate = {row["case_id"]: row for row in by_system[system]}
        ids = sorted(set(baseline) | set(candidate))
        pairs = [(baseline.get(cid), candidate.get(cid)) for cid in ids]
        complete = [(a, b) for a, b in pairs if a and b and a["status"] == b["status"] == "ok"]
        lexical = [(a, b) for a, b in complete if a.get("metrics", {}).get("scores") is not None
                   and b.get("metrics", {}).get("scores") is not None]
        base_quality = quality([a for a, _ in lexical])
        cand_quality = quality([b for _, b in lexical])
        latencies = {}
        for key in ENDPOINTS:
            observed = [(a["metrics"].get(key), b["metrics"].get(key)) for a, b in complete]
            valid = [(a, b) for a, b in observed if a is not None and b is not None]
            base_dist = distribution([a for a, _ in valid])
            cand_dist = distribution([b for _, b in valid])
            latencies[key] = {
                "pairs": len(valid), "missing_pairs": len(ids) - len(valid),
                "baseline": base_dist, "candidate": cand_dist,
                "paired_delta_s": distribution([b - a for a, b in valid] + [None] * (len(ids) - len(valid))),
                "p50_delta_s": cand_dist["p50"] - base_dist["p50"] if valid else None,
                "p95_delta_s": cand_dist["p95"] - base_dist["p95"] if valid else None,
            }
        invariant_deltas = {}
        for key in ("tail_errors", "tail_missing_words", "tail_excess_repetitions", "rollback_words",
                    "hallucination_events", "hallucination_words"):
            observed = [(a["metrics"].get(key), b["metrics"].get(key)) for a, b in complete]
            valid = [(a, b) for a, b in observed if a is not None and b is not None]
            invariant_deltas[key] = {"pairs": len(valid), "unavailable_or_inapplicable_pairs": len(ids) - len(valid),
                                     "delta": sum(b - a for a, b in valid) if valid else None}
        result["paired"][system] = {
            "invariant_deltas": invariant_deltas,
            "aggregate_rollback_rate_delta": (sum(b["metrics"]["rollback_words"] - a["metrics"]["rollback_words"]
                                                   for a, b in lexical) / base_quality["reference_units"]
                                                if lexical else None),
            "pairs": len(complete), "missing_pairs": len(ids) - len(complete),
            "lexical_pairs": len(lexical), "baseline_quality": base_quality,
            "candidate_quality": cand_quality,
            "csdi_delta": {key: cand_quality[key] - base_quality[key] for key in
                           ("correct", "substitutions", "deletions", "insertions", "errors")},
            "micro_wer_delta": cand_quality["micro_wer"] - base_quality["micro_wer"] if lexical else None,
            "latencies": latencies,
        }
    return result


def load_snapshot() -> dict:
    return {"at": time.monotonic(), "wall_time": time.time(), "load_average": os.getloadavg(),
            "cpu_count": os.cpu_count(), "os": platform.platform(),
            "no_load_control": True,
            "unavailable": ["30s clearance", "continuous CPU/memory/thermal/swap", "GPU/ANE activity", "external workload exclusion"],
            "note": "Snapshot only. Caller owns external clearance and continuous telemetry; this runner cannot attest it."}


async def run_trial(system: str, command: list[str], case: EvaluationCase, metadata: dict,
                    warmup: tuple[EvaluationCase, dict], *, deadline: float, stderr_path: Path,
                    load_status: str) -> dict:
    events: list[dict] = []
    row = {"label": LABEL, "system": system, "slot": {"Qwen": "baseline", "Nemotron": "N1", "Parakeet": "P1"}[system],
           "case_id": case.id, "metadata": metadata, "reference": case.reference, "command": command,
           "status": "failed", "load_status": load_status, "no_load_control": True,
           "load_before": load_snapshot(), "events": events, "metrics": {}, "stderr_path": str(stderr_path)}
    client = None
    timing: dict = {}
    measured_events: list[dict] = []
    spawned = time.monotonic()
    row["spawn_start"] = spawned
    try:
        pcm = read_pcm(case, metadata)
        warm_pcm = read_pcm(*warmup)
        with stderr_path.open("wb") as stderr:
            if deadline <= time.monotonic():
                raise TimeoutError("global deadline exhausted before spawn")
            async with asyncio.timeout(deadline - time.monotonic()):
                client = await IPCClient.spawn(command, events, stderr=stderr)
                row["pid"] = client.process.pid
                ready = await client.request("info", deadline=min(deadline, spawned + 180))
                row["readiness"] = ready
                row["spawn_to_ready_s"] = time.monotonic() - spawned
                if ready.get("status") != "ready" or ready.get("sample_rate") != 16000:
                    raise ValueError("not ready at 16000 Hz")
                if system == "Qwen" and ready.get("final_accuracy_pass_enabled") is not True:
                    raise ValueError("Qwen final accuracy is not enabled")
                client.phase = "warmup"
                warm_start = len(events)
                warm_timing = await run_stream(client, warm_pcm, events, deadline=deadline)
                warm_metrics = analyze_stream(events[warm_start:], warmup[0].reference,
                                              warm_timing["t0"], warm_timing["t_stop"])
                row["warmup"] = {"case_id": warmup[0].id, "timing": warm_timing,
                                 "violations": warm_metrics["violations"], "scored": False}
                if warm_metrics["violations"]:
                    raise ValueError("warmup stream invariant failure")
                client.phase = "measured"
                client.events = measured_events
                await run_stream(client, pcm, measured_events, deadline=deadline, timing=timing)
                if timing["session_id"] == warm_timing["session_id"]:
                    raise ValueError("measured create reused warmup session_id")
                row["status"] = "ok"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        events.extend(measured_events)
        if client:
            client.events = events
            try:
                await client.close()
            except Exception as error:
                row["cleanup_error"] = f"{type(error).__name__}: {error}"
                row["status"] = "failed"
        row["timing"] = timing
        if "t0" in timing:
            metrics = analyze_stream(measured_events, case.reference, timing["t0"], timing["t_stop"], kind=metadata["kind"])
            metrics.update({key: timing.get(key) for key in ENDPOINTS if key in timing})
            metrics["spawn_to_ready_s"] = row.get("spawn_to_ready_s")
            metrics["startup_inclusive_useful_s"] = timing["t0"] - spawned + metrics["first_useful_s"] if metrics["first_useful_s"] is not None else None
            row["metrics"] = metrics
            if metrics["violations"] or metrics["hallucination_events"]:
                row["status"] = "failed"
        row["load_after"] = load_snapshot()
        row["finished"] = time.monotonic()
    return row


def load_commands(path: Path) -> dict[str, list[str]]:
    config = json.loads(path.read_text())
    if set(config) != set(SYSTEMS):
        raise ValueError("command config must have exactly Qwen, Nemotron, Parakeet (N1/P1 only)")
    for command in config.values():
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise ValueError("each command must be a nonempty argv string array; no shell")
    return config


async def execute(args, cases, commands, receipt) -> int:
    started = time.monotonic()
    deadline = started + args.max_seconds - 2  # Reserve bounded owned-process cleanup.
    warmup = next(((case, meta) for case, meta in cases if not meta["diagnostic"] and meta["kind"] == "short"), None)
    if warmup is None:
        raise ValueError("first dev-short warmup missing")
    blocks = ordered_blocks([case.id for case, _ in cases])
    if args.case_limit:
        blocks = blocks[:args.case_limit]
    index = {case.id: (case, meta) for case, meta in cases}
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier experiment (including failed/partial outputs).
    with args.output.open("x") as output:
        def emit(row):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
        emit(dict(receipt, type="receipt", planned_trials=len(blocks) * 3))
        for cid, systems in blocks:
            for system in systems:
                case, meta = index[cid]
                if time.monotonic() >= deadline:
                    row = {"label": LABEL, "system": system, "case_id": cid, "status": "not-run",
                           "error": "global deadline exhausted", "metrics": {},
                           "no_load_control": True, "load_status": args.load_status}
                else:
                    stderr_path = args.output.with_name(f"{args.output.name}.{len(rows):03d}.{system}.stderr")
                    row = await run_trial(system, commands[system], case, meta, warmup,
                                          deadline=deadline, stderr_path=stderr_path, load_status=args.load_status)
                rows.append(row)
                emit(dict(row, type="trial"))
        summary = summarize_trials(rows)
        summary.update(type="summary", elapsed_s=time.monotonic() - started,
                       scope="transport-smoke" if args.case_limit else "dev-single-pass",
                       complete_dev_pass=not args.case_limit and len(rows) == 81 and all(r["status"] == "ok" for r in rows),
                       no_load_control=True, load_status=args.load_status)
        emit(summary)
    print(json.dumps({"label": LABEL, "output": str(args.output), "trials": len(rows),
                      "failed_or_missing": sum(r["status"] != "ok" for r in rows),
                      "complete_dev_pass": summary["complete_dev_pass"]}))
    return 1 if any(r["status"] != "ok" for r in rows) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NON-ACCEPTANCE dev-only real-time ASR diagnostics")
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "english-parity"
    parser.add_argument("--manifest", type=Path, default=fixtures / "dev.jsonl")
    parser.add_argument("--diagnostics", type=Path, default=fixtures / "diagnostics.jsonl")
    parser.add_argument("--commands", type=Path, required=True, help="JSON: Qwen/Nemotron/Parakeet -> exact argv arrays")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL path (never overwritten)")
    parser.add_argument("--execute", action="store_true", help="requires external owner pre-run approval; otherwise plan only")
    parser.add_argument("--load-status", choices=["blocked-load-control", "diagnostic-under-load", "external-clearance", "uncontrolled"], default="blocked-load-control")
    parser.add_argument("--case-limit", type=int, help="1..27 blocks; transport smoke only, never completed dev gate")
    parser.add_argument("--max-seconds", type=float, default=14400, help="global startup/warmup/measurement budget, at most four hours")
    args = parser.parse_args()
    if not 2 < args.max_seconds <= 14400 or (args.case_limit is not None and not 1 <= args.case_limit <= 27):
        parser.error("invalid bounded deadline/case limit")
    if args.execute and args.load_status == "uncontrolled" and not args.case_limit:
        parser.error("uncontrolled execution is only for explicit --case-limit transport smoke")
    commands = load_commands(args.commands)
    cases = load_dev_cases(args.manifest, args.diagnostics)
    receipt = {"label": LABEL, "commands": commands, "load_status": args.load_status,
               "no_load_control": True, "max_seconds": args.max_seconds,
               "phase": "dev", "repeat": 1, "clock": "time.monotonic",
               "clock_resolution_s": time.get_clock_info("monotonic").resolution,
               "python": platform.python_version(), "case_limit": args.case_limit,
               "hashes": {str(path): sha256(path.read_bytes()) for path in
                          (args.manifest, args.diagnostics, args.commands, Path(__file__),
                           Path(__file__).with_name("transcript_metrics.py"),
                           fixtures / "protocol.md", fixtures / "dev-only-addendum.md")},
               "external_receipt_required": ["owner/reviewer approval", "binaries/metallib/assets/SDK/source hashes",
                                             "exact baseline flags/live VAD", "compute units", "continuous load clearance",
                                             "builds/downloads complete; no other ASR benchmark"]}
    if not args.execute or args.load_status == "blocked-load-control":
        print(json.dumps(dict(receipt, status="plan-only" if not args.execute else "blocked-load-control",
                              blocks=ordered_blocks([case.id for case, _ in cases])[:args.case_limit]), indent=2))
        return 2 if args.execute else 0
    return asyncio.run(execute(args, cases, commands, receipt))


if __name__ == "__main__":
    raise SystemExit(main())
