#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy", "psutil", "requests"]
# ///
"""
Benchmark native Yuwp ASR server with single-model presets and concurrent HTTP clients.

This script targets the Swift MLX ASR server, not the old Python sidecar.
It can compare the small/large single-model presets, capture final transcripts,
and measure how latency and memory behave as concurrency increases.

Interface note:
- use `yuwp-asr serve --model <path-or-repo-id>`

Examples:
  # Compare the two built-in single-model presets on a balanced corpus.
  uv run benchmarks/cli.py asr load small large --balanced 5 --concurrency 1 2 4

  # Stress one model with real-time pacing and save JSON for later review.
  uv run benchmarks/cli.py asr load large --balanced 5 --concurrency 1 2 4 \
    --pace realtime --capture-text --json /tmp/native-bench.json

  # Run on explicit files.
  uv run benchmarks/cli.py asr load small large \
    --files path/to/audio1.flac path/to/audio2.flac
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import requests

SAMPLE_RATE = 16_000
CHUNK_SEC = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)
READY_TIMEOUT = 240.0
READY_POLL_SEC = 0.5
REQUEST_TIMEOUT = 120.0
MEMORY_SAMPLE_SEC = 0.2
DEFAULT_CORPUS_ROOT = Path.home() / "Library" / "Application Support" / "Yuwp" / "recordings"
DEFAULT_JSON_PATH = "/tmp/yuwp-native-bench.json"

MODEL_ALIASES = {
    "small": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "large": "mlx-community/Qwen3-ASR-1.7B-bf16",
    "0.6B-4bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "1.7B-bf16": "mlx-community/Qwen3-ASR-1.7B-bf16",
}


@dataclass(slots=True)
class AudioCase:
    path: Path
    duration_s: float
    pcm: np.ndarray
    saved_text: str
    cjk_count: int


@dataclass(slots=True)
class PreparedModel:
    spec: str
    repo_id: str | None
    resolved_path: Path
    label: str


class MemorySampler:
    def __init__(self, pid: int, interval_s: float = MEMORY_SAMPLE_SEC):
        self.pid = pid
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples_mb: list[float] = []

    def start(self) -> None:
        self._stop.clear()
        self.samples_mb.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def peak_mb(self) -> float:
        return max(self.samples_mb) if self.samples_mb else 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples_mb.append(process_tree_rss_mb(self.pid))
            self._stop.wait(self.interval_s)


def looks_like_path(spec: str) -> bool:
    return spec.startswith("/") or spec.startswith("~") or spec.startswith(".")


def resolve_model_spec(spec: str) -> PreparedModel:
    raw = spec.strip()
    if not raw:
        raise ValueError("empty model spec")

    alias = MODEL_ALIASES.get(raw, raw)
    if looks_like_path(alias):
        path = Path(alias).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"model directory not found: {path}")
        return PreparedModel(spec=raw, repo_id=None, resolved_path=path, label=path.name)

    repo_id = alias if "/" in alias else MODEL_ALIASES.get(alias, alias)
    if "/" not in repo_id:
        raise FileNotFoundError(f"could not resolve model spec: {spec}")

    path = find_hf_snapshot(repo_id)
    if path is None:
        raise FileNotFoundError(
            f"model not found in Hugging Face cache: {repo_id}\n"
            f"Download it first in Yuwp.app or place it in ~/.cache/huggingface/hub"
        )
    return PreparedModel(spec=raw, repo_id=repo_id, resolved_path=path, label=repo_id.split("/")[-1])


def find_hf_snapshot(repo_id: str) -> Path | None:
    owner, name = repo_id.split("/", 1)
    snapshots_dir = None
    for root in hf_roots():
        candidate = root / f"models--{owner}--{name}" / "snapshots"
        if candidate.is_dir():
            snapshots_dir = candidate
            break
    if snapshots_dir is None:
        return None

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in snapshots:
        if is_valid_model_dir(snap):
            return snap
    return None


def hf_roots() -> list[Path]:
    env = os.environ
    roots: list[Path] = []
    if env.get("HF_HOME"):
        roots.append(Path(env["HF_HOME"]).expanduser() / "hub")
    if env.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(env["HUGGINGFACE_HUB_CACHE"]).expanduser())
    roots.append(Path("~/.cache/huggingface/hub").expanduser())
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root.expanduser())
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def is_valid_model_dir(path: Path) -> bool:
    required = ["config.json", "model.safetensors", "vocab.json", "merges.txt"]
    return all((path / name).exists() for name in required)


def find_server_binary() -> Path:
    candidates = [
        Path(".build/arm64-apple-macosx/release/yuwp-asr"),
        Path(".build/release/yuwp-asr"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("server binary not found — run: swift build -c release --product yuwp-asr")


def count_cjk(text: str) -> int:
    return sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")


def normalize(text: str) -> str:
    chars: list[str] = []
    for ch in text.casefold():
        if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf"):
            chars.append(ch)
    return "".join(chars)


def text_similarity(a: str, b: str) -> float | None:
    if not a and not b:
        return 1.0
    if not a or not b:
        return None
    na = normalize(a)
    nb = normalize(b)
    if not na and not nb:
        return 1.0
    return SequenceMatcher(a=na, b=nb).ratio()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def companion_json(audio_path: Path) -> Path:
    return audio_path.with_suffix(".json")


def load_saved_text(audio_path: Path) -> str:
    meta_path = companion_json(audio_path)
    if not meta_path.exists():
        return ""
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return ""
    return (meta.get("transcript") or "").strip()


def decode_audio(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {result.stderr.decode(errors='ignore')[:200]}")
    return np.frombuffer(result.stdout, dtype=np.int16)


def collect_audio_paths(args: argparse.Namespace) -> list[Path]:
    if args.files:
        files = [Path(p).expanduser().resolve() for p in args.files]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError("missing audio file(s):\n  " + "\n  ".join(missing))
        return files

    root = Path(args.root).expanduser()
    globs = [str(root / "**/*.flac"), str(root / "**/*.wav")]
    candidates = sorted({Path(p).resolve() for pattern in globs for p in glob.glob(pattern, recursive=True)})

    if args.date:
        parts = args.date.split("-")
        needle = "/".join(parts)
        candidates = [p for p in candidates if needle in str(p)]

    if args.balanced is not None:
        with_saved = [p for p in candidates if load_saved_text(p)]
        cjk = [p for p in with_saved if count_cjk(load_saved_text(p)) > 0]
        non_cjk = [p for p in with_saved if count_cjk(load_saved_text(p)) == 0]
        picked = cjk[-args.balanced :] + non_cjk[-args.balanced :]
        return sorted(dict.fromkeys(picked))

    last_n = args.last if args.last is not None else 10
    return candidates[-last_n:]


def prepare_cases(paths: list[Path]) -> list[AudioCase]:
    prepared: list[AudioCase] = []
    for path in paths:
        pcm = decode_audio(path)
        duration_s = len(pcm) / SAMPLE_RATE
        saved = load_saved_text(path)
        prepared.append(
            AudioCase(
                path=path,
                duration_s=duration_s,
                pcm=pcm,
                saved_text=saved,
                cjk_count=count_cjk(saved),
            )
        )
    return prepared


def wait_for_ready(port: int) -> tuple[float, dict[str, Any]]:
    url = f"http://127.0.0.1:{port}/v1/info"
    start = time.perf_counter()
    deadline = start + READY_TIMEOUT
    while time.perf_counter() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.ok:
                payload = resp.json()
                if payload.get("status") == "ready":
                    return time.perf_counter() - start, payload
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(READY_POLL_SEC)
    raise TimeoutError(f"server not ready on port {port} after {READY_TIMEOUT:.0f}s")


def process_tree_rss_mb(pid: int) -> float:
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return 0.0
    total = 0
    for proc in [parent, *parent.children(recursive=True)]:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            pass
    return total / (1024 * 1024)


def launch_server(server_bin: Path, model: PreparedModel, port: int, warmup: bool) -> tuple[subprocess.Popen[str], float, dict[str, Any]]:
    cmd = [
        str(server_bin),
        "serve",
        "--model",
        str(model.resolved_path),
        "--batch-model",
        str(model.resolved_path),
        "--transport",
        "http",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    if warmup:
        cmd.append("--warmup")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        load_s, info = wait_for_ready(port)
        return proc, load_s, info
    except Exception:
        terminate_process(proc)
        raise


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def stream_case(case: AudioCase, base_url: str, pace: str, capture_text: bool) -> dict[str, Any]:
    session = requests.Session()
    create = session.post(f"{base_url}/v1/audio/transcriptions/stream", json={}, timeout=REQUEST_TIMEOUT)
    create.raise_for_status()
    session_id = create.json()["session_id"]

    chunk_latencies_ms: list[float] = []
    stream_text = ""
    first_nonempty_ms: float | None = None
    start = time.perf_counter()

    for idx in range(0, len(case.pcm), CHUNK_SAMPLES):
        if pace == "realtime":
            due = (idx // CHUNK_SAMPLES) * CHUNK_SEC
            remaining = due - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)

        chunk = case.pcm[idx : idx + CHUNK_SAMPLES]
        t0 = time.perf_counter()
        resp = session.post(
            f"{base_url}/v1/audio/transcriptions/stream/{session_id}",
            data=chunk.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        chunk_ms = (time.perf_counter() - t0) * 1000
        chunk_latencies_ms.append(chunk_ms)

        next_text = resp.json().get("text", stream_text)
        if first_nonempty_ms is None and next_text.strip():
            first_nonempty_ms = (time.perf_counter() - start) * 1000
        stream_text = next_text

    stop = session.delete(f"{base_url}/v1/audio/transcriptions/stream/{session_id}", timeout=REQUEST_TIMEOUT)
    stop.raise_for_status()
    final_text = stop.json().get("text", stream_text)
    total_s = time.perf_counter() - start

    saved_text = case.saved_text
    stream_similarity = text_similarity(stream_text, saved_text)
    final_similarity = text_similarity(final_text, saved_text)

    result: dict[str, Any] = {
        "file": case.path.name,
        "duration_s": round(case.duration_s, 3),
        "chunk_count": len(chunk_latencies_ms),
        "avg_chunk_ms": sum(chunk_latencies_ms) / len(chunk_latencies_ms) if chunk_latencies_ms else 0.0,
        "p50_chunk_ms": percentile(chunk_latencies_ms, 50),
        "p95_chunk_ms": percentile(chunk_latencies_ms, 95),
        "first_nonempty_ms": first_nonempty_ms,
        "total_time_s": total_s,
        "realtime_factor": total_s / case.duration_s if case.duration_s > 0 else 0.0,
        "saved_chars": len(saved_text),
        "cjk_saved": case.cjk_count,
        "stream_similarity": stream_similarity,
        "final_similarity": final_similarity,
        "stream_exact_norm": normalize(stream_text) == normalize(saved_text) if saved_text else None,
        "final_exact_norm": normalize(final_text) == normalize(saved_text) if saved_text else None,
        "_chunk_latencies_ms": chunk_latencies_ms,
    }
    if capture_text:
        result["saved_text"] = saved_text
        result["stream_text"] = stream_text
        result["final_text"] = final_text
    return result


def summarize_run(session_results: list[dict[str, Any]], wall_time_s: float, rss_before_mb: float, peak_rss_mb: float) -> dict[str, Any]:
    all_chunk_latencies = [
        latency
        for session in session_results
        for latency in session.pop("_chunk_latencies_ms", [])
    ]
    firsts = [s["first_nonempty_ms"] for s in session_results if s["first_nonempty_ms"] is not None]
    final_sim = [s["final_similarity"] for s in session_results if s["final_similarity"] is not None]
    stream_sim = [s["stream_similarity"] for s in session_results if s["stream_similarity"] is not None]
    total_audio_s = sum(s["duration_s"] for s in session_results)

    return {
        "sessions": len(session_results),
        "total_audio_s": total_audio_s,
        "wall_time_s": wall_time_s,
        "throughput_x": total_audio_s / wall_time_s if wall_time_s > 0 else 0.0,
        "rss_before_mb": rss_before_mb,
        "peak_rss_mb": peak_rss_mb,
        "peak_delta_mb": peak_rss_mb - rss_before_mb,
        "avg_chunk_ms": sum(all_chunk_latencies) / len(all_chunk_latencies) if all_chunk_latencies else 0.0,
        "p50_chunk_ms": percentile(all_chunk_latencies, 50),
        "p95_chunk_ms": percentile(all_chunk_latencies, 95),
        "avg_first_nonempty_ms": sum(firsts) / len(firsts) if firsts else None,
        "files_with_partial": len(firsts),
        "avg_realtime_factor": sum(s["realtime_factor"] for s in session_results) / len(session_results) if session_results else 0.0,
        "stream_similarity_mean": sum(stream_sim) / len(stream_sim) if stream_sim else None,
        "final_similarity_mean": sum(final_sim) / len(final_sim) if final_sim else None,
        "stream_exact_norm": sum(1 for s in session_results if s.get("stream_exact_norm") is True),
        "final_exact_norm": sum(1 for s in session_results if s.get("final_exact_norm") is True),
    }


def benchmark_concurrency(
    base_url: str,
    cases: list[AudioCase],
    concurrency: int,
    repeats: int,
    pace: str,
    capture_text: bool,
    sampler: MemorySampler,
    pid: int,
) -> dict[str, Any]:
    work_items = [cases[i % len(cases)] for i in range(len(cases) * repeats)]
    session_results: list[dict[str, Any]] = []

    rss_before_mb = process_tree_rss_mb(pid)
    sampler.start()
    wall_start = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(stream_case, case, base_url, pace, capture_text) for case in work_items]
            for future in concurrent.futures.as_completed(futures):
                session_results.append(future.result())
    finally:
        wall_time_s = time.perf_counter() - wall_start
        sampler.stop()

    session_results.sort(key=lambda s: s["file"])
    summary = summarize_run(session_results, wall_time_s, rss_before_mb, sampler.peak_mb)
    return {
        "concurrency": concurrency,
        "repeats": repeats,
        "pace": pace,
        "summary": summary,
        "sessions": session_results,
    }


def print_cases(cases: list[AudioCase]) -> None:
    print(f"Selected {len(cases)} audio file(s):")
    for case in cases:
        flavor = "CJK" if case.cjk_count > 0 else "EN"
        saved_note = " +saved" if case.saved_text else ""
        print(f"  - {case.path.name}  {case.duration_s:.1f}s  {flavor}{saved_note}")


def print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    print("\nSUMMARY")
    print("=" * 104)
    header = (
        f"{'model':<18} {'conc':>4} {'load_s':>7} {'rss':>8} {'peak':>8} "
        f"{'avg_ms':>8} {'p95_ms':>8} {'first_ms':>9} {'rtf':>7} {'sim':>7}"
    )
    print(header)
    print("-" * len(header))
    for model_result in results:
        for run in model_result["runs"]:
            summary = run["summary"]
            first = summary["avg_first_nonempty_ms"]
            sim = summary["final_similarity_mean"]
            print(
                f"{model_result['label']:<18} {run['concurrency']:>4} {model_result['load_s']:>7.2f} "
                f"{summary['rss_before_mb']:>8.0f} {summary['peak_rss_mb']:>8.0f} "
                f"{summary['avg_chunk_ms']:>8.1f} {summary['p95_chunk_ms']:>8.1f} "
                f"{(f'{first:.1f}' if first is not None else '-'):>9} "
                f"{summary['avg_realtime_factor']:>7.3f} {(f'{sim:.3f}' if sim is not None else '-'):>7}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark native yuwp-asr serve with single-model presets and concurrent clients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s small large --balanced 5 --concurrency 1 2 4
  %(prog)s large --last 10 --pace realtime --concurrency 1 2
  %(prog)s small large --files file1.flac file2.flac --capture-text --json /tmp/out.json
""",
    )
    parser.add_argument("models", nargs="+", help="Model specs: small, large, 0.6B-4bit, 1.7B-bf16, repo id, or local path")
    parser.add_argument("--files", nargs="+", help="Explicit audio files to benchmark")
    parser.add_argument("--root", default=str(DEFAULT_CORPUS_ROOT), help=f"Corpus root (default: {DEFAULT_CORPUS_ROOT})")
    parser.add_argument("--last", type=int, help="Use the last N audio files from the corpus root")
    parser.add_argument("--balanced", type=int, help="Use last N CJK + last N non-CJK files with saved transcripts")
    parser.add_argument("--date", type=str, help="Filter corpus by date (YYYY-MM-DD)")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1], help="Concurrent client counts to test (default: 1)")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat the selected corpus this many times per concurrency level")
    parser.add_argument("--pace", choices=["none", "realtime"], default="none", help="Send chunks as fast as possible or at real-time pace")
    parser.add_argument("--warmup", action="store_true", help="Pass --warmup when launching yuwp-asr serve")
    parser.add_argument("--port-base", type=int, default=9790, help="Base TCP port for launched servers (default: 9790)")
    parser.add_argument("--capture-text", action="store_true", help="Store saved/stream/final text in JSON output")
    parser.add_argument("--json", metavar="PATH", help="Write full JSON results to PATH")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        server_bin = find_server_binary()
        models = [resolve_model_spec(spec) for spec in args.models]
        paths = collect_audio_paths(args)
        if not paths:
            raise FileNotFoundError("no audio files found")
        cases = prepare_cases(paths)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print_cases(cases)
    print(f"\nServer: {server_bin}")
    print("Models:")
    for model in models:
        print(f"  - {model.spec} -> {model.resolved_path}")

    results: list[dict[str, Any]] = []
    for idx, model in enumerate(models, start=1):
        port = args.port_base + idx
        base_url = f"http://127.0.0.1:{port}"
        if not args.quiet:
            print(f"\n=== {model.label} on port {port} ===")
        proc, load_s, info = launch_server(server_bin, model, port, args.warmup)
        sampler = MemorySampler(proc.pid)
        model_result: dict[str, Any] = {
            "spec": model.spec,
            "label": model.label,
            "repo_id": model.repo_id,
            "resolved_path": str(model.resolved_path),
            "port": port,
            "load_s": load_s,
            "server_info": info,
            "runs": [],
        }

        try:
            for concurrency in args.concurrency:
                if not args.quiet:
                    print(f"  -> concurrency {concurrency} x repeats {args.repeats} ({args.pace})")
                run = benchmark_concurrency(
                    base_url=base_url,
                    cases=cases,
                    concurrency=concurrency,
                    repeats=args.repeats,
                    pace=args.pace,
                    capture_text=args.capture_text,
                    sampler=sampler,
                    pid=proc.pid,
                )
                model_result["runs"].append(run)
                if not args.quiet:
                    s = run["summary"]
                    sim = s["final_similarity_mean"]
                    print(
                        "     "
                        f"avg_chunk={s['avg_chunk_ms']:.1f}ms  "
                        f"p95={s['p95_chunk_ms']:.1f}ms  "
                        f"peak_rss={s['peak_rss_mb']:.0f}MB  "
                        f"sim={(f'{sim:.3f}' if sim is not None else '-') }"
                    )
        finally:
            terminate_process(proc)
            if proc.stderr:
                stderr_text = proc.stderr.read().strip()
                if stderr_text and not args.quiet:
                    tail = "\n".join(stderr_text.splitlines()[-8:])
                    print(f"  stderr tail:\n{tail}")

        results.append(model_result)

    print_summary(results)

    if args.json:
        out_path = Path(args.json).expanduser()
        out_path.write_text(json.dumps({"cases": [case.path.name for case in cases], "results": results}, ensure_ascii=False, indent=2))
        print(f"\nWrote {out_path}")
    elif args.capture_text:
        out_path = Path(DEFAULT_JSON_PATH)
        out_path.write_text(json.dumps({"cases": [case.path.name for case in cases], "results": results}, ensure_ascii=False, indent=2))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
