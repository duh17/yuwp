#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# ///

"""Benchmark Yuwp ASR transport overhead: HTTP vs stdio.

Runs the same audio through streaming create/feed/stop sessions and reports
latency stats for each transport.
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import math
import os
import pathlib
import select
import socket
import statistics
import struct
import subprocess
import sys
import time
import wave
from typing import Any

REQUIRED_MODEL_FILES = ("config.json", "model.safetensors", "vocab.json", "merges.txt")


@dataclasses.dataclass
class SessionMetrics:
    create_ms: float
    feed_mean_ms: float
    feed_p50_ms: float
    feed_p95_ms: float
    feed_total_ms: float
    stop_ms: float
    total_ms: float
    final_chars: int
    updates_with_text: int


@dataclasses.dataclass
class ModeResult:
    mode: str
    startup_ms: float
    sessions: list[SessionMetrics]


class BenchError(RuntimeError):
    pass


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


class HTTPTransportClient:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.base_path = "/v1/audio/transcriptions/stream"

    def _request(self, method: str, path: str, body: bytes = b"") -> dict[str, Any]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        try:
            headers = {"Content-Length": str(len(body))}
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            if response.status != 200:
                raise BenchError(f"HTTP {response.status} for {method} {path}: {raw.decode('utf-8', errors='ignore')}")
            if not raw:
                return {}
            return json.loads(raw)
        finally:
            conn.close()

    def wait_ready(self, timeout_s: float) -> float:
        start = time.perf_counter()
        deadline = start + timeout_s
        last_error: Exception | None = None
        while time.perf_counter() < deadline:
            try:
                payload = self._request("GET", "/v1/info")
                if payload.get("status") == "ready":
                    return (time.perf_counter() - start) * 1000
            except Exception as error:  # noqa: BLE001
                last_error = error
            time.sleep(0.2)

        raise BenchError(f"HTTP server did not become ready within {timeout_s:.1f}s ({last_error})")

    def create(self) -> str:
        payload = self._request("POST", self.base_path)
        sid = payload.get("session_id")
        if not isinstance(sid, str) or not sid:
            raise BenchError(f"Missing session_id in create response: {payload}")
        return sid

    def feed(self, session_id: str, chunk: bytes) -> dict[str, Any]:
        return self._request("POST", f"{self.base_path}/{session_id}", body=chunk)

    def stop(self, session_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"{self.base_path}/{session_id}")

    def close(self) -> None:
        return


class StdioTransportClient:
    def __init__(self, process: subprocess.Popen[bytes], timeout_s: float) -> None:
        if process.stdin is None or process.stdout is None:
            raise BenchError("stdio process missing stdin/stdout pipes")
        self.process = process
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.timeout_s = timeout_s
        self.request_id = 1

    def close(self) -> None:
        try:
            self.stdin.close()
        except Exception:  # noqa: BLE001
            pass

    def _read_exact(self, size: int, timeout_s: float) -> bytes:
        if size == 0:
            return b""

        fd = self.stdout.fileno()
        chunks: list[bytes] = []
        remaining = size
        deadline = time.perf_counter() + timeout_s

        while remaining > 0:
            wait = deadline - time.perf_counter()
            if wait <= 0:
                raise BenchError("Timed out reading stdio frame")

            ready, _, _ = select.select([fd], [], [], wait)
            if not ready:
                raise BenchError("Timed out reading stdio frame")

            chunk = os.read(fd, remaining)
            if not chunk:
                raise BenchError("stdio pipe closed while reading response")

            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)

    def _request(self, command: str, session_id: str | None = None, binary: bytes = b"", timeout_s: float | None = None) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise BenchError(f"stdio server exited with code {self.process.returncode}")

        request_id = self.request_id
        self.request_id += 1

        metadata = {
            "id": request_id,
            "command": command,
        }
        if session_id is not None:
            metadata["session_id"] = session_id

        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        frame = struct.pack(">II", len(metadata_bytes), len(binary)) + metadata_bytes + binary

        self.stdin.write(frame)
        self.stdin.flush()

        response_timeout = timeout_s if timeout_s is not None else self.timeout_s
        header = self._read_exact(8, response_timeout)
        metadata_len, binary_len = struct.unpack(">II", header)
        metadata_payload = self._read_exact(metadata_len, response_timeout)
        if binary_len:
            _ = self._read_exact(binary_len, response_timeout)

        response = json.loads(metadata_payload)
        if response.get("id") != request_id:
            raise BenchError(f"Out-of-order stdio response id={response.get('id')} expected={request_id}")
        if not response.get("ok", False):
            raise BenchError(f"stdio request failed: {response.get('error', 'unknown error')}")

        return response

    def wait_ready(self, timeout_s: float) -> float:
        start = time.perf_counter()
        payload = self._request("info", timeout_s=timeout_s)
        if payload.get("status") != "ready":
            raise BenchError(f"Unexpected stdio info response: {payload}")
        return (time.perf_counter() - start) * 1000

    def create(self) -> str:
        payload = self._request("create")
        sid = payload.get("session_id")
        if not isinstance(sid, str) or not sid:
            raise BenchError(f"Missing session_id in stdio create response: {payload}")
        return sid

    def feed(self, session_id: str, chunk: bytes) -> dict[str, Any]:
        return self._request("feed", session_id=session_id, binary=chunk)

    def stop(self, session_id: str) -> dict[str, Any]:
        return self._request("stop", session_id=session_id)


def format_ms(value: float) -> str:
    return f"{value:7.2f}"


def require_model_dir(path: pathlib.Path) -> pathlib.Path:
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).exists()]
    if missing:
        raise BenchError(f"Model directory missing required files: {', '.join(missing)} ({path})")
    return path


def resolve_default_model() -> pathlib.Path:
    root = pathlib.Path.home() / "Library" / "Application Support" / "Yuwp" / "models"
    candidates = [
        root / "mlx-community--Qwen3-ASR-0.6B-4bit",
        root / "mlx-community--Qwen3-ASR-0.6B-bf16",
        root / "mlx-community--Qwen3-ASR-1.7B-4bit",
        root / "mlx-community--Qwen3-ASR-1.7B-bf16",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return require_model_dir(candidate)
            except BenchError:
                continue
    raise BenchError("Could not auto-resolve a local model. Pass --model <dir>.")


def resolve_server_binary(explicit: str | None) -> pathlib.Path:
    if explicit:
        path = pathlib.Path(explicit).expanduser().resolve()
        if not path.exists():
            raise BenchError(f"Server binary not found: {path}")
        return path

    candidates = [
        pathlib.Path(".build/arm64-apple-macosx/release/yuwp-asr"),
        pathlib.Path(".build/arm64-apple-macosx/debug/yuwp-asr"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise BenchError("Could not find yuwp-asr binary. Build it first with `swift build --product yuwp-asr`.")


def load_wav_pcm16(path: pathlib.Path) -> tuple[bytes, int, float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        if channels != 1:
            raise BenchError(f"Expected mono WAV, got {channels} channels ({path})")
        if sample_rate != 16_000:
            raise BenchError(
                f"Expected 16kHz WAV for streaming benchmark, got {sample_rate}Hz ({path}). "
                "Use a 16k fixture like Tests/fixtures/jfk.wav or asr_zh.wav."
            )
        if sample_width != 2:
            raise BenchError(f"Expected 16-bit WAV, got sample width {sample_width} ({path})")
        pcm = wav.readframes(frames)
    duration_s = frames / float(sample_rate)
    return pcm, sample_rate, duration_s


def split_chunks(pcm: bytes, sample_rate: int, chunk_ms: int) -> list[bytes]:
    samples_per_chunk = int(sample_rate * (chunk_ms / 1000.0))
    if samples_per_chunk <= 0:
        raise BenchError("chunk_ms must be positive")
    bytes_per_chunk = samples_per_chunk * 2
    return [pcm[offset: offset + bytes_per_chunk] for offset in range(0, len(pcm), bytes_per_chunk)]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(
    server_binary: pathlib.Path,
    model_dir: pathlib.Path,
    mode: str,
    *,
    disable_batch_retranscribe: bool,
) -> tuple[subprocess.Popen[bytes], int]:
    port = find_free_port()
    args = [
        str(server_binary),
        "serve",
        "--model",
        str(model_dir),
        "--transport",
        mode,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if disable_batch_retranscribe:
        args.append("--disable-batch-retranscribe")

    env = os.environ.copy()
    env["YUWP_DIAGNOSTIC_LOGGING"] = "0"

    process = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    return process, port


def stop_server(process: subprocess.Popen[bytes], mode: str, client: HTTPTransportClient | StdioTransportClient | None) -> None:
    if mode == "stdio" and client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def run_session(
    client: HTTPTransportClient | StdioTransportClient,
    chunks: list[bytes],
) -> SessionMetrics:
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    session_id = client.create()
    create_ms = (time.perf_counter() - t0) * 1000

    feed_rtts: list[float] = []
    updates_with_text = 0
    for chunk in chunks:
        t_feed = time.perf_counter()
        payload = client.feed(session_id, chunk)
        feed_rtts.append((time.perf_counter() - t_feed) * 1000)
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            updates_with_text += 1

    t_stop = time.perf_counter()
    final_payload = client.stop(session_id)
    stop_ms = (time.perf_counter() - t_stop) * 1000

    total_ms = (time.perf_counter() - t_total_start) * 1000
    final_text = final_payload.get("text")
    final_chars = len(final_text) if isinstance(final_text, str) else 0

    sorted_rtts = sorted(feed_rtts)
    return SessionMetrics(
        create_ms=create_ms,
        feed_mean_ms=statistics.mean(feed_rtts) if feed_rtts else 0.0,
        feed_p50_ms=percentile(sorted_rtts, 0.50),
        feed_p95_ms=percentile(sorted_rtts, 0.95),
        feed_total_ms=sum(feed_rtts),
        stop_ms=stop_ms,
        total_ms=total_ms,
        final_chars=final_chars,
        updates_with_text=updates_with_text,
    )


def benchmark_mode(
    mode: str,
    server_binary: pathlib.Path,
    model_dir: pathlib.Path,
    chunks: list[bytes],
    runs: int,
    ready_timeout_s: float,
    request_timeout_s: float,
    disable_batch_retranscribe: bool,
) -> ModeResult:
    process, port = start_server(
        server_binary,
        model_dir,
        mode,
        disable_batch_retranscribe=disable_batch_retranscribe,
    )

    client: HTTPTransportClient | StdioTransportClient | None = None
    try:
        if mode == "http":
            client = HTTPTransportClient("127.0.0.1", port, timeout_s=request_timeout_s)
        elif mode == "stdio":
            client = StdioTransportClient(process, timeout_s=request_timeout_s)
        else:
            raise BenchError(f"Unknown mode: {mode}")

        startup_ms = client.wait_ready(ready_timeout_s)

        sessions: list[SessionMetrics] = []
        for _ in range(runs):
            sessions.append(run_session(client, chunks))

        return ModeResult(mode=mode, startup_ms=startup_ms, sessions=sessions)
    finally:
        stop_server(process, mode, client)


def avg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize(mode_result: ModeResult) -> dict[str, float]:
    sessions = mode_result.sessions
    return {
        "startup_ms": mode_result.startup_ms,
        "create_ms": avg([s.create_ms for s in sessions]),
        "feed_mean_ms": avg([s.feed_mean_ms for s in sessions]),
        "feed_p50_ms": avg([s.feed_p50_ms for s in sessions]),
        "feed_p95_ms": avg([s.feed_p95_ms for s in sessions]),
        "feed_total_ms": avg([s.feed_total_ms for s in sessions]),
        "stop_ms": avg([s.stop_ms for s in sessions]),
        "total_ms": avg([s.total_ms for s in sessions]),
        "final_chars": avg([float(s.final_chars) for s in sessions]),
    }


def print_results(
    mode_results: list[ModeResult],
    audio_duration_s: float,
    runs: int,
    chunk_ms: int,
) -> None:
    print()
    print(f"Benchmark: {runs} session(s) per transport, chunk={chunk_ms}ms, audio={audio_duration_s:.2f}s")

    summaries = {result.mode: summarize(result) for result in mode_results}

    print()
    print("Per-mode averages (ms):")
    print("mode    startup   create   feed_mean   feed_p50   feed_p95   feed_total   stop    total")
    for mode in ("http", "stdio"):
        if mode not in summaries:
            continue
        s = summaries[mode]
        print(
            f"{mode:5} {format_ms(s['startup_ms'])} {format_ms(s['create_ms'])}"
            f" {format_ms(s['feed_mean_ms'])} {format_ms(s['feed_p50_ms'])} {format_ms(s['feed_p95_ms'])}"
            f" {format_ms(s['feed_total_ms'])} {format_ms(s['stop_ms'])} {format_ms(s['total_ms'])}"
        )

    if "http" in summaries and "stdio" in summaries:
        print()
        print("Delta (stdio vs http):")
        keys = ["startup_ms", "create_ms", "feed_mean_ms", "feed_p95_ms", "feed_total_ms", "stop_ms", "total_ms"]
        for key in keys:
            http_value = summaries["http"][key]
            stdio_value = summaries["stdio"][key]
            delta = stdio_value - http_value
            pct = (delta / http_value * 100.0) if http_value else 0.0
            print(f"  {key:12s} {delta:+8.2f} ms ({pct:+6.1f}%)")

        for mode in ("http", "stdio"):
            rtf = summaries[mode]["total_ms"] / (audio_duration_s * 1000.0)
            print(f"  realtime factor ({mode}): {rtf:.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ASR transport (HTTP vs stdio)")
    parser.add_argument("--audio", default="Tests/fixtures/jfk.wav", help="Path to WAV audio file")
    parser.add_argument("--model", default=None, help="Model directory path (auto-detect if omitted)")
    parser.add_argument("--server-bin", default=None, help="Path to yuwp-asr binary")
    parser.add_argument("--runs", type=int, default=3, help="Sessions per transport")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Audio chunk size in ms")
    parser.add_argument("--ready-timeout", type=float, default=90.0, help="Server ready timeout seconds")
    parser.add_argument("--request-timeout", type=float, default=30.0, help="Per-request timeout seconds")
    parser.add_argument("--mode", choices=["http", "stdio", "both"], default="both", help="Transport mode to run")
    parser.add_argument(
        "--disable-batch-retranscribe",
        action="store_true",
        help="Disable server-side batch retranscribe pass for less variable finalization",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audio_path = pathlib.Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise BenchError(f"Audio file not found: {audio_path}")

    model_dir = require_model_dir(pathlib.Path(args.model).expanduser().resolve()) if args.model else resolve_default_model()
    server_binary = resolve_server_binary(args.server_bin)

    pcm, sample_rate, audio_duration_s = load_wav_pcm16(audio_path)
    chunks = split_chunks(pcm, sample_rate, args.chunk_ms)

    modes = [args.mode] if args.mode in {"http", "stdio"} else ["http", "stdio"]

    print(f"Audio:       {audio_path}")
    print(f"Model:       {model_dir}")
    print(f"Server:      {server_binary}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Chunks:      {len(chunks)} @ {args.chunk_ms}ms")

    mode_results: list[ModeResult] = []
    for mode in modes:
        print()
        print(f"== Benchmarking {mode} ==")
        result = benchmark_mode(
            mode=mode,
            server_binary=server_binary,
            model_dir=model_dir,
            chunks=chunks,
            runs=args.runs,
            ready_timeout_s=args.ready_timeout,
            request_timeout_s=args.request_timeout,
            disable_batch_retranscribe=args.disable_batch_retranscribe,
        )
        mode_results.append(result)
        print(f"startup: {result.startup_ms:.2f}ms")
        for index, session in enumerate(result.sessions, start=1):
            print(
                f"  run {index}: total={session.total_ms:.2f}ms create={session.create_ms:.2f}ms"
                f" feed_mean={session.feed_mean_ms:.2f}ms feed_p95={session.feed_p95_ms:.2f}ms"
                f" stop={session.stop_ms:.2f}ms final_chars={session.final_chars}"
            )

    print_results(mode_results, audio_duration_s=audio_duration_s, runs=args.runs, chunk_ms=args.chunk_ms)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        raise SystemExit(130)
