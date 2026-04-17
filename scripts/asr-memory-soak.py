#!/usr/bin/env -S uv run --python 3.14 --script
"""
Soak-test Yuwp's streaming HTTP server for memory growth.

Why this exists:
- `asr-stream-test` benchmarks in-process StreamingSession behavior.
- This harness exercises the real HTTP server path (`swift-mlx-asr-server`)
  and samples the live server process RSS over time.

Usage examples:
  # 10-minute accelerated soak (no real-time sleeping)
  scripts/asr-memory-soak.py --audio Tests/fixtures/jfk.wav --duration-sec 600

  # 10-minute real-time soak with 2.25s chunk cadence
  scripts/asr-memory-soak.py --audio Tests/fixtures/jfk.wav --duration-sec 600 --realtime

  # Pin to a specific server PID and write JSON report
  scripts/asr-memory-soak.py --server-pid 12345 --output /tmp/asr-memory-soak.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class Sample:
    t_sec: float
    rss_mb: float


@dataclass
class SoakReport:
    host: str
    port: int
    server_pid: int
    audio_path: str
    duration_sec: float
    chunk_sec: float
    realtime: bool
    session_id: str
    chunks_sent: int
    chunks_with_text: int
    first_text_sec: float | None
    final_text_len: int
    rss_start_mb: float
    rss_end_mb: float
    rss_peak_mb: float
    rss_delta_mb: float
    rss_growth_rate_mb_per_min: float
    samples: list[Sample]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soak-test streaming server memory growth")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9748)
    p.add_argument("--audio", type=Path, default=Path("Tests/fixtures/jfk.wav"))
    p.add_argument("--duration-sec", type=float, default=600.0, help="Total soak duration")
    p.add_argument("--chunk-sec", type=float, default=2.25, help="Chunk size to send")
    p.add_argument("--sample-interval-sec", type=float, default=1.0, help="RSS polling interval")
    p.add_argument("--server-pid", type=int, default=None, help="Optional explicit server PID")
    p.add_argument("--realtime", action="store_true", help="Sleep chunk_sec between sends")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON report output")
    p.add_argument("--request-timeout-sec", type=float, default=120.0, help="HTTP request timeout")
    p.add_argument("--max-feed-errors", type=int, default=5, help="Abort if feed timeout/error count exceeds this")
    return p.parse_args()


def http_json(method: str, url: str, body: bytes | None = None, timeout_sec: float = 30.0) -> dict[str, Any]:
    req = Request(url=url, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/octet-stream")
        req.data = body
    with urlopen(req, timeout=timeout_sec) as r:
        raw = r.read()
        return json.loads(raw.decode("utf-8"))


def resolve_server_pid(host: str, port: int) -> int:
    # lsof is available on macOS; use it to find LISTEN pid on given host/port.
    cmd = [
        "lsof",
        "-nP",
        f"-iTCP@{host}:{port}",
        "-sTCP:LISTEN",
        "-t",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if not out:
        raise RuntimeError(f"No LISTEN pid found on {host}:{port}. Is swift-mlx-asr-server running?")
    first = out.splitlines()[0].strip()
    return int(first)


def rss_mb_for_pid(pid: int) -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
    if not out:
        raise RuntimeError(f"Could not read RSS for pid {pid}")
    rss_kb = int(out)
    return rss_kb / 1024.0


def load_wav_pcm16_mono_16k(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        comptype = w.getcomptype()
        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels: {path}")
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sampwidth}: {path}")
        if framerate != 16_000:
            raise ValueError(f"Expected 16kHz WAV, got {framerate}Hz: {path}")
        if comptype != "NONE":
            raise ValueError(f"Expected PCM WAV, got compression {comptype}: {path}")
        return w.readframes(w.getnframes())


def chunk_bytes(pcm: bytes, chunk_sec: float, sample_rate: int = 16_000) -> list[bytes]:
    samples_per_chunk = int(round(chunk_sec * sample_rate))
    bytes_per_chunk = samples_per_chunk * 2  # s16le
    if bytes_per_chunk <= 0:
        raise ValueError("chunk_sec must be > 0")

    chunks: list[bytes] = []
    i = 0
    while i < len(pcm):
        c = pcm[i : i + bytes_per_chunk]
        if len(c) < bytes_per_chunk:
            c = c + b"\x00" * (bytes_per_chunk - len(c))
        chunks.append(c)
        i += bytes_per_chunk

    if not chunks:
        chunks.append(b"\x00" * bytes_per_chunk)
    return chunks


def main() -> int:
    args = parse_args()
    base = f"http://{args.host}:{args.port}"

    try:
        info = http_json("GET", f"{base}/v1/info", timeout_sec=args.request_timeout_sec)
    except (HTTPError, URLError, TimeoutError) as e:
        print(f"[soak] Failed to query /v1/info: {e}", file=sys.stderr)
        return 1

    if info.get("status") != "ready":
        print(f"[soak] Server not ready: {info}", file=sys.stderr)
        return 1

    try:
        server_pid = args.server_pid or resolve_server_pid(args.host, args.port)
    except Exception as e:
        print(f"[soak] Failed to resolve server PID: {e}", file=sys.stderr)
        return 1

    try:
        pcm = load_wav_pcm16_mono_16k(args.audio)
        chunks = chunk_bytes(pcm, args.chunk_sec)
    except Exception as e:
        print(f"[soak] Failed to prepare audio: {e}", file=sys.stderr)
        return 1

    try:
        create = http_json(
            "POST",
            f"{base}/v1/audio/transcriptions/stream",
            body=b"",
            timeout_sec=args.request_timeout_sec,
        )
        sid = create["session_id"]
    except Exception as e:
        print(f"[soak] Failed to create streaming session: {e}", file=sys.stderr)
        return 1

    print(f"[soak] server_pid={server_pid} session_id={sid} chunk_sec={args.chunk_sec} realtime={args.realtime}")

    stop_event = threading.Event()
    start_t = time.monotonic()
    samples: list[Sample] = []
    samples_lock = threading.Lock()

    def sampler() -> None:
        while not stop_event.is_set():
            t = time.monotonic() - start_t
            try:
                rss = rss_mb_for_pid(server_pid)
            except Exception:
                break
            with samples_lock:
                samples.append(Sample(t_sec=t, rss_mb=rss))
            time.sleep(args.sample_interval_sec)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    chunks_sent = 0
    chunks_with_text = 0
    first_text_sec: float | None = None
    chunk_idx = 0
    feed_errors = 0

    try:
        end_t = start_t + args.duration_sec
        while time.monotonic() < end_t:
            body = chunks[chunk_idx % len(chunks)]
            chunk_idx += 1
            try:
                resp = http_json(
                    "POST",
                    f"{base}/v1/audio/transcriptions/stream/{sid}",
                    body=body,
                    timeout_sec=args.request_timeout_sec,
                )
            except Exception as e:
                feed_errors += 1
                print(f"[soak] feed error #{feed_errors}: {e}", file=sys.stderr)
                if feed_errors > args.max_feed_errors:
                    raise RuntimeError(
                        f"Too many feed errors ({feed_errors} > {args.max_feed_errors})"
                    ) from e
                continue
            chunks_sent += 1

            text = str(resp.get("text", "")).strip()
            if text:
                chunks_with_text += 1
                if first_text_sec is None:
                    first_text_sec = time.monotonic() - start_t

            if chunks_sent % 20 == 0:
                with samples_lock:
                    current_rss = samples[-1].rss_mb if samples else float("nan")
                    peak_rss = max((s.rss_mb for s in samples), default=float("nan"))
                print(f"[soak] chunks={chunks_sent} rss_now={current_rss:.1f}MB rss_peak={peak_rss:.1f}MB")

            if args.realtime:
                time.sleep(args.chunk_sec)

        final_resp = http_json(
            "DELETE",
            f"{base}/v1/audio/transcriptions/stream/{sid}",
            timeout_sec=args.request_timeout_sec,
        )
        final_text = str(final_resp.get("text", ""))

    except KeyboardInterrupt:
        print("\n[soak] Interrupted, stopping session...", file=sys.stderr)
        try:
            final_resp = http_json(
                "DELETE",
                f"{base}/v1/audio/transcriptions/stream/{sid}",
                timeout_sec=args.request_timeout_sec,
            )
            final_text = str(final_resp.get("text", ""))
        except Exception:
            final_text = ""
    except Exception as e:
        print(f"[soak] Error during soak: {e}", file=sys.stderr)
        try:
            http_json(
                "DELETE",
                f"{base}/v1/audio/transcriptions/stream/{sid}",
                timeout_sec=args.request_timeout_sec,
            )
        except Exception:
            pass
        return 1
    finally:
        stop_event.set()
        th.join(timeout=2)

    with samples_lock:
        final_samples = list(samples)

    if not final_samples:
        print("[soak] No RSS samples collected", file=sys.stderr)
        return 1

    rss_start = final_samples[0].rss_mb
    rss_end = final_samples[-1].rss_mb
    rss_peak = max(s.rss_mb for s in final_samples)
    rss_delta = rss_end - rss_start
    duration_min = max(args.duration_sec / 60.0, 1e-9)

    report = SoakReport(
        host=args.host,
        port=args.port,
        server_pid=server_pid,
        audio_path=str(args.audio),
        duration_sec=args.duration_sec,
        chunk_sec=args.chunk_sec,
        realtime=args.realtime,
        session_id=sid,
        chunks_sent=chunks_sent,
        chunks_with_text=chunks_with_text,
        first_text_sec=first_text_sec,
        final_text_len=len(final_text),
        rss_start_mb=rss_start,
        rss_end_mb=rss_end,
        rss_peak_mb=rss_peak,
        rss_delta_mb=rss_delta,
        rss_growth_rate_mb_per_min=rss_delta / duration_min,
        samples=final_samples,
    )

    summary = {
        "chunks_sent": report.chunks_sent,
        "first_text_sec": report.first_text_sec,
        "final_text_len": report.final_text_len,
        "rss_start_mb": round(report.rss_start_mb, 1),
        "rss_end_mb": round(report.rss_end_mb, 1),
        "rss_peak_mb": round(report.rss_peak_mb, 1),
        "rss_delta_mb": round(report.rss_delta_mb, 1),
        "rss_growth_rate_mb_per_min": round(report.rss_growth_rate_mb_per_min, 2),
    }
    print("[soak] summary=" + json.dumps(summary))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        payload["samples"] = [asdict(s) for s in report.samples]
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[soak] wrote report: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
