#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Compare Yuwp ASR against an OpenAI-compatible remote ASR endpoint.

This benchmark is meant for side-by-side checks against services like oMLX.
It probes endpoint capabilities first, then runs batch and optional streaming
measurements on the same audio corpus.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

SAMPLE_RATE = 16_000
CHUNK_SEC = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)
REQUEST_TIMEOUT = 180.0
READY_TIMEOUT = 240.0
READY_POLL_SEC = 0.5
DEFAULT_CORPUS_ROOT = Path.home() / "Library" / "Application Support" / "Yuwp" / "recordings"
DEFAULT_JSON_PATH = "/tmp/yuwp-vs-openai-asr.json"
DEFAULT_LOCAL_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_REMOTE_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
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
    pcm_bytes: bytes
    sample_count: int
    saved_text: str


@dataclass(slots=True)
class PreparedModel:
    spec: str
    repo_id: str | None
    resolved_path: Path
    label: str


@dataclass(slots=True)
class EndpointProbe:
    name: str
    base_url: str
    models_url: str | None
    health_url: str | None
    has_batch: bool
    has_stream: bool
    models: list[str]
    openapi_error: str | None = None


def looks_like_path(spec: str) -> bool:
    return spec.startswith("/") or spec.startswith("~") or spec.startswith(".")


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


def find_hf_snapshot(repo_id: str) -> Path | None:
    owner, name = repo_id.split("/", 1)
    for root in hf_roots():
        candidate = root / f"models--{owner}--{name}" / "snapshots"
        if not candidate.is_dir():
            continue
        snaps = sorted((p for p in candidate.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snaps:
            if is_valid_model_dir(snap):
                return snap
    return None


def resolve_model_spec(spec: str) -> PreparedModel:
    raw = spec.strip()
    alias = MODEL_ALIASES.get(raw, raw)
    if looks_like_path(alias):
        path = Path(alias).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"model directory not found: {path}")
        return PreparedModel(spec=raw, repo_id=None, resolved_path=path, label=path.name)
    repo_id = alias
    path = find_hf_snapshot(repo_id)
    if path is None:
        raise FileNotFoundError(f"model not found in Hugging Face cache: {repo_id}")
    return PreparedModel(spec=raw, repo_id=repo_id, resolved_path=path, label=repo_id.split("/")[-1])


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


def decode_audio(path: Path) -> bytes:
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"
    ], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {result.stderr.decode(errors='ignore')[:200]}")
    return result.stdout


def collect_audio_paths(args: argparse.Namespace) -> list[Path]:
    if args.files:
        return [Path(p).expanduser().resolve() for p in args.files]
    root = Path(args.root).expanduser()
    globs = [str(root / "**/*.flac"), str(root / "**/*.wav")]
    candidates = sorted({Path(p).resolve() for pattern in globs for p in glob.glob(pattern, recursive=True)})
    if args.date:
        needle = "/".join(args.date.split("-"))
        candidates = [p for p in candidates if needle in str(p)]
    if args.last is not None:
        return candidates[-args.last :]
    return candidates[-10:]


def prepare_cases(paths: list[Path]) -> list[AudioCase]:
    cases: list[AudioCase] = []
    for p in paths:
        pcm_bytes = decode_audio(p)
        sample_count = len(pcm_bytes) // 2
        cases.append(AudioCase(path=p, duration_s=sample_count / SAMPLE_RATE, pcm_bytes=pcm_bytes, sample_count=sample_count, saved_text=load_saved_text(p)))
    return cases


def find_server_binary() -> Path:
    for candidate in [
        Path(".build/arm64-apple-macosx/release/yuwp-asr"),
        Path(".build/release/yuwp-asr"),
    ]:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("server binary not found")


def http_json(method: str, url: str, *, fields: dict[str, str] | None = None, files: dict[str, tuple[str, bytes, str]] | None = None, body: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = REQUEST_TIMEOUT) -> tuple[int, dict[str, Any]]:
    req_headers = dict(headers or {})
    data: bytes
    if files is not None:
        boundary = f"----PiBench{int(time.time() * 1000)}"
        req_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        chunks: list[bytes] = []
        for key, value in (fields or {}).items():
            chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
        for key, (filename, file_bytes, content_type) in files.items():
            chunks.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
                + file_bytes + b"\r\n"
            )
        chunks.append(f"--{boundary}--\r\n".encode())
        data = b"".join(chunks)
    elif body is not None:
        data = body
    elif fields is not None:
        req_headers["Content-Type"] = "application/json"
        data = json.dumps(fields).encode()
    else:
        data = b""
    req = Request(url, data=data, headers=req_headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode() or "{}"
            return resp.status, json.loads(payload)
    except Exception as exc:
        status = getattr(exc, "code", 0)
        raw = exc.read().decode() if hasattr(exc, "read") else json.dumps({"error": str(exc)})
        try:
            return status, json.loads(raw)
        except Exception:
            return status, {"error": raw or str(exc)}


def wait_for_ready(port: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/info"
    deadline = time.perf_counter() + READY_TIMEOUT
    while time.perf_counter() < deadline:
        status, payload = http_json("GET", url, timeout=2)
        if status == 200 and payload.get("status") == "ready":
            return payload
        time.sleep(READY_POLL_SEC)
    raise TimeoutError(f"server not ready on port {port}")


def launch_local_server(model: PreparedModel, port: int, warmup: bool) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    server_bin = find_server_binary()
    cmd = [
        str(server_bin),
        "serve",
        "--transport",
        "http",
        "--model",
        str(model.resolved_path),
        "--batch-model",
        str(model.resolved_path),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    if warmup:
        cmd.append("--warmup")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        info = wait_for_ready(port)
        return proc, info
    except Exception:
        terminate_process(proc)
        raise


def terminate_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def normalize_url(base_url: str) -> str:
    return base_url.rstrip("/")


def fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=5) as resp:
        return json.load(resp)


def probe_endpoint(name: str, base_url: str) -> EndpointProbe:
    base_url = normalize_url(base_url)
    has_batch = False
    has_stream = False
    openapi_error = None
    info_payload: dict[str, Any] | None = None
    try:
        spec = fetch_json(base_url + "/openapi.json")
        paths = spec.get("paths", {})
        has_batch = "/v1/audio/transcriptions" in paths
        has_stream = "/v1/audio/transcriptions/stream" in paths
    except Exception as exc:
        openapi_error = str(exc)
    models: list[str] = []
    models_url = None
    for candidate in [base_url + "/v1/models", base_url + "/models"]:
        try:
            payload = fetch_json(candidate)
            models = [item.get("id", "") for item in payload.get("data", []) if item.get("id")]
            models_url = candidate
            break
        except Exception:
            pass
    health_url = None
    for candidate in [base_url + "/v1/info", base_url + "/health"]:
        try:
            payload = fetch_json(candidate)
            health_url = candidate
            if candidate.endswith('/v1/info'):
                info_payload = payload
            break
        except Exception:
            pass
    if info_payload and info_payload.get("status") == "ready":
        has_batch = True
        has_stream = True
        model_id = info_payload.get("model_id") or info_payload.get("active_model_id")
        if model_id and model_id not in models:
            models.append(model_id)
    return EndpointProbe(name=name, base_url=base_url, models_url=models_url, health_url=health_url, has_batch=has_batch, has_stream=has_stream, models=models, openapi_error=openapi_error)


def batch_transcribe(base_url: str, model: str, case: AudioCase, verbose: bool = False) -> dict[str, Any]:
    if verbose:
        print(f"    [batch] start {case.path.name} ({case.duration_s:.1f}s)", flush=True)
    t0 = time.perf_counter()
    status, body = http_json("POST", base_url + "/v1/audio/transcriptions", fields={"model": model}, files={"file": (case.path.name, case.path.read_bytes(), "audio/wav")})
    elapsed = time.perf_counter() - t0
    if status < 200 or status >= 300:
        if verbose:
            print(f"    [batch] fail {case.path.name} status={status} elapsed={elapsed:.2f}s", flush=True)
        return {"ok": False, "status": status, "elapsed_s": elapsed, "error": body}
    text = (body.get("text") or "").strip()
    if verbose:
        print(f"    [batch] done {case.path.name} elapsed={elapsed:.2f}s rtf={elapsed / case.duration_s if case.duration_s else 0.0:.3f}", flush=True)
    return {"ok": True, "status": status, "elapsed_s": elapsed, "text": text, "rtf": elapsed / case.duration_s if case.duration_s else 0.0}


def stream_transcribe(base_url: str, model: str, case: AudioCase, verbose: bool = False) -> dict[str, Any]:
    if verbose:
        print(f"    [stream] create {case.path.name} ({case.duration_s:.1f}s)", flush=True)
    status, create_body = http_json("POST", base_url + "/v1/audio/transcriptions/stream", fields={"model": model})
    if status < 200 or status >= 300:
        if verbose:
            print(f"    [stream] create fail {case.path.name} status={status}", flush=True)
        return {"ok": False, "status": status, "error": create_body}
    session_id = create_body.get("session_id")
    if not session_id:
        return {"ok": False, "status": status, "error": {"message": "missing session_id"}}
    text = ""
    first_nonempty_ms = None
    chunk_latencies_ms: list[float] = []
    start = time.perf_counter()
    total_chunks = max(1, (case.sample_count + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES)
    for chunk_index, idx in enumerate(range(0, case.sample_count, CHUNK_SAMPLES), start=1):
        start_byte = idx * 2
        end_byte = min((idx + CHUNK_SAMPLES) * 2, len(case.pcm_bytes))
        chunk = case.pcm_bytes[start_byte:end_byte]
        if verbose:
            print(f"    [stream] chunk {chunk_index}/{total_chunks} {case.path.name}", flush=True)
        t0 = time.perf_counter()
        status, feed_body = http_json("POST", base_url + f"/v1/audio/transcriptions/stream/{session_id}", body=chunk, headers={"Content-Type": "application/octet-stream"})
        chunk_ms = (time.perf_counter() - t0) * 1000
        chunk_latencies_ms.append(chunk_ms)
        if status < 200 or status >= 300:
            if verbose:
                print(f"    [stream] chunk fail {chunk_index}/{total_chunks} status={status}", flush=True)
            return {"ok": False, "status": status, "error": feed_body}
        text = feed_body.get("text", text)
        if first_nonempty_ms is None and text.strip():
            first_nonempty_ms = (time.perf_counter() - start) * 1000
        if verbose:
            chars = len(text.strip())
            print(f"    [stream] chunk ok {chunk_index}/{total_chunks} {chunk_ms:.1f}ms text_chars={chars}", flush=True)
    if verbose:
        print(f"    [stream] finalize {case.path.name}", flush=True)
    status, stop_body = http_json("DELETE", base_url + f"/v1/audio/transcriptions/stream/{session_id}")
    if status < 200 or status >= 300:
        if verbose:
            print(f"    [stream] finalize fail {case.path.name} status={status}", flush=True)
        return {"ok": False, "status": status, "error": stop_body}
    final_text = (stop_body.get("text") or text).strip()
    total_s = time.perf_counter() - start
    if verbose:
        print(f"    [stream] done {case.path.name} elapsed={total_s:.2f}s rtf={total_s / case.duration_s if case.duration_s else 0.0:.3f}", flush=True)
    return {
        "ok": True,
        "status": status,
        "text": final_text,
        "elapsed_s": total_s,
        "rtf": total_s / case.duration_s if case.duration_s else 0.0,
        "first_nonempty_ms": first_nonempty_ms,
        "avg_chunk_ms": sum(chunk_latencies_ms) / len(chunk_latencies_ms) if chunk_latencies_ms else 0.0,
    }


def resolve_remote_request_model(requested: str, available: list[str]) -> str:
    if not available:
        return requested
    if requested in available:
        return requested
    requested_lower = requested.lower()
    alias = MODEL_ALIASES.get(requested, requested)
    alias_lower = alias.lower()
    for candidate in available:
        c = candidate.lower()
        if c == requested_lower or c == alias_lower:
            return candidate
        if requested_lower in c or alias_lower in c:
            return candidate
        if requested_lower.split('/')[-1] == c or alias_lower.split('/')[-1] == c:
            return candidate
    return requested


def summarize(label: str, probe: EndpointProbe, batch_results: list[dict[str, Any]], stream_results: list[dict[str, Any]]) -> dict[str, Any]:
    batch_ok = [r for r in batch_results if r.get("ok")]
    stream_ok = [r for r in stream_results if r.get("ok")]
    return {
        "label": label,
        "capabilities": {
            "models_url": probe.models_url,
            "health_url": probe.health_url,
            "has_batch": probe.has_batch,
            "has_stream": probe.has_stream,
            "models": probe.models,
            "openapi_error": probe.openapi_error,
        },
        "batch": {
            "successes": len(batch_ok),
            "failures": len(batch_results) - len(batch_ok),
            "avg_elapsed_s": sum(r["elapsed_s"] for r in batch_ok) / len(batch_ok) if batch_ok else None,
            "avg_rtf": sum(r["rtf"] for r in batch_ok) / len(batch_ok) if batch_ok else None,
            "errors": [r for r in batch_results if not r.get("ok")],
        },
        "stream": {
            "successes": len(stream_ok),
            "failures": len(stream_results) - len(stream_ok),
            "avg_elapsed_s": sum(r["elapsed_s"] for r in stream_ok) / len(stream_ok) if stream_ok else None,
            "avg_rtf": sum(r["rtf"] for r in stream_ok) / len(stream_ok) if stream_ok else None,
            "avg_first_nonempty_ms": sum(r["first_nonempty_ms"] for r in stream_ok if r.get("first_nonempty_ms") is not None) / max(1, len([r for r in stream_ok if r.get("first_nonempty_ms") is not None])) if stream_ok else None,
            "errors": [r for r in stream_results if not r.get("ok")],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local Yuwp ASR against an OpenAI-compatible ASR endpoint.")
    parser.add_argument("--remote-url", default="http://127.0.0.1:8400", help="Remote OpenAI-compatible base URL")
    parser.add_argument("--remote-model", default=DEFAULT_REMOTE_MODEL, help="Remote model id to request")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL, help="Local Yuwp model spec or path")
    parser.add_argument("--local-port", type=int, default=9791)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument("--files", nargs="+", help="Explicit audio files")
    parser.add_argument("--root", default=str(DEFAULT_CORPUS_ROOT))
    parser.add_argument("--last", type=int, default=3)
    parser.add_argument("--date")
    parser.add_argument("--json", default=DEFAULT_JSON_PATH)
    parser.add_argument("--verbose", action="store_true", help="Print per-file and per-chunk progress logs")
    return parser.parse_args()


def print_probe(probe: EndpointProbe) -> None:
    print(f"{probe.name}: {probe.base_url}")
    print(f"  batch:   {'yes' if probe.has_batch else 'no'}")
    print(f"  stream:  {'yes' if probe.has_stream else 'no'}")
    print(f"  models:  {', '.join(probe.models) if probe.models else '(none listed)'}")
    if probe.openapi_error:
        print(f"  openapi: {probe.openapi_error}")


def main() -> None:
    args = parse_args()
    paths = collect_audio_paths(args)
    if not paths:
        raise SystemExit("no audio files found")
    cases = prepare_cases(paths)

    local_proc: subprocess.Popen[str] | None = None
    output: dict[str, Any] = {"cases": [str(c.path) for c in cases], "results": {}}
    try:
        if not args.skip_local:
            local_model = resolve_model_spec(args.local_model)
            local_proc, local_info = launch_local_server(local_model, args.local_port, args.warmup)
            local_probe = probe_endpoint("local", f"http://127.0.0.1:{args.local_port}")
            print_probe(local_probe)
            request_model = local_info.get("model_id") or local_info.get("active_model_id") or "qwen3-asr-0.6b"
            batch_results = [batch_transcribe(local_probe.base_url, request_model, case, verbose=args.verbose) for case in cases]
            stream_results = [stream_transcribe(local_probe.base_url, request_model, case, verbose=args.verbose) for case in cases] if local_probe.has_stream else []
            output["results"]["local"] = summarize("local", local_probe, batch_results, stream_results)
            output["results"]["local"]["request_model"] = request_model

        if not args.skip_remote:
            remote_probe = probe_endpoint("remote", args.remote_url)
            print_probe(remote_probe)
            remote_request_model = resolve_remote_request_model(args.remote_model, remote_probe.models)
            if args.verbose and remote_request_model != args.remote_model:
                print(f"  remote model: {args.remote_model} -> {remote_request_model}", flush=True)
            batch_results = [batch_transcribe(remote_probe.base_url, remote_request_model, case, verbose=args.verbose) for case in cases] if remote_probe.has_batch else []
            stream_results = [stream_transcribe(remote_probe.base_url, remote_request_model, case, verbose=args.verbose) for case in cases] if remote_probe.has_stream else []
            output["results"]["remote"] = summarize("remote", remote_probe, batch_results, stream_results)
            output["results"]["remote"]["request_model"] = remote_request_model

        if "local" in output["results"] and "remote" in output["results"]:
            l = output["results"]["local"]
            r = output["results"]["remote"]
            output["comparison"] = {
                "batch_rtf_ratio_remote_over_local": (r["batch"]["avg_rtf"] / l["batch"]["avg_rtf"]) if l["batch"]["avg_rtf"] and r["batch"]["avg_rtf"] else None,
                "stream_rtf_ratio_remote_over_local": (r["stream"]["avg_rtf"] / l["stream"]["avg_rtf"]) if l["stream"]["avg_rtf"] and r["stream"]["avg_rtf"] else None,
            }

        out = Path(args.json).expanduser()
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2))
        print(f"\nWrote {out}")
        print(json.dumps(output.get("comparison", {}), indent=2))
    finally:
        stderr_text = None
        if local_proc and local_proc.stderr:
            terminate_process(local_proc)
            stderr_text = local_proc.stderr.read().strip()
        if stderr_text:
            print("\nlocal stderr tail:")
            print("\n".join(stderr_text.splitlines()[-8:]))


if __name__ == "__main__":
    main()
