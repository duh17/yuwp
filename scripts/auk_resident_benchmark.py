#!/usr/bin/env python3
"""Resident yuwp-tts HTTP latency for Qwen3-TTS, AuK-Flash, and AuK Base.

Measures load vs generation separately. Do not compare CLI cold start to these numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def rss_kb(pid: int) -> int:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out or "0")
    except (subprocess.CalledProcessError, ValueError):
        return 0


def wait_info(url: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"server never became ready: {last}")


def stream_speech(url: str, body: dict, timeout: float) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            elapsed = time.perf_counter() - t0
            line = raw.decode().strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_client_elapsed"] = elapsed
            events.append(obj)
    audio = [e for e in events if e.get("event") == "audio"]
    done = next(e for e in events if e.get("event") == "done")
    return {
        "events": [e.get("event") for e in events],
        "metadata": next((e for e in events if e.get("event") == "metadata"), {}),
        "first_audio_client_seconds": audio[0]["_client_elapsed"] if audio else None,
        "done_client_seconds": done["_client_elapsed"],
        "first_audio_seconds": done.get("first_audio_seconds"),
        "audio_duration_seconds": done.get("audio_duration_seconds"),
        "wall_seconds": done.get("wall_seconds"),
        "chunks": done.get("chunks"),
        "audio_events": len(audio),
    }


def run_one(args: argparse.Namespace, spec: dict) -> dict:
    log_path = Path(args.out_dir) / f"{spec['name']}-server.log"
    cmd = [
        args.tts_bin,
        "serve",
        "--transport",
        "http",
        "--host",
        args.host,
        "--port",
        str(spec["port"]),
        "--model",
        spec["model"],
    ]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        info = wait_info(f"http://{args.host}:{spec['port']}/v1/info", timeout=args.load_timeout)
        time.sleep(0.5)
        rss_before = rss_kb(proc.pid)
        peak = {"kb": rss_before}

        def watch() -> None:
            while proc.poll() is None and peak.get("run", True):
                peak["kb"] = max(peak["kb"], rss_kb(proc.pid))
                time.sleep(0.2)

        peak["run"] = True
        import threading

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        gen = stream_speech(
            f"http://{args.host}:{spec['port']}/v1/audio/speech/stream",
            spec["body"],
            timeout=args.gen_timeout,
        )
        peak["run"] = False
        watcher.join(timeout=1)
        rss_after = rss_kb(proc.pid)
        peak_kb = max(peak["kb"], rss_after)
        duration = float(gen["audio_duration_seconds"] or 0)
        wall = float(gen["wall_seconds"] or gen["done_client_seconds"] or 0)
        return {
            "name": spec["name"],
            "model": spec["model"],
            "info": info,
            "load_seconds": info.get("load_seconds"),
            "backend": info.get("backend"),
            "variant": info.get("variant"),
            "generation": gen,
            "rss_before_gen_kb": rss_before,
            "peak_rss_kb": peak_kb,
            "peak_rss_gb": round(peak_kb / 1024 / 1024, 3),
            "rtf": (wall / duration) if duration else None,
        }
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-bin", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--load-timeout", type=float, default=180)
    parser.add_argument("--gen-timeout", type=float, default=600)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--flash-model", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--port-qwen", type=int, default=17941)
    parser.add_argument("--port-flash", type=int, default=17942)
    parser.add_argument("--port-base", type=int, default=17943)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sentence = "The sky is blue and the sun is bright."
    specs = [
        {
            "name": "qwen3-tts",
            "model": args.qwen_model,
            "port": args.port_qwen,
            "body": {
                "input": sentence,
                "voice": "A clear, calm adult male narrator.",
            },
        },
        {
            "name": "auk-flash",
            "model": args.flash_model,
            "port": args.port_flash,
            "body": {
                "instruction": f"Say the following with the same voice: '{sentence}'",
                "ref_audio": args.ref_audio,
                "gen_seconds": 4,
            },
        },
        {
            "name": "auk-base",
            "model": args.base_model,
            "port": args.port_base,
            "body": {
                "instruction": f"Say the following with the same voice: '{sentence}'",
                "ref_audio": args.ref_audio,
                "gen_seconds": 4,
            },
        },
    ]
    rows = [run_one(args, spec) for spec in specs]
    out = {
        "machine": "M3 Ultra",
        "note": "Resident HTTP server. load_seconds is model load; generation metrics exclude process start.",
        "rows": rows,
    }
    json_path = Path(args.out_dir) / "resident-latency.json"
    json_path.write_text(json.dumps(out, indent=2) + "\n")
    md = ["# Resident TTS latency (M3 Ultra)", "", "| Model | Load s | First audio s | Wall s | Audio s | RTF | Peak RSS GB |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        gen = row["generation"]
        md.append(
            "| {name} | {load:.2f} | {first:.2f} | {wall:.2f} | {dur:.2f} | {rtf:.2f} | {mem:.2f} |".format(
                name=row["name"],
                load=float(row["load_seconds"] or 0),
                first=float(gen["first_audio_seconds"] if gen["first_audio_seconds"] not in (None, -1) else gen["first_audio_client_seconds"] or 0),
                wall=float(gen["wall_seconds"] or gen["done_client_seconds"] or 0),
                dur=float(gen["audio_duration_seconds"] or 0),
                rtf=float(row["rtf"] or 0),
                mem=float(row["peak_rss_gb"] or 0),
            )
        )
    md.append("")
    md.append(f"Source: `{json_path}`.")
    (Path(args.out_dir) / "resident-latency.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
