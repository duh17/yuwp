#!/usr/bin/env -S uv run --python 3.14 --script
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path.home() / "workspace" / "yuwp"
APP = Path("/Applications/Yuwp.app")
APP_BIN = APP / "Contents" / "MacOS" / "Yuwp"
SERVER_BIN = APP / "Contents" / "MacOS" / "asr-server"
METALLIB = APP / "Contents" / "MacOS" / "mlx.metallib"
SPARKLE_FW = APP / "Contents" / "Frameworks" / "Sparkle.framework"
LOGFILE = Path("/tmp/yuwp.log")
DIAGNOSTICS_DIR = Path.home() / "Library" / "Logs" / "DiagnosticReports"
TCC_DB = Path.home() / "Library" / "Application Support" / "com.apple.TCC" / "TCC.db"
MODELS_DIR = Path.home() / "Library" / "Application Support" / "Yuwp" / "models"
RECORDINGS_DIR = Path.home() / "Library" / "Application Support" / "Yuwp" / "recordings"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
RELEASE_DIR = REPO_ROOT / "release"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9748
APP_DOMAIN = "com.yuwp.app"


def run(cmd: list[str], timeout: int = 20, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def bash(script: str, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return run(["bash", "-lc", script], timeout=timeout)


def die(message: str, exit_code: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def latest_crash_paths() -> dict[str, str | None]:
    result: dict[str, str | None] = {"Yuwp": None, "asr-server": None}
    if not DIAGNOSTICS_DIR.exists():
        return result
    for prefix in result:
        matches = sorted(DIAGNOSTICS_DIR.glob(f"{prefix}-*.ips"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            result[prefix] = str(matches[0])
    return result


def read_codesign() -> dict[str, Any]:
    if not APP.exists():
        return {"present": False, "mode": "missing"}
    proc = run(["codesign", "-dv", "--verbose=4", str(APP)], timeout=20)
    text = (proc.stderr or "") + (proc.stdout or "")
    authority = None
    team_id = None
    signature = None
    runtime = False
    for line in text.splitlines():
        if line.startswith("Authority=") and authority is None:
            authority = line.split("=", 1)[1].strip()
        elif line.startswith("TeamIdentifier="):
            team_id = line.split("=", 1)[1].strip()
        elif line.startswith("Signature="):
            signature = line.split("=", 1)[1].strip()
        elif line.startswith("CodeDirectory") and "runtime" in line:
            runtime = True
    mode = "developer_id" if authority and authority.startswith("Developer ID Application:") else "adhoc"
    if signature == "adhoc":
        mode = "adhoc"
    return {
        "present": True,
        "authority": authority,
        "team_id": team_id,
        "signature": signature,
        "runtime": runtime,
        "mode": mode,
        "raw": text.strip(),
    }


def pid_for(path_fragment: str) -> int | None:
    proc = run(["pgrep", "-f", path_fragment], timeout=10)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    line = proc.stdout.strip().splitlines()[0]
    try:
        return int(line.strip())
    except ValueError:
        return None


def process_info() -> dict[str, Any]:
    app_pid = pid_for(str(APP_BIN))
    server_pid = pid_for(str(SERVER_BIN))
    server_ppid = None
    if server_pid is not None:
        proc = run(["ps", "-p", str(server_pid), "-o", "ppid="], timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                server_ppid = int(proc.stdout.strip())
            except ValueError:
                server_ppid = None
    return {
        "app_pid": app_pid,
        "server_pid": server_pid,
        "server_ppid": server_ppid,
        "app_running": app_pid is not None,
        "server_running": server_pid is not None,
    }


def health_info() -> dict[str, Any] | None:
    proc = run(["curl", "-sf", f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1/info"], timeout=10)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"raw": proc.stdout.strip()}


def tcc_rows() -> list[dict[str, Any]]:
    if not TCC_DB.exists():
        return []
    query = (
        "select service,client,client_type,auth_value,auth_reason,flags,last_modified "
        f"from access where client='{APP_DOMAIN}' order by service;"
    )
    proc = run(["sqlite3", str(TCC_DB), query], timeout=10)
    rows: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return rows
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 7:
            continue
        rows.append(
            {
                "service": parts[0],
                "client": parts[1],
                "client_type": int(parts[2]) if parts[2] else None,
                "auth_value": int(parts[3]) if parts[3] else None,
                "auth_reason": int(parts[4]) if parts[4] else None,
                "flags": int(parts[5]) if parts[5] else None,
                "last_modified": int(parts[6]) if parts[6] else None,
            }
        )
    return rows


def defaults_info() -> dict[str, Any] | None:
    script = (
        f"defaults export {APP_DOMAIN} - 2>/dev/null | "
        "plutil -convert json -o - - 2>/dev/null || true"
    )
    proc = bash(script, timeout=10)
    text = proc.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def path_rows() -> list[dict[str, Any]]:
    return [
        {"label": "repo_root", "path": str(REPO_ROOT), "exists": REPO_ROOT.exists()},
        {"label": "app_bundle", "path": str(APP), "exists": APP.exists()},
        {"label": "app_binary", "path": str(APP_BIN), "exists": APP_BIN.exists()},
        {"label": "server_binary", "path": str(SERVER_BIN), "exists": SERVER_BIN.exists()},
        {"label": "metallib", "path": str(METALLIB), "exists": METALLIB.exists()},
        {"label": "sparkle_framework", "path": str(SPARKLE_FW), "exists": SPARKLE_FW.exists()},
        {"label": "logfile", "path": str(LOGFILE), "exists": LOGFILE.exists()},
        {"label": "diagnostics_dir", "path": str(DIAGNOSTICS_DIR), "exists": DIAGNOSTICS_DIR.exists()},
        {"label": "tcc_db", "path": str(TCC_DB), "exists": TCC_DB.exists()},
        {"label": "models_dir", "path": str(MODELS_DIR), "exists": MODELS_DIR.exists()},
        {"label": "recordings_dir", "path": str(RECORDINGS_DIR), "exists": RECORDINGS_DIR.exists()},
        {"label": "hf_cache", "path": str(HF_CACHE), "exists": HF_CACHE.exists()},
        {"label": "release_dir", "path": str(RELEASE_DIR), "exists": RELEASE_DIR.exists()},
    ]


def collect_status() -> dict[str, Any]:
    signing = read_codesign()
    procs = process_info()
    health = health_info()
    tcc = tcc_rows()
    defaults = defaults_info()
    crashes = latest_crash_paths()
    accessibility_granted = any(r["service"] == "kTCCServiceAccessibility" and r["auth_value"] == 2 for r in tcc)
    microphone_granted = any(r["service"] == "kTCCServiceMicrophone" and r["auth_value"] == 2 for r in tcc)
    health_ready = isinstance(health, dict) and health.get("status") == "ready"
    return {
        "summary": {
            "app_running": procs["app_running"],
            "server_running": procs["server_running"],
            "health_ready": health_ready,
            "accessibility_granted": accessibility_granted,
            "microphone_granted": microphone_granted,
            "sign_mode": signing.get("mode"),
            "latest_crash_present": any(crashes.values()),
        },
        "paths": path_rows(),
        "signing": signing,
        "processes": procs,
        "health": health,
        "tcc": tcc,
        "defaults": defaults,
        "latest_crashes": crashes,
    }


def tail_lines(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    proc = run(["tail", "-n", str(lines), str(path)], timeout=10)
    return proc.stdout.splitlines()


def unified_lines(window: str, lines: int) -> list[str]:
    predicate = 'process == "Yuwp" OR process == "asr-server"'
    proc = run(
        [
            "log",
            "show",
            "--style",
            "compact",
            "--last",
            window,
            "--predicate",
            predicate,
        ],
        timeout=40,
    )
    if proc.returncode != 0:
        return []
    out_lines = proc.stdout.splitlines()
    return out_lines[-lines:]


def render_human_status(data: dict[str, Any]) -> str:
    parts = []
    summary = data["summary"]
    signing = data["signing"]
    procs = data["processes"]
    health = data["health"]
    crashes = data["latest_crashes"]
    parts.append("Yuwp status")
    parts.append("")
    parts.append(f"repo:          {REPO_ROOT}")
    parts.append(f"app:           {APP}")
    parts.append(f"signing:       {signing.get('authority') or signing.get('signature') or 'missing'}")
    parts.append(f"team_id:       {signing.get('team_id') or '-'}")
    parts.append(f"app_pid:       {procs.get('app_pid') or '-'}")
    parts.append(f"server_pid:    {procs.get('server_pid') or '-'}")
    parts.append(f"server_ppid:   {procs.get('server_ppid') or '-'}")
    if isinstance(health, dict):
        parts.append(f"health:        {health.get('status', 'unknown')} @ http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1/info")
        if health.get("streaming_model"):
            parts.append(f"stream_model:  {health.get('streaming_model')}")
        if health.get("batch_model"):
            parts.append(f"batch_model:   {health.get('batch_model')}")
    else:
        parts.append(f"health:        unavailable @ http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1/info")
    parts.append(f"accessibility: {'granted' if summary['accessibility_granted'] else 'missing'}")
    parts.append(f"microphone:    {'granted' if summary['microphone_granted'] else 'missing'}")
    parts.append(f"logfile:       {LOGFILE if LOGFILE.exists() else f'{LOGFILE} (missing)'}")
    parts.append(f"last_crash:    {crashes.get('Yuwp') or '-'}")
    parts.append(f"server_crash:  {crashes.get('asr-server') or '-'}")
    return "\n".join(parts)


def render_human_paths(rows: list[dict[str, Any]]) -> str:
    lines = ["Yuwp debug paths", ""]
    for row in rows:
        suffix = "✓" if row["exists"] else "✗"
        lines.append(f"{row['label']:<18} {suffix}  {row['path']}")
    return "\n".join(lines)


def render_human_tcc(rows: list[dict[str, Any]]) -> str:
    lines = ["Yuwp TCC", ""]
    if not rows:
        lines.append("No TCC rows for com.yuwp.app")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"{row['service']}: auth_value={row['auth_value']} auth_reason={row['auth_reason']} flags={row['flags']}"
        )
    return "\n".join(lines)


def render_json(data: Any, compact: bool) -> str:
    return json.dumps(data, separators=(",", ":") if compact else None, indent=None if compact else 2, sort_keys=not compact)


def print_data(data: Any, args: argparse.Namespace, human_renderer=None) -> None:
    if args.json or args.compact:
        print(render_json(data, compact=args.compact))
        return
    if human_renderer is None:
        if isinstance(data, list):
            print("\n".join(data))
        else:
            print(data)
        return
    print(human_renderer(data))


def make_parser() -> argparse.ArgumentParser:
    examples = """Examples:
  yuwp-workflow.py
      Catch up on app, server, signing, health, and latest crash in one call.

  yuwp-workflow.py logs 200
      Read the last 200 lines from /tmp/yuwp.log after a failed dictation run.

  yuwp-workflow.py unified 30m 150
      Inspect recent AVFAudio / AppKit / CoreAudio / dyld messages.

  yuwp-workflow.py --json status
      Return structured state for agents.

  yuwp-workflow.py crash
      Locate the newest Yuwp and asr-server crash reports.
"""
    parser = argparse.ArgumentParser(
        prog="yuwp-workflow.py",
        description="Catch-up and debug hub for Yuwp.app + asr-server.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON to stdout")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON to stdout")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Catch-up overview: app, server, signing, health, TCC, latest crash")
    sub.add_parser("paths", help="List important debug paths with existence checks")
    sub.add_parser("tcc", help="Show Accessibility and Microphone TCC rows for com.yuwp.app")
    sub.add_parser("defaults", help="Dump persisted UserDefaults for com.yuwp.app")
    sub.add_parser("health", help="Fetch http://127.0.0.1:9748/v1/info")
    sub.add_parser("crash", help="Show latest Yuwp and asr-server crash report paths")

    logs_parser = sub.add_parser("logs", help="Tail /tmp/yuwp.log")
    logs_parser.add_argument("lines", nargs="?", default="120", help="Number of lines to tail (default: 120)")

    unified_parser = sub.add_parser("unified", help="Tail unified logs for Yuwp + asr-server")
    unified_parser.add_argument("window", nargs="?", default="10m", help="log show window (default: 10m)")
    unified_parser.add_argument("lines", nargs="?", default="120", help="Number of lines to keep (default: 120)")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.json and args.compact:
        die("Choose only one of --json or --compact")

    command = args.command or "status"
    if command == "status":
        print_data(collect_status(), args, render_human_status)
    elif command == "paths":
        print_data({"summary": {"existing": sum(1 for row in path_rows() if row['exists'])}, "paths": path_rows()}, args, lambda data: render_human_paths(data["paths"]))
    elif command == "tcc":
        rows = tcc_rows()
        summary = {
            "accessibility_granted": any(r["service"] == "kTCCServiceAccessibility" and r["auth_value"] == 2 for r in rows),
            "microphone_granted": any(r["service"] == "kTCCServiceMicrophone" and r["auth_value"] == 2 for r in rows),
        }
        print_data({"summary": summary, "rows": rows}, args, lambda data: render_human_tcc(data["rows"]))
    elif command == "defaults":
        data = defaults_info() or {}
        print_data({"summary": {"present": bool(data)}, "defaults": data}, args, lambda d: json.dumps(d["defaults"], indent=2, sort_keys=True) if d["defaults"] else "No defaults for com.yuwp.app")
    elif command == "health":
        data = health_info()
        payload = {"summary": {"ready": isinstance(data, dict) and data.get("status") == "ready"}, "health": data}
        print_data(payload, args, lambda d: json.dumps(d["health"], indent=2, sort_keys=True) if d["health"] else "Health endpoint unavailable")
    elif command == "crash":
        crashes = latest_crash_paths()
        payload = {"summary": {"present": any(crashes.values())}, "crashes": crashes}
        print_data(payload, args, lambda d: "\n".join(["Latest crashes", "", f"Yuwp:      {d['crashes'].get('Yuwp') or '-'}", f"asr-server: {d['crashes'].get('asr-server') or '-'}"]))
    elif command == "logs":
        try:
            lines = int(args.lines)
        except ValueError:
            die(f"Invalid line count: {args.lines}")
        payload = {"summary": {"path": str(LOGFILE), "exists": LOGFILE.exists(), "lines": lines}, "lines": tail_lines(LOGFILE, lines)}
        print_data(payload, args, lambda d: "\n".join(d["lines"]))
    elif command == "unified":
        try:
            lines = int(args.lines)
        except ValueError:
            die(f"Invalid line count: {args.lines}")
        payload = {"summary": {"window": args.window, "lines": lines}, "lines": unified_lines(args.window, lines)}
        print_data(payload, args, lambda d: "\n".join(d["lines"]))
    else:
        die(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
