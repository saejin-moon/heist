#!/usr/bin/env python3
"""
tools/status.py -- Check active training runs, log files, and checkpoints.

Usage:
    uv run python tools/status.py
    uv run python tools/status.py --logdir log
    uv run python tools/status.py --watch  # poll status every 5 seconds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_checkpoints() -> list[dict]:
    ckpt_dir = REPO_ROOT / "checkpoints"
    if not ckpt_dir.is_dir():
        return []

    runs = []
    for d in sorted(ckpt_dir.iterdir()):
        if not d.is_dir():
            continue
        marker = d / "complete.json"
        is_complete = marker.is_file()
        info = {"run_name": d.name, "complete": is_complete}
        if is_complete:
            try:
                data = json.loads(marker.read_text())
                info.update(data)
            except Exception:
                pass
        runs.append(info)
    return runs


def check_logs() -> list[dict]:
    log_dir = REPO_ROOT / "log"
    if not log_dir.is_dir():
        return []

    logs = []
    for p in sorted(log_dir.glob("*.log")):
        size = p.stat().st_size
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
        lines = p.read_text(errors="replace").splitlines()
        tail = lines[-5:] if lines else []
        logs.append(
            {
                "file": p.name,
                "size_kb": round(size / 1024, 1),
                "mtime": mtime,
                "lines_count": len(lines),
                "tail": tail,
            }
        )
    return logs


def print_status():
    print("=" * 72)
    print(f"HEIST Training Status -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    ckpts = check_checkpoints()
    print("\n[1] Checkpoints")
    if not ckpts:
        print("    No checkpoints found.")
    else:
        print(f"    {'Run Name':<35} {'Status':<12} {'Completed Steps':<18}")
        print("    " + "-" * 65)
        for c in ckpts:
            status = "COMPLETE ✅" if c["complete"] else "IN PROGRESS 🔄"
            steps = str(c.get("completed_steps", "N/A"))
            print(f"    {c['run_name']:<35} {status:<12} {steps:<18}")

    logs = check_logs()
    print("\n[2] Log Files")
    if not logs:
        print("    No log files found in log/")
    else:
        for log_file in logs:
            print(
                f"\n  📄 {log_file['file']} ({log_file['size_kb']} KB, last modified: {log_file['mtime']})"
            )
            print("     Tail output:")
            for line in log_file["tail"]:
                print(f"       | {line}")

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(description="HEIST training log and status checker")
    ap.add_argument("--watch", action="store_true", help="continuously poll status")
    ap.add_argument("--interval", type=int, default=5, help="watch interval in seconds")
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                print_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0
    else:
        print_status()
        return 0


if __name__ == "__main__":
    sys.exit(main())
