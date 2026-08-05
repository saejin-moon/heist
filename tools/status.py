#!/usr/bin/env python3
"""
tools/status.py -- HEIST Terminal Dashboard (using Rich).

Provides a fullscreen, live-updating dashboard for training campaigns,
log files, model concurrency, and checkpoint status.

Usage:
    uv run python tools/status.py
    uv run python tools/status.py --watch      # Poll status continuously
    uv run python tools/status.py --interval 3  # Set poll interval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "log"
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"

MODEL_NAMES = [
    "ippo",
    "mappo",
    "mappo_car",
    "mappo_cir",
    "comm",
    "comm_cir",
    "comm_cir_car",
    "qmix",
    "coma",
    "coma_cir",
]


def get_running_pids() -> set[int]:
    """Find running python PIDs related to HEIST training/eval."""
    pids = set()
    try:
        import subprocess

        out = subprocess.check_output(["ps", "aux"], text=True)
        for line in out.splitlines():
            if (
                "src/train_" in line
                or "src/eval_stage.py" in line
                or "train.zsh" in line
            ):
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    pids.add(int(parts[1]))
    except Exception:
        pass
    return pids


def _get_next_run_id(results_dir: Path, prefix: str = "run") -> str:
    if not results_dir.is_dir():
        return f"{prefix}001"[:6]
    existing = [
        d.name
        for d in results_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(prefix)
        and d.name[len(prefix) :].isdigit()
    ]
    nums = [int(name[len(prefix) :]) for name in existing if len(name) <= 6]
    next_num = max(nums) + 1 if nums else 1
    return f"{prefix}{next_num:03d}"[:6]


def get_active_campaign_info() -> dict:
    """Detect current campaign configuration (Run ID, active stages, side-tasks mode)."""
    launch_file = LOG_DIR / "launch.out"
    info = {
        "run_id": "N/A",
        "active_stages": set(),
        "side_tasks": False,
        "fast_mode": False,
        "active_run_start": 0.0,
    }

    launch_mtime = launch_file.stat().st_mtime if launch_file.is_file() else 0
    launch_run_id = None
    launch_stages = set()

    if launch_file.is_file():
        try:
            content = launch_file.read_text(errors="replace")
            if "[SIDE-TASKS ENABLED]" in content:
                info["side_tasks"] = True
            if "[FAST MODE]" in content:
                info["fast_mode"] = True

            runs = re.findall(r"Campaign Evaluation Run ID:\s*(\w+)", content)
            if runs:
                launch_run_id = runs[-1]

            m_stages = re.findall(r"Starting training for stage (\d+)", content)
            if m_stages:
                launch_stages = {int(s) for s in m_stages}
        except Exception:
            pass

    latest_log_mtime = 0
    latest_log_file = None
    recent_log_mtimes = []

    if LOG_DIR.is_dir():
        for p in LOG_DIR.glob("*.log"):
            mtime = p.stat().st_mtime
            if mtime > latest_log_mtime:
                latest_log_mtime = mtime
                latest_log_file = p

    if LOG_DIR.is_dir() and latest_log_mtime > 0:
        for p in LOG_DIR.glob("*.log"):
            mtime = p.stat().st_mtime
            if (latest_log_mtime - mtime) < 600:
                recent_log_mtimes.append(mtime)

    pids = get_running_pids()
    active_training = len(pids) > 0 or (
        latest_log_mtime > 0 and (time.time() - latest_log_mtime) < 120
    )

    latest_results_mtime = 0
    latest_results_run_id = None
    if RESULTS_DIR.is_dir():
        all_runs = sorted(
            [
                d
                for d in RESULTS_DIR.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ],
            key=lambda x: (x.stat().st_mtime, x.name),
        )
        if all_runs:
            latest_results_dir = all_runs[-1]
            latest_results_mtime = latest_results_dir.stat().st_mtime
            latest_results_run_id = latest_results_dir.name

    # Determine whether launch.out is fresh or stale
    is_launch_fresh = (
        launch_run_id is not None
        and launch_mtime >= latest_results_mtime - 5
        and (
            not active_training
            or abs(launch_mtime - latest_log_mtime) <= 300
            or launch_mtime >= latest_log_mtime
        )
    )

    if is_launch_fresh:
        info["run_id"] = launch_run_id
        info["active_stages"] = launch_stages or {0}
        info["active_run_start"] = launch_mtime
    else:
        # If launch.out is stale, determine side_tasks and active_stages from recent logs
        if latest_log_file and active_training:
            info["side_tasks"] = "_st_" in latest_log_file.name

        if LOG_DIR.is_dir() and latest_log_mtime > 0 and (active_training or latest_log_mtime > latest_results_mtime):
            for p in LOG_DIR.glob("*.log"):
                mtime = p.stat().st_mtime
                if (latest_log_mtime - mtime) < 300:
                    m_stage = re.search(r"_s(\d+)\.log$", p.name)
                    if m_stage:
                        info["active_stages"].add(int(m_stage.group(1)))

        if active_training and latest_log_mtime > latest_results_mtime:
            prefix = "st" if info["side_tasks"] else "run"
            info["run_id"] = _get_next_run_id(RESULTS_DIR, prefix=prefix)
            info["active_run_start"] = min(recent_log_mtimes) if recent_log_mtimes else latest_log_mtime
        else:
            if latest_results_run_id and latest_results_run_id.startswith("run"):
                info["run_id"] = latest_results_run_id
            elif latest_results_run_id:
                prefix = "st" if info["side_tasks"] else "run"
                info["run_id"] = _get_next_run_id(RESULTS_DIR, prefix=prefix)
            else:
                info["run_id"] = launch_run_id or "N/A"
            info["active_run_start"] = latest_results_mtime or latest_log_mtime

    if not info["active_stages"]:
        info["active_stages"] = {0}

    return info


def check_models_status(
    active_stages: set[int],
    running_pids: set[int],
    active_run_start: float = 0.0,
) -> list[dict]:
    """Gather status across all 10 models for the active stage, ignoring stale checkpoints."""
    results = []
    active_stage = max(active_stages) if active_stages else 0

    for name in MODEL_NAMES:
        possible_log_names = [
            f"{name}_st_s{active_stage}.log",
            f"{name}_s{active_stage}.log",
        ]
        log_path = None
        log_mtime = 0.0
        for p_name in possible_log_names:
            candidate = LOG_DIR / p_name
            if candidate.is_file():
                log_path = candidate
                log_mtime = candidate.stat().st_mtime
                break

        is_log_fresh = (
            log_path is not None
            and (
                active_run_start == 0.0
                or log_mtime >= active_run_start - 60
                or (time.time() - log_mtime) < 300
            )
        )

        possible_ckpt_names = [
            f"{name}_st_s{active_stage}",
            f"{name}_s{active_stage}",
        ]
        ckpt_complete = False
        ckpt_exists = False
        ckpt_mtime = 0.0
        completed_steps = "-"
        for c_name in possible_ckpt_names:
            ckpt_dir = CKPT_DIR / c_name
            marker = ckpt_dir / "complete.json"
            if ckpt_dir.is_dir():
                ckpt_exists = True
                ckpt_mtime = ckpt_dir.stat().st_mtime
            if marker.is_file():
                ckpt_exists = True
                marker_mtime = marker.stat().st_mtime
                if marker_mtime > ckpt_mtime:
                    ckpt_mtime = marker_mtime
                is_marker_fresh = (
                    active_run_start == 0.0
                    or marker_mtime >= active_run_start - 60
                    or (is_log_fresh and abs(marker_mtime - log_mtime) < 300)
                )
                if is_marker_fresh:
                    ckpt_complete = True
                    try:
                        data = json.loads(marker.read_text())
                        completed_steps = str(data.get("completed_steps", "-"))
                    except Exception:
                        pass
                    break

        step_str = "-"
        sps_str = "-"
        reward_str = "-"
        runtime_str = "-"
        status = "QUEUED"

        if is_log_fresh and log_path:
            mtime_diff = time.time() - log_mtime
            is_recent = mtime_diff < 30

            lines = log_path.read_text(errors="replace").splitlines()
            if lines:
                for line_str in reversed(lines):
                    m_prog = re.search(
                        r"step=(\d+)\s+sps=(\d+)\s+mean_reward=([-\d.]+)", line_str
                    )
                    if m_prog:
                        step_str = m_prog.group(1)
                        sps_str = m_prog.group(2)
                        reward_str = f"{float(m_prog.group(3)):.3f}"
                        break
                    m_done = re.search(
                        r"training done in ([\d.]+s|[\d.]+ min)", line_str
                    )
                    if m_done:
                        runtime_str = m_done.group(1)
                        break

            has_finished = lines and any(
                "training done" in line_str for line_str in lines[-5:]
            )
            if is_recent and not has_finished:
                status = "RUNNING"
            elif ckpt_complete or has_finished:
                status = "COMPLETE"
            else:
                status = "PAUSED"

        if ckpt_complete and status == "COMPLETE":
            checkpoint_cell = "[bold green]SAVED[/bold green]"
        elif ckpt_exists:
            is_ckpt_fresh = (
                active_run_start == 0.0
                or ckpt_mtime >= active_run_start - 60
                or (is_log_fresh and abs(ckpt_mtime - log_mtime) < 300)
            )
            if is_ckpt_fresh and status == "RUNNING":
                checkpoint_cell = "[bold yellow]SAVING[/bold yellow]"
            elif is_ckpt_fresh:
                checkpoint_cell = "[green]EXISTS[/green]"
            else:
                checkpoint_cell = "[dim yellow]STALE (OLD)[/dim yellow]"
        else:
            checkpoint_cell = "[dim]PENDING[/dim]"

        results.append(
            {
                "model": name,
                "stage": active_stage,
                "status": status,
                "steps": step_str
                if status == "RUNNING" or completed_steps == "N/A"
                else completed_steps,
                "sps": sps_str if status == "RUNNING" else "-",
                "mean_reward": reward_str,
                "runtime": runtime_str,
                "checkpoint": checkpoint_cell,
            }
        )

    return results


def make_dashboard_panel() -> Panel:
    info = get_active_campaign_info()
    pids = get_running_pids()
    models = check_models_status(
        info["active_stages"],
        pids,
        active_run_start=info.get("active_run_start", 0.0),
    )

    active_stage = max(info["active_stages"]) if info["active_stages"] else 0
    st_text = (
        "[bold green]ENABLED[/bold green]"
        if info["side_tasks"]
        else "[dim]DISABLED[/dim]"
    )
    fast_text = "[bold yellow]ON[/bold yellow]" if info["fast_mode"] else "OFF"
    time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    title_text = (
        f"[bold cyan]HEIST MARL DASHBOARD[/bold cyan]  |  "
        f"Run ID: [bold gold1]{info['run_id']}[/bold gold1]  |  "
        f"Active Stage: [bold magenta]{active_stage}[/bold magenta]  |  "
        f"Side-Tasks: {st_text}  |  "
        f"Fast Mode: {fast_text}"
    )

    # 1. Models Matrix Table
    table = Table(
        expand=True,
        box=None,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Algorithm Model", style="bold cyan", no_wrap=True)
    table.add_column("Stage", justify="center", no_wrap=True)
    table.add_column("Execution Status", justify="center", no_wrap=True)
    table.add_column("Timesteps", justify="right", no_wrap=True)
    table.add_column("SPS", justify="right", no_wrap=True)
    table.add_column("Mean Reward", justify="right", no_wrap=True)
    table.add_column("Duration", justify="center", no_wrap=True)
    table.add_column("Checkpoint", justify="center", no_wrap=True)

    for m in models:
        st = m["status"]
        if st == "RUNNING":
            status_cell = "[bold black on green] RUNNING [/bold black on green]"
        elif st == "COMPLETE":
            status_cell = "[bold white on blue] COMPLETE [/bold white on blue]"
        elif st == "QUEUED":
            status_cell = "[dim white] QUEUED [/dim white]"
        else:
            status_cell = f"[bold black on yellow] {st} [/bold black on yellow]"

        table.add_row(
            m["model"],
            str(m["stage"]),
            status_cell,
            m["steps"],
            m["sps"],
            m["mean_reward"],
            m["runtime"],
            m["checkpoint"],
        )

    # 2. Live Log Feed
    log_feed = Text()
    if LOG_DIR.is_dir():
        log_files = sorted(
            [
                p
                for p in LOG_DIR.glob("*.log")
                if (time.time() - p.stat().st_mtime) < 60
            ],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if log_files:
            latest = log_files[0]
            log_feed.append(f"Log Stream: {latest.name}\n", style="bold gold1")
            lines = latest.read_text(errors="replace").splitlines()[-3:]
            for line_str in lines:
                log_feed.append(f"   > {line_str}\n", style="dim white")
        else:
            log_feed.append(
                "No active log file modifications in the last 60 seconds.",
                style="dim italic",
            )

    grid = Table.grid(expand=True)
    grid.add_row(
        Panel(
            table,
            title="[bold white]Model Execution Matrix[/bold white]",
            border_style="bright_blue",
        )
    )
    grid.add_row(
        Panel(
            log_feed,
            title="[bold white]Diagnostic Log Stream[/bold white]",
            border_style="dim blue",
            height=6,
        )
    )

    return Panel(
        grid,
        title=title_text,
        subtitle=f"[dim]Updated: {time_str}  |  Press Ctrl+C to exit[/dim]",
        border_style="cyan",
    )


def main():
    ap = argparse.ArgumentParser(description="HEIST Fullscreen Terminal Dashboard")
    ap.add_argument("--watch", action="store_true", help="continuously poll status")
    ap.add_argument("--interval", type=int, default=3, help="watch interval in seconds")
    args = ap.parse_args()

    console = Console()

    if args.watch:
        try:
            with Live(
                make_dashboard_panel(),
                console=console,
                screen=True,
                refresh_per_second=1,
            ) as live:
                while True:
                    time.sleep(args.interval)
                    live.update(make_dashboard_panel())
        except KeyboardInterrupt:
            return 0
    else:
        console.print(make_dashboard_panel())
        return 0


if __name__ == "__main__":
    sys.exit(main())
