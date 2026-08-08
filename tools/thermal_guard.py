#!/usr/bin/env python3
"""
tools/thermal_guard.py -- Thermal Protection Guard for HEIST Training.

Checks CPU and GPU temperatures. Exits with 0 if safe, or 1 if temperatures
exceed safety thresholds (CPU > 85°C, GPU > 83°C).
"""

import argparse
import os
import subprocess
import sys
import time

CPU_MAX_TEMP = 85.0
GPU_MAX_TEMP = 83.0


def get_cpu_temp() -> float:
    max_t = 0.0
    thermal_dir = "/sys/class/thermal"
    cpu_keywords = [
        "x86_pkg_temp",
        "coretemp",
        "cpu-thermal",
        "cpu_thermal",
        "k10temp",
        "zenpower",
        "tcpu",
    ]
    found_specific = False

    if os.path.exists(thermal_dir):
        # 1. Try dedicated CPU thermal sensors first
        for zone in os.listdir(thermal_dir):
            if zone.startswith("thermal_zone"):
                type_path = os.path.join(thermal_dir, zone, "type")
                if os.path.exists(type_path):
                    try:
                        with open(type_path) as f:
                            z_type = f.read().strip().lower()
                        if any(kw in z_type for kw in cpu_keywords):
                            temp_path = os.path.join(thermal_dir, zone, "temp")
                            if os.path.exists(temp_path):
                                with open(temp_path) as f:
                                    t = float(f.read().strip()) / 1000.0
                                if 0.0 < t < 115.0:
                                    max_t = max(max_t, t)
                                    found_specific = True
                    except Exception:
                        pass

        # 2. Fallback to generic thermal zones if no dedicated CPU sensor was found
        if not found_specific:
            for zone in os.listdir(thermal_dir):
                if zone.startswith("thermal_zone"):
                    temp_path = os.path.join(thermal_dir, zone, "temp")
                    if os.path.exists(temp_path):
                        try:
                            with open(temp_path) as f:
                                t = float(f.read().strip()) / 1000.0
                            if 0.0 < t < 115.0:
                                max_t = max(max_t, t)
                        except Exception:
                            pass
    return max_t


def get_gpu_temp() -> float:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
        temps = [
            float(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()
        ]
        if temps:
            return max(temps)
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", action="store_true", help="Print temperature log with timestamp"
    )
    parser.add_argument(
        "--max-cpu-temp",
        type=float,
        default=CPU_MAX_TEMP,
        help="Max CPU temp threshold",
    )
    parser.add_argument(
        "--max-gpu-temp",
        type=float,
        default=GPU_MAX_TEMP,
        help="Max GPU temp threshold",
    )
    args = parser.parse_args()

    max_cpu = args.max_cpu_temp
    max_gpu = args.max_gpu_temp

    cpu_temp = get_cpu_temp()
    gpu_temp = get_gpu_temp()
    time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    is_danger = (cpu_temp >= max_cpu) or (gpu_temp >= max_gpu)

    if args.log:
        status_str = "DANGER" if is_danger else "OK"
        print(
            f"[{time_str}] [HW TEMP] CPU: {cpu_temp:.1f}°C (Limit: {max_cpu:.1f}°C) | GPU: {gpu_temp:.1f}°C (Limit: {max_gpu:.1f}°C) | Status: {status_str}"
        )

    if is_danger:
        print(
            f"[{time_str}] [THERMAL KILL SWITCH TRIGGERED] High Temperature Detected! CPU: {cpu_temp:.1f}°C, GPU: {gpu_temp:.1f}°C"
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
