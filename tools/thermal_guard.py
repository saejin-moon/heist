#!/usr/bin/env python3
"""
tools/thermal_guard.py -- Thermal Protection Guard for HEIST Training.

Checks CPU and GPU temperatures. Exits with 0 if safe, or 1 if temperatures
exceed safety thresholds (CPU > 85°C, GPU > 83°C).
"""

import sys
import os
import time
import argparse
import subprocess

CPU_MAX_TEMP = 85.0
GPU_MAX_TEMP = 83.0

def get_cpu_temp() -> float:
    max_t = 0.0
    thermal_dir = "/sys/class/thermal"
    if os.path.exists(thermal_dir):
        for zone in os.listdir(thermal_dir):
            if zone.startswith("thermal_zone"):
                temp_path = os.path.join(thermal_dir, zone, "temp")
                if os.path.exists(temp_path):
                    try:
                        t = float(open(temp_path).read().strip()) / 1000.0
                        if t < 150.0 and t > max_t:
                            max_t = t
                    except Exception:
                        pass
    return max_t

def get_gpu_temp() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        )
        temps = [float(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
        if temps:
            return max(temps)
    except Exception:
        pass
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="Print temperature log with timestamp")
    args = parser.parse_args()

    cpu_temp = get_cpu_temp()
    gpu_temp = get_gpu_temp()
    time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    is_danger = (cpu_temp >= CPU_MAX_TEMP) or (gpu_temp >= GPU_MAX_TEMP)

    if args.log:
        status_str = "DANGER" if is_danger else "OK"
        print(f"[{time_str}] [HW TEMP] CPU: {cpu_temp:.1f}°C (Limit: {CPU_MAX_TEMP}°C) | GPU: {gpu_temp:.1f}°C (Limit: {GPU_MAX_TEMP}°C) | Status: {status_str}")

    if is_danger:
        print(f"[{time_str}] [THERMAL KILL SWITCH TRIGGERED] High Temperature Detected! CPU: {cpu_temp:.1f}°C, GPU: {gpu_temp:.1f}°C")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
