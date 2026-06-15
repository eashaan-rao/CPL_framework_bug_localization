"""
Logs GPU memory and utilization every INTERVAL_SECONDS to a timestamped log file.
Run in a separate terminal alongside experiments:
    python Scripts/log_gpu_usage.py
"""

import subprocess
import time
import os
from datetime import datetime

INTERVAL_SECONDS = 5 * 60  # 5 minutes
LOG_DIR = "/home/cs21d002_eashaan/PhD/Objective1/results"
LOG_FILE = os.path.join(LOG_DIR, f"gpu_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

QUERY = "timestamp,name,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory"
HEADER = f"{'timestamp':<22} {'gpu':<12} {'used_MiB':>10} {'free_MiB':>10} {'total_MiB':>10} {'gpu_util%':>10} {'mem_util%':>10}"
SEP = "-" * len(HEADER)


def query_gpu():
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return result.stdout.strip().splitlines()


def format_row(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return line
    ts, name, used, free, total, gpu_util, mem_util = parts
    return f"{ts:<22} {name:<12} {used:>10} {free:>10} {total:>10} {gpu_util:>10} {mem_util:>10}"


def log(message, also_print=True):
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")
    if also_print:
        print(message)


log(f"GPU usage log started — interval: {INTERVAL_SECONDS // 60} min")
log(f"Log file: {LOG_FILE}")
log(SEP)
log(HEADER)
log(SEP)

while True:
    rows = query_gpu()
    for row in rows:
        log(format_row(row))
    time.sleep(INTERVAL_SECONDS)
