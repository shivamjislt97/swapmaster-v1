"""
Direct GPU stats via nvidia-smi — completely independent of bot.py.
"""

import subprocess


def get_gpu_stats() -> dict:
    result = {
        "gpu_name":    "Unknown",
        "gpu_util":    0,
        "vram_used":   0.0,
        "vram_total":  0.0,
        "vram_free":   0.0,
        "temperature": 0,
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 6:
            result["gpu_name"]    = parts[0]
            result["gpu_util"]    = int(parts[1])
            result["vram_used"]   = round(int(parts[2]) / 1024, 1)
            result["vram_total"]  = round(int(parts[3]) / 1024, 1)
            result["vram_free"]   = round(int(parts[4]) / 1024, 1)
            result["temperature"] = int(parts[5])
    except Exception:
        pass
    return result
