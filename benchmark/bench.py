"""GPU throughput benchmark for Isaac Lab Go2 flat-ground training.

Runs a short RSL-RL training job at several robot counts, parses the per-iteration
"Steps per second" line, and samples nvidia-smi for peak VRAM. Works on Windows and Linux/WSL.

    python bench.py --isaaclab <IsaacLab checkout> --label windows
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

TASK = "Isaac-Velocity-Flat-UnitreeGo2"
STEPS_PER_ENV = 24
WARMUP_ITERS = 5  # iterations dropped before taking the steady-state median

RE_SPS = re.compile(r"Steps per second:\s*([\d.]+)")
RE_COLLECT = re.compile(r"Collection time:\s*([\d.]+)s")
RE_LEARN = re.compile(r"Learning time:\s*([\d.]+)s")
OOM_MARKERS = ("out of memory", "CUDA_ERROR_OUT_OF_MEMORY", "cudaErrorMemoryAllocation")


def query_vram_mib() -> int | None:
    """Total used VRAM on GPU 0 in MiB (WDDM/WSL do not report per-process usage)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class VramSampler(threading.Thread):
    """Background thread that records peak GPU memory use while a run is going."""

    def __init__(self, interval: float = 0.5):
        """Prepare a sampler that polls every ``interval`` seconds."""
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._halt = threading.Event()

    def run(self):
        """Poll nvidia-smi and keep the highest value until stopped."""
        while not self._halt.is_set():
            used = query_vram_mib()
            if used is not None:
                self.peak = max(self.peak, used)
            self._halt.wait(self.interval)

    def stop(self):
        """Stop polling and wait for the thread to finish."""
        self._halt.set()
        self.join()


def run_one(isaaclab: Path, uv: str, num_envs: int, iters: int, log_path: Path) -> dict:
    """Train at one robot count, write its log to ``log_path``, return throughput and VRAM figures."""
    cmd = [
        uv, "run", "isaaclab", "train", "--rl_library", "rsl_rl", "--task", TASK,
        "--num_envs", str(num_envs), "--max_iterations", str(iters),
        "--run_name", f"bench_{num_envs}", "physics=newton_mjwarp", "--viz", "none",
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    baseline = query_vram_mib() or 0
    sampler = VramSampler()
    sampler.start()
    sps, collect, learn = [], [], []
    oom = False
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=isaaclab, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            log.write(line)
            if m := RE_SPS.search(line):
                sps.append(float(m.group(1)))
            elif m := RE_COLLECT.search(line):
                collect.append(float(m.group(1)))
            elif m := RE_LEARN.search(line):
                learn.append(float(m.group(1)))
            if any(k in line for k in OOM_MARKERS):
                oom = True
        code = proc.wait()
    wall = time.time() - t0
    sampler.stop()

    steady = sps[WARMUP_ITERS:] or sps
    return {
        "num_envs": num_envs,
        "exit_code": code,
        "oom": oom,
        "iterations_logged": len(sps),
        "steps_per_s_median": round(statistics.median(steady)) if steady else None,
        "steps_per_s_max": round(max(steady)) if steady else None,
        "collection_s_median": round(statistics.median(collect[WARMUP_ITERS:] or collect), 3) if collect else None,
        "learning_s_median": round(statistics.median(learn[WARMUP_ITERS:] or learn), 3) if learn else None,
        "vram_baseline_mib": baseline,
        "vram_peak_mib": sampler.peak,
        "vram_used_by_run_mib": sampler.peak - baseline,
        "wall_s": round(wall, 1),
        "log": str(log_path),
    }


def main():
    """Run the benchmark at each requested robot count and save the results as JSON."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isaaclab", type=Path, required=True, help="Isaac Lab checkout")
    p.add_argument("--label", default=platform.system().lower())
    p.add_argument("--envs", default="256,1024,4096,8192")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()

    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv"))
    args.out.mkdir(parents=True, exist_ok=True)
    total_vram = None
    try:
        total_vram = int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True).stdout.strip())
    except (OSError, ValueError):
        pass

    results = []
    for n in [int(x) for x in args.envs.split(",")]:
        print(f"[{args.label}] {n} envs x {args.iters} iterations ...", flush=True)
        r = run_one(args.isaaclab, uv, n, args.iters, args.out / f"{args.label}_{n}.log")
        r["vram_total_mib"] = total_vram
        results.append(r)
        print(f"[{args.label}] {json.dumps(r)}", flush=True)
        (args.out / f"{args.label}.json").write_text(json.dumps(results, indent=2))
        if r["exit_code"] != 0 or r["oom"]:
            print(f"[{args.label}] stopping: run failed (exit {r['exit_code']}, oom={r['oom']})", flush=True)
            break

    print(f"\n{args.label}: env  steps/s(median)  peak VRAM MiB  wall s")
    for r in results:
        print(f"  {r['num_envs']:>6}  {r['steps_per_s_median']!s:>14}  {r['vram_peak_mib']:>13}  {r['wall_s']:>6}")


if __name__ == "__main__":
    main()
