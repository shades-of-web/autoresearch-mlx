"""Run train.py 10 times and save the best checkpoint."""

import os
import re
import shutil
import subprocess
import sys

NUM_RUNS = 10
TRAIN_CMD = ["uv", "run", "train.py"]
LATEST = "latest_checkpoint.npz"
BEST = "best_checkpoint.npz"

results = []
best_bpb = float("inf")
best_run = -1

for run in range(1, NUM_RUNS + 1):
    print(f"\n{'='*60}")
    print(f"  RUN {run}/{NUM_RUNS}")
    print(f"{'='*60}\n", flush=True)

    proc = subprocess.Popen(
        TRAIN_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)

    proc.wait()

    if proc.returncode != 0:
        print(f"\nRun {run} FAILED (exit code {proc.returncode})")
        results.append(("FAIL", None))
        continue

    val_bpb = None
    for line in output_lines:
        m = re.search(r"val_bpb:\s+([\d.]+)", line)
        if m:
            val_bpb = float(m.group(1))
            break

    if val_bpb is None:
        print(f"\nRun {run}: could not parse val_bpb")
        results.append(("NO_BPB", None))
        continue

    results.append(("OK", val_bpb))
    print(f"\nRun {run} val_bpb: {val_bpb:.6f}", end="")

    if val_bpb < best_bpb:
        best_bpb = val_bpb
        best_run = run
        shutil.copy2(LATEST, BEST)
        print(f"  <-- NEW BEST")
    else:
        print(f"  (best: {best_bpb:.6f} from run {best_run})")

    if os.path.exists(LATEST):
        os.remove(LATEST)

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
for i, (status, bpb) in enumerate(results, 1):
    marker = " <-- BEST" if i == best_run else ""
    if bpb is not None:
        print(f"  Run {i:2d}: val_bpb = {bpb:.6f}{marker}")
    else:
        print(f"  Run {i:2d}: {status}")

if best_run > 0:
    print(f"\nBest: run {best_run} with val_bpb = {best_bpb:.6f}")
    print(f"Checkpoint saved to: {BEST}")
else:
    print("\nNo successful runs.")
