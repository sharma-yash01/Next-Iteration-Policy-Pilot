"""
Entrypoint: load HumanEval+, run repair loops, compute pilot metrics, save summary.
"""

from __future__ import annotations

import json
import os
import time

from tqdm import tqdm

from analyze import compute_ece, compute_feature_auc, compute_waste_rate, print_summary
from config import DATA_DIR, MODEL, N_PROBLEMS, RESULTS_DIR
from data_lcb import get_problems
from repair import run_repair_loop


def main():
    """Run pilot: repair loops on first N_PROBLEMS (LiveCodeBench), then compute and print metrics."""
    # Archive trajectory data only when explicitly requested (preserves crash-resume).
    if os.environ.get("PILOT_FORCE_CLEAN") == "1" and os.path.isdir(DATA_DIR):
        jsonl_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")]
        if jsonl_files:
            backup = f"data/trajectories_poisoned_{int(time.time())}"
            os.rename(DATA_DIR, backup)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n = int(os.environ.get("PILOT_N_PROBLEMS", N_PROBLEMS))
    problems = get_problems()[:n]
    trajectories = [
        run_repair_loop(p["task_id"], p["prompt"], MODEL, problem_dict=p)
        for p in tqdm(problems, desc="Repair loops")
    ]

    waste = compute_waste_rate(trajectories)
    ece = compute_ece(trajectories)
    auc_mean, auc_std = compute_feature_auc(trajectories)

    print_summary(waste, ece, auc_mean, auc_std)
    with open(f"{RESULTS_DIR}/pilot_summary.json", "w") as f:
        json.dump(
            {
                "waste_rate": waste,
                "ece": ece,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
