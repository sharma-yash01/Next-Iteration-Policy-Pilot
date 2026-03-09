#!/usr/bin/env python3
"""
Analyze pilot trajectory data and print a detailed diagnostic report.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

import numpy as np

PASS_THRESHOLD = 0.8
IMPROVEMENT_THRESHOLD = 0.05
MAX_ITERATIONS = 5
DATA_PATTERN = "pilot-data/trajectories/*.jsonl"
PILOT_SUMMARY_PATH = "pilot-data/results/pilot_summary.json"


def load_trajectories(path_pattern: str) -> list[list[dict[str, Any]]]:
    """Load all trajectory files into memory."""
    files = sorted(glob.glob(path_pattern), key=_file_sort_key)
    trajectories: list[list[dict[str, Any]]] = []
    for path in files:
        steps: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                steps.append(json.loads(text))
        if steps:
            steps.sort(key=lambda s: int(s.get("iteration", 0)))
            trajectories.append(steps)
    return trajectories


def _file_sort_key(path: str) -> tuple[int, str]:
    """Sort numeric IDs numerically and mixed IDs lexicographically."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return (0, f"{int(stem):09d}") if stem.isdigit() else (1, stem)


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _fmt_pct(num: float) -> str:
    return f"{100.0 * num:.2f}%"


def _safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def _state(pass_rate: float) -> str:
    if pass_rate >= PASS_THRESHOLD:
        return "solved"
    if pass_rate <= 0.0:
        return "zero"
    return "partial"


def _update_transition_counts(
    prev: float,
    nxt: float,
    transitions: Counter[str],
) -> str:
    """Update state transition counters and return delta class."""
    delta = nxt - prev
    if delta > 0:
        delta_class = "improved"
    elif delta < 0:
        delta_class = "regressed"
    else:
        delta_class = "flat"
    prev_state = _state(prev)
    next_state = _state(nxt)
    if prev_state == "solved" and next_state != "solved":
        transitions["solved->regressed"] += 1
    transitions[f"{prev_state}->{next_state}"] += 1
    return delta_class


def _first_solved_iteration(traj: list[dict[str, Any]]) -> int | None:
    for step in traj:
        if float(step.get("pass_rate", 0.0)) >= PASS_THRESHOLD:
            return int(step.get("iteration", 0))
    return None


def compute_waste_for_trajectory(traj: list[dict[str, Any]]) -> float:
    """Match pilot-impl/analyze.py waste definition exactly."""
    first = _first_solved_iteration(traj)
    oracle = first if first is not None else MAX_ITERATIONS - 1
    return ((MAX_ITERATIONS - 1) - oracle) / (MAX_ITERATIONS - 1)


def compute_ece(trajectories: list[list[dict[str, Any]]], n_bins: int = 15) -> float:
    """Match pilot-impl/analyze.py ECE definition exactly."""
    confs: list[float] = []
    outcomes: list[int] = []
    for traj in trajectories:
        for step in traj:
            confs.append(float(step.get("self_verification_score", 0.5)))
            outcomes.append(int(float(step.get("pass_rate", 0.0)) >= PASS_THRESHOLD))
    if not confs:
        return 0.0
    conf_arr = np.array(confs)
    out_arr = np.array(outcomes)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (conf_arr >= lo) & (conf_arr < hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr <= 1.0)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(conf_arr)) * abs(conf_arr[mask].mean() - out_arr[mask].mean())
    return float(ece)


def classify_outcome(traj: list[dict[str, Any]]) -> str:
    first_solved = _first_solved_iteration(traj)
    if first_solved == 0:
        return "Solved-at-0"
    if first_solved is not None:
        return "Solved-later"
    max_pass = max(float(s.get("pass_rate", 0.0)) for s in traj)
    if max_pass > 0.0:
        return "Partial"
    return "Never-passed"


def print_dataset_overview(trajectories: list[list[dict[str, Any]]]) -> None:
    _section("Section 1: Dataset Overview")
    n_problems = len(trajectories)
    step_counts = [len(t) for t in trajectories]
    total_steps = sum(step_counts)
    timestamps = [float(s.get("timestamp", 0.0)) for t in trajectories for s in t]
    ts_min = min(timestamps) if timestamps else 0.0
    ts_max = max(timestamps) if timestamps else 0.0
    duration_sec = max(ts_max - ts_min, 0.0)
    print(f"Problems: {n_problems}")
    print(f"Total steps: {total_steps}")
    print(f"Steps/problem: min={min(step_counts)}, max={max(step_counts)}, mean={mean(step_counts):.2f}")
    print(f"Timestamps: earliest={ts_min:.3f}, latest={ts_max:.3f}, span_sec={duration_sec:.3f}")
    print(f"Span hours: {duration_sec / 3600.0:.2f}")


def print_outcome_classification(
    trajectories: list[list[dict[str, Any]]],
) -> tuple[dict[str, list[list[dict[str, Any]]]], list[dict[str, Any]]]:
    _section("Section 2: Problem Outcome Classification")
    buckets: dict[str, list[list[dict[str, Any]]]] = {
        "Solved-at-0": [],
        "Solved-later": [],
        "Partial": [],
        "Never-passed": [],
    }
    details: list[dict[str, Any]] = []
    for traj in trajectories:
        bucket = classify_outcome(traj)
        buckets[bucket].append(traj)
        details.append(
            {
                "problem_id": str(traj[0].get("problem_id")),
                "bucket": bucket,
                "first_solved": _first_solved_iteration(traj),
                "max_pass_rate": max(float(s.get("pass_rate", 0.0)) for s in traj),
                "final_pass_rate": float(traj[-1].get("pass_rate", 0.0)),
            }
        )
    total = len(trajectories)
    for bucket in ("Solved-at-0", "Solved-later", "Partial", "Never-passed"):
        count = len(buckets[bucket])
        ratio = count / total if total else 0.0
        print(f"{bucket:>12}: {count:3d} ({_fmt_pct(ratio)})")
    return buckets, details


def print_waste_decomposition(
    trajectories: list[list[dict[str, Any]]],
    buckets: dict[str, list[list[dict[str, Any]]]],
) -> None:
    _section("Section 3: Waste Rate Decomposition")
    all_wastes = [compute_waste_for_trajectory(t) for t in trajectories]
    overall = mean(all_wastes) if all_wastes else 0.0
    print(f"Overall waste rate: {overall:.6f} ({_fmt_pct(overall)})")
    for bucket_name in ("Solved-at-0", "Solved-later", "Partial", "Never-passed"):
        bucket_trajs = buckets[bucket_name]
        bucket_wastes = [compute_waste_for_trajectory(t) for t in bucket_trajs]
        avg = mean(bucket_wastes) if bucket_wastes else 0.0
        print(f"{bucket_name:>12} waste: {avg:.6f} ({_fmt_pct(avg)}) over {len(bucket_trajs)} problems")
    print()
    print("Solved-at-0 problems (all post-0 iterations are waste by definition):")
    solved0 = sorted(
        (
            str(traj[0].get("problem_id")),
            compute_waste_for_trajectory(traj),
        )
        for traj in buckets["Solved-at-0"]
    )
    if not solved0:
        print("- none")
    else:
        for pid, waste in solved0:
            wasted_iters = int(round(waste * (MAX_ITERATIONS - 1)))
            print(f"- {pid}: wasted_iters={wasted_iters}, waste_rate={waste:.3f}")
    print()
    print("Note: Unsolved trajectories are assigned oracle=iteration 4, so they contribute 0 waste.")


def print_pass_rate_trajectory_analysis(trajectories: list[list[dict[str, Any]]]) -> None:
    _section("Section 4: Pass Rate Trajectory Analysis")
    by_iter: dict[int, list[float]] = defaultdict(list)
    delta_counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    for traj in trajectories:
        for step in traj:
            by_iter[int(step.get("iteration", 0))].append(float(step.get("pass_rate", 0.0)))
        for i in range(len(traj) - 1):
            prev = float(traj[i].get("pass_rate", 0.0))
            nxt = float(traj[i + 1].get("pass_rate", 0.0))
            delta_class = _update_transition_counts(prev, nxt, transitions)
            delta_counts[delta_class] += 1
    for it in range(MAX_ITERATIONS):
        values = by_iter[it]
        print(
            f"Iteration {it}: mean_pass_rate={mean(values):.4f}, median={_safe_median(values):.4f}, "
            f"min={min(values):.4f}, max={max(values):.4f}"
        )
    print()
    total_pairs = sum(delta_counts.values())
    print(
        "Consecutive transitions: "
        f"total={total_pairs}, improved={delta_counts['improved']}, "
        f"regressed={delta_counts['regressed']}, flat={delta_counts['flat']}"
    )
    print()
    print("State transition counts:")
    for key in (
        "zero->zero",
        "zero->partial",
        "zero->solved",
        "partial->partial",
        "partial->solved",
        "partial->zero",
        "solved->solved",
        "solved->partial",
        "solved->zero",
        "solved->regressed",
    ):
        if key in transitions:
            print(f"- {key}: {transitions[key]}")


def print_self_verification_audit(trajectories: list[list[dict[str, Any]]]) -> None:
    _section("Section 5: Self-Verification Score Audit")
    scores = [float(s.get("self_verification_score", 0.5)) for t in trajectories for s in t]
    c = Counter(scores)
    print(f"Total score entries: {len(scores)}")
    print(f"Unique score values: {len(c)}")
    for score, count in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
        print(f"- score={score:.6f}, count={count}")
    ece = compute_ece(trajectories)
    print(f"ECE (recomputed): {ece:.6f}")
    mostly_half = (c.get(0.5, 0) / len(scores)) if scores else 0.0
    if mostly_half > 0.9:
        print("Interpretation: verifier confidence is dominated by 0.5 fallback-like values.")
    else:
        print("Interpretation: verifier has non-trivial score variation.")


def print_label_distribution_for_auc(trajectories: list[list[dict[str, Any]]]) -> None:
    _section("Section 6: Label Distribution For AUC")
    labels: list[int] = []
    for traj in trajectories:
        for i in range(len(traj) - 1):
            prev = float(traj[i].get("pass_rate", 0.0))
            nxt = float(traj[i + 1].get("pass_rate", 0.0))
            labels.append(int((nxt - prev) >= IMPROVEMENT_THRESHOLD))
    total = len(labels)
    pos = sum(labels)
    neg = total - pos
    pos_rate = pos / total if total else 0.0
    print(f"Label rule: (pass_rate[i+1] - pass_rate[i]) >= {IMPROVEMENT_THRESHOLD:.2f}")
    print(f"Total labels: {total}")
    print(f"Positive: {pos}")
    print(f"Negative: {neg}")
    print(f"Positive rate: {_fmt_pct(pos_rate)}")
    if pos < 20:
        print("AUC reliability warning: <20 positives, cross-validation is unstable/directional only.")
    if pos == 0 or neg == 0:
        print("AUC undefined in strict ROC sense: only one class present.")


def print_error_breakdown(trajectories: list[list[dict[str, Any]]]) -> None:
    _section("Section 7: Error Type Breakdown")
    all_errors: Counter[str] = Counter()
    per_iter: dict[int, Counter[str]] = defaultdict(Counter)
    for traj in trajectories:
        for step in traj:
            it = int(step.get("iteration", 0))
            errs = step.get("error_types", [])
            for err in errs:
                name = str(err)
                all_errors[name] += 1
                per_iter[it][name] += 1
    if not all_errors:
        print("No errors logged.")
        return
    print("Aggregate error counts:")
    total = sum(all_errors.values())
    for err, count in all_errors.most_common():
        print(f"- {err}: {count} ({_fmt_pct(count / total)})")
    print()
    print("Per-iteration error counts:")
    for it in range(MAX_ITERATIONS):
        row = per_iter[it]
        if not row:
            print(f"- Iteration {it}: none")
            continue
        ordered = ", ".join(f"{k}={v}" for k, v in row.most_common())
        print(f"- Iteration {it}: {ordered}")


def print_patch_duplicate_analysis(trajectories: list[list[dict[str, Any]]]) -> None:
    _section("Section 8: Patch Delta / Duplicate Analysis")
    patch_values = [
        float(step.get("patch_delta", 0.0))
        for traj in trajectories
        for step in traj
        if int(step.get("iteration", 0)) >= 1
    ]
    if patch_values:
        print(
            "Patch delta (iterations 1-4): "
            f"count={len(patch_values)}, mean={mean(patch_values):.2f}, "
            f"median={_safe_median(patch_values):.2f}, min={min(patch_values):.2f}, max={max(patch_values):.2f}"
        )
    dup_per_iter: dict[int, int] = defaultdict(int)
    oscillating = 0
    null_responses = 0
    for traj in trajectories:
        for step in traj:
            it = int(step.get("iteration", 0))
            features = step.get("features", {})
            if int(features.get("is_duplicate", 0)) == 1:
                dup_per_iter[it] += 1
            if int(features.get("is_oscillating", 0)) == 1:
                oscillating += 1
            if bool(step.get("llm_null_response", False)):
                null_responses += 1
    print("Near-duplicate count (feature is_duplicate==1) by iteration:")
    for it in range(MAX_ITERATIONS):
        print(f"- iteration {it}: {dup_per_iter.get(it, 0)}")
    print(f"Oscillating flags (total): {oscillating}")
    print(f"LLM null-response flags (total): {null_responses}")


def _fmt_small_list(values: list[Any], max_items: int = 5) -> str:
    if len(values) <= max_items:
        return str(values)
    return f"{values[:max_items]}...(+{len(values) - max_items})"


def print_per_problem_detail_table(
    trajectories: list[list[dict[str, Any]]],
    outcome_details: list[dict[str, Any]],
) -> None:
    _section("Section 9: Per-Problem Detail Table")
    detail_map = {d["problem_id"]: d for d in outcome_details}
    rows = []
    for traj in trajectories:
        pid = str(traj[0].get("problem_id"))
        pass_rates = [round(float(s.get("pass_rate", 0.0)), 3) for s in traj]
        sv_scores = [round(float(s.get("self_verification_score", 0.5)), 3) for s in traj]
        nulls = [int(bool(s.get("llm_null_response", False))) for s in traj]
        waste = compute_waste_for_trajectory(traj)
        rows.append(
            {
                "problem_id": pid,
                "final_pass_rate": float(traj[-1].get("pass_rate", 0.0)),
                "max_pass_rate": max(float(s.get("pass_rate", 0.0)) for s in traj),
                "bucket": detail_map[pid]["bucket"],
                "waste": waste,
                "pass_rates": pass_rates,
                "sv_scores": sv_scores,
                "nulls": nulls,
            }
        )
    rows.sort(key=lambda r: (-r["final_pass_rate"], -r["max_pass_rate"], r["problem_id"]))
    print("problem_id | pass_rates[it0..it4] | bucket | waste | sv_scores | null_responses")
    for row in rows:
        print(
            f"{row['problem_id']:>10} | {_fmt_small_list(row['pass_rates']):<24} | "
            f"{row['bucket']:<12} | {row['waste']:.3f} | "
            f"{_fmt_small_list(row['sv_scores']):<24} | {row['nulls']}"
        )


def print_key_takeaways(
    trajectories: list[list[dict[str, Any]]],
    buckets: dict[str, list[list[dict[str, Any]]]],
) -> None:
    _section("Section 10: Key Takeaways")
    total = len(trajectories) or 1
    solved0 = len(buckets["Solved-at-0"])
    solved_later = len(buckets["Solved-later"])
    partial = len(buckets["Partial"])
    never = len(buckets["Never-passed"])
    waste = mean([compute_waste_for_trajectory(t) for t in trajectories]) if trajectories else 0.0

    labels = 0
    positives = 0
    for traj in trajectories:
        for i in range(len(traj) - 1):
            labels += 1
            if float(traj[i + 1].get("pass_rate", 0.0)) - float(traj[i].get("pass_rate", 0.0)) >= IMPROVEMENT_THRESHOLD:
                positives += 1
    scores = [float(s.get("self_verification_score", 0.5)) for t in trajectories for s in t]
    frac_half = (scores.count(0.5) / len(scores)) if scores else 0.0
    print(f"- Waste is {_fmt_pct(waste)} because many trajectories are Never-passed ({never}/{total}) or Partial ({partial}/{total}),")
    print("  and those trajectories contribute zero waste under the oracle-first definition.")
    print(f"- Solved-at-0 is {solved0}/{total}; these cases contribute all post-iteration-0 waste.")
    print(f"- Solved-later is {solved_later}/{total}; gradual repair is relatively rare.")
    print(f"- AUC labels are extremely sparse: positives={positives} of {labels} ({_fmt_pct(positives / labels) if labels else '0.00%'}).")
    print(f"- Self-verification scores at exactly 0.5: {_fmt_pct(frac_half)}; ECE should be interpreted with this caveat.")
    print("- Overall signal suggests a bimodal regime (instant solve vs stalled failure), with few informative intermediate improvements.")


def print_pilot_summary_reference() -> None:
    _section("Reference: Existing pilot_summary.json")
    if not os.path.isfile(PILOT_SUMMARY_PATH):
        print(f"Missing file: {PILOT_SUMMARY_PATH}")
        return
    with open(PILOT_SUMMARY_PATH, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    for key in ("waste_rate", "ece", "auc_mean", "auc_std"):
        value = summary.get(key)
        if isinstance(value, float) and math.isnan(value):
            printable = "NaN"
        else:
            printable = str(value)
        print(f"- {key}: {printable}")


def main() -> None:
    trajectories = load_trajectories(DATA_PATTERN)
    if not trajectories:
        print(f"No trajectories found at pattern: {DATA_PATTERN}")
        return
    print_pilot_summary_reference()
    print_dataset_overview(trajectories)
    buckets, outcome_details = print_outcome_classification(trajectories)
    print_waste_decomposition(trajectories, buckets)
    print_pass_rate_trajectory_analysis(trajectories)
    print_self_verification_audit(trajectories)
    print_label_distribution_for_auc(trajectories)
    print_error_breakdown(trajectories)
    print_patch_duplicate_analysis(trajectories)
    print_per_problem_detail_table(trajectories, outcome_details)
    print_key_takeaways(trajectories, buckets)


if __name__ == "__main__":
    main()
