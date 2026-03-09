"""
Pilot metrics: waste rate, self-verification ECE, feature AUC. Prints summary table.
"""

from __future__ import annotations

import numpy as np

from config import IMPROVEMENT_THRESHOLD, MAX_ITERATIONS, PASS_THRESHOLD


def compute_waste_rate(trajectories: list[list[dict]]) -> float:
    """
    Waste rate = mean over trajectories of (iterations after oracle-first stop) / (max repair iters).

    Args:
        trajectories: List of trajectory lists; each step has iteration, pass_rate.

    Returns:
        Mean waste rate in [0, 1].
    """
    waste = []
    for traj in trajectories:
        oracle = next(
            (s["iteration"] for s in traj if s["pass_rate"] >= PASS_THRESHOLD),
            MAX_ITERATIONS - 1,
        )
        waste.append(((MAX_ITERATIONS - 1) - oracle) / (MAX_ITERATIONS - 1))
    return float(np.mean(waste))


def compute_ece(
    trajectories: list[list[dict]],
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error: confidence = self_verification_score, outcome = binary solve.

    Args:
        trajectories: List of trajectory lists; each step has self_verification_score, pass_rate.
        n_bins: Number of equal-width bins.

    Returns:
        ECE (scalar).
    """
    confs, outcomes = [], []
    for traj in trajectories:
        for step in traj:
            confs.append(step["self_verification_score"])
            outcomes.append(int(step["pass_rate"] >= PASS_THRESHOLD))

    confs = np.array(confs)
    outcomes = np.array(outcomes)
    if len(confs) == 0:
        return 0.0

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confs >= lo) & (confs < hi)
        else:
            mask = (confs >= lo) & (confs <= 1.0)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(confs)) * abs(
            confs[mask].mean() - outcomes[mask].mean()
        )
    return float(ece)


def compute_feature_auc(trajectories: list[list[dict]]) -> tuple[float, float]:
    """
    AUC for predicting "next iteration improves" from features. Label imbalance handled by scale_pos_weight.

    Args:
        trajectories: List of trajectory lists; each step has features, pass_rate.

    Returns:
        (mean_auc, std_auc) from 5-fold CV.
    """
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - 1):
            X.append(list(traj[i]["features"].values()))
            y.append(
                int(
                    traj[i + 1]["pass_rate"] - traj[i]["pass_rate"]
                    >= IMPROVEMENT_THRESHOLD
                )
            )

    X = np.array(X)
    y = np.array(y)
    n_pos = int(y.sum())
    print(f"  Labels: {n_pos} positive / {len(y)} total")
    if n_pos < 20:
        print("  WARNING: <20 positive labels — treat AUC as directional only")

    if len(y) == 0 or (n_pos == 0 or n_pos == len(y)):
        return 0.5, 0.0

    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        scale_pos_weight=max((len(y) - n_pos) / max(n_pos, 1), 1),
        n_estimators=100,
        max_depth=4,
        random_state=42,
        eval_metric="auc",
    )
    aucs = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(aucs.mean()), float(aucs.std())


def print_summary(
    waste_rate: float,
    ece: float,
    auc_mean: float,
    auc_std: float,
) -> None:
    """
    Print pilot summary table and GREEN/YELLOW/RED decision.

    Args:
        waste_rate: From compute_waste_rate.
        ece: From compute_ece.
        auc_mean: Mean AUC from compute_feature_auc.
        auc_std: Std AUC from compute_feature_auc.
    """
    rows = [
        ("Waste Rate", f"{waste_rate:.1%}", ">25%", waste_rate > 0.25),
        ("Self-Verification ECE", f"{ece:.3f}", ">0.200", ece > 0.2),
        (
            f"Feature AUC (±{auc_std:.3f})",
            f"{auc_mean:.3f}",
            ">0.650",
            auc_mean > 0.65,
        ),
    ]
    print("\n" + "=" * 58)
    print(f"{'Metric':<28} {'Value':>8}  {'Threshold':>9}  Status")
    print("-" * 58)
    for name, val, thresh, green in rows:
        print(f"{name:<28} {val:>8}  {thresh:>9}  {'GREEN' if green else 'RED'}")
    print("=" * 58)
    greens = sum(r[3] for r in rows)
    decision = {
        3: "\nDECISION: GREEN — Full COLM study, start immediately",
        2: "\nDECISION: YELLOW — Characterization paper, target EMNLP/TMLR",
        1: "\nDECISION: RED — Kill topic, move to Topic 2 or 5",
        0: "\nDECISION: RED — Kill topic, move to Topic 2 or 5",
    }
    print(decision[greens])
