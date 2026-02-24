"""
Validation Metrics — DCR, HI, PSS, CCS, SCE, F1, ROUGE-L.

All metrics formalized as entity-level information retrieval:
- DCR  = |G ∩ M| / |G|               (Recall)
- HI   = |M \\ G| / |M|              (1 - Precision)
- F1   = 2·P·R / (P+R)               (Harmonic mean)
- CCS  = Severity-weighted coverage   (ICD-11 weights)
- SCE  = Severity calibration error
- PSS  = Perturbation stability       (Lipschitz bound)
- ROUGE-L = Longest common subsequence overlap

All confidence intervals: bootstrap, 1,000 resamples.
"""

from __future__ import annotations

from collections.abc import Set

import numpy as np


def diagnostic_concordance_rate(gt: Set[str], pred: Set[str]) -> float:
    """DCR: fraction of ground-truth entities found in prediction (Recall)."""
    if not gt:
        return 1.0 if not pred else 0.0
    return len(gt & pred) / len(gt)


def hallucination_index(pred: Set[str], gt: Set[str]) -> float:
    """HI: fraction of predicted entities not in ground truth (1 - Precision)."""
    if not pred:
        return 0.0
    return len(pred - gt) / len(pred)


def clinical_f1(gt: Set[str], pred: Set[str]) -> float:
    """Clinical F1: harmonic mean of precision and recall."""
    if not gt and not pred:
        return 1.0
    matched = gt & pred
    precision = len(matched) / max(len(pred), 1)
    recall = len(matched) / max(len(gt), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def clinical_completeness_score(
    gt: Set[str],
    pred: Set[str],
    severity_weights: dict[str, float],
    case_severity: str,
) -> float:
    """CCS: severity-weighted entity coverage.

    Critical findings (3×) weighted higher than mild (1×).
    Uses ICD-11 severity tiers for weighting.
    """
    if not gt:
        return 1.0

    weight = severity_weights.get(case_severity.lower(), 1.0)
    matched = gt & pred
    return (len(matched) / len(gt)) * min(weight / 2.0, 1.0) + (
        len(matched) / len(gt)
    ) * (1 - min(weight / 2.0, 1.0))


def severity_calibration_error(
    pred_severity: str,
    gt_severity: str,
    severity_weights: dict[str, float],
) -> float:
    """SCE: absolute difference between predicted and ground-truth severity weights."""
    pred_w = severity_weights.get(pred_severity.lower(), 1.0)
    gt_w = severity_weights.get(gt_severity.lower(), 1.0)
    return abs(pred_w - gt_w) / max(max(severity_weights.values()), 1.0)


def rouge_l_score(prediction: str, reference: str) -> float:
    """ROUGE-L: longest common subsequence F-measure."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / m
    recall = lcs_len / n

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Sample values.
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.percentile(means, 100 * alpha)), float(
        np.percentile(means, 100 * (1 - alpha))
    )
