"""Metrics and cost-sensitive threshold selection."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def wilson_interval(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def choose_threshold(y_true, scores, c_fn, c_fp, max_fpr):
    """Lowest expected-cost threshold whose false positive rate is <= max_fpr.

    cost = c_fn * missed frauds + c_fp * false alerts. Flag when score >= threshold.
    """
    y = np.asarray(y_true).astype(int)
    fpr, tpr, thr = roc_curve(y, scores, drop_intermediate=False)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    cost = c_fn * pos * (1 - tpr) + c_fp * neg * fpr
    ok = np.where(fpr <= max_fpr)[0]
    best = ok[np.argmin(cost[ok])]
    t = float(thr[best])
    if not np.isfinite(t):
        t = float(np.max(scores) + 1e-9)
    return t


def evaluate_scores(y_true, scores, threshold, c_fn, c_fp):
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(scores) >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    cost = c_fn * fn + c_fp * fp
    baseline = c_fn * (tp + fn)
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": float(recall), "recall_ci95": wilson_interval(tp, tp + fn),
        "fpr": float(fpr), "fpr_ci95": wilson_interval(fp, fp + tn),
        "precision": float(precision), "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, scores)), "pr_auc": float(average_precision_score(y, scores)),
        "cost": float(cost), "cost_if_nothing_flagged": float(baseline),
        "cost_saving_vs_flag_nothing": float(1 - cost / baseline) if baseline else float("nan"),
        "alert_rate": float(pred.mean()),
    }
