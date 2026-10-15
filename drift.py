"""Data-drift detection using the Population Stability Index (PSI) plus a KS test.

PSI rule of thumb (industry convention, not a law): <0.10 stable, 0.10-0.25 watch, >0.25 significant shift.
Only input/score drift can be measured instantly; true fraud labels arrive late (chargebacks),
so label-based performance monitoring must be done separately when labels become available.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config as C
from .features import add_features

MONITORED_NUMERIC = ["amount", "hour", "device_risk_score", "ip_risk_score"]
MONITORED_CATEGORICAL = ["transaction_type", "merchant_category", "country"]
EPS = 1e-4


def status_from_psi(psi):
    if psi is None or not np.isfinite(psi):
        return "unknown"
    return "alert" if psi >= C.PSI_ALERT else "warning" if psi >= C.PSI_WARN else "ok"


def _psi(ref_p, cur_p):
    ref_p = np.clip(ref_p, EPS, None)
    cur_p = np.clip(cur_p, EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_numeric(ref, cur, bins=10):
    ref = np.asarray(ref, dtype=float); cur = np.asarray(cur, dtype=float)
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant reference feature
        return 0.0 if np.allclose(cur, edges[0]) else 0.5
    inner = edges[1:-1]
    ref_c = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(edges) - 1)
    cur_c = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(edges) - 1)
    return _psi(ref_c / ref_c.sum(), cur_c / cur_c.sum())


def psi_categorical(ref, cur):
    ref = pd.Series(ref).astype(str); cur = pd.Series(cur).astype(str)
    cats = sorted(set(ref) | set(cur))
    ref_p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_p = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_p, cur_p)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Compare a current batch with the training-time reference sample (raw columns, optional 'score')."""
    ref = add_features(reference) if "hour" not in reference else reference
    cur = add_features(current) if "hour" not in current else current
    feats = {}
    numeric = list(MONITORED_NUMERIC)
    if "score" in ref.columns and "score" in cur.columns:
        numeric.append("score")
    for col in numeric:
        psi = psi_numeric(ref[col], cur[col])
        ks = ks_2samp(ref[col].astype(float), cur[col].astype(float))
        feats[col] = {"type": "numeric", "psi": psi, "ks_stat": float(ks.statistic),
                      "ks_pvalue": float(ks.pvalue), "status": status_from_psi(psi)}
    for col in MONITORED_CATEGORICAL:
        psi = psi_categorical(ref[col], cur[col])
        feats[col] = {"type": "categorical", "psi": psi, "status": status_from_psi(psi)}
    max_psi = max((v["psi"] for v in feats.values() if np.isfinite(v["psi"])), default=float("nan"))
    return {
        "n_reference": int(len(ref)), "n_current": int(len(cur)),
        "features": feats, "max_psi": float(max_psi),
        "overall_status": status_from_psi(max_psi),
        "drifted_features": [k for k, v in feats.items() if v["status"] == "alert"],
        "thresholds": {"warning": C.PSI_WARN, "alert": C.PSI_ALERT},
    }
