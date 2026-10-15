"""Synthetic transaction generator.

IMPORTANT: this is SYNTHETIC data created for demonstration. It is not real bank data,
and results on it say nothing certain about performance on real transactions.
Fraud rows follow partly different distributions from normal rows, but the groups
deliberately overlap (stealthy frauds, odd-looking legitimate rows).
"""
import argparse

import numpy as np
import pandas as pd

from . import config as C


def _w(weights):
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def _pick(rng, values, p_legit, p_fraud, fraud_profile):
    out = np.empty(len(fraud_profile), dtype=object)
    f = fraud_profile
    out[~f] = rng.choice(values, size=int((~f).sum()), p=_w(p_legit))
    out[f] = rng.choice(values, size=int(f.sum()), p=_w(p_fraud))
    return out


def generate_synthetic_transactions(n_rows=120_000, fraud_rate=0.008, seed=C.RANDOM_STATE,
                                    start="2025-01-01", days=90, drift=False):
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n_rows * fraud_rate))
    y = np.zeros(n_rows, dtype=int)
    y[:n_fraud] = 1
    rng.shuffle(y)
    is_f = y == 1

    # 20% of frauds are "stealthy" (rarely follow the fraud profile); ~8% of legit rows look odd.
    adherence = np.where(rng.random(n_rows) < 0.20, 0.20, 0.65)

    def profile():
        return np.where(is_f, rng.random(n_rows) < adherence, rng.random(n_rows) < 0.08)

    p_type, p_cat, p_cty, p_hour, p_amt, p_dev, p_ip = (profile() for _ in range(7))

    ttype = _pick(rng, C.TRANSACTION_TYPES, [50, 20, 12, 8, 10], [10, 5, 30, 25, 30], p_type)
    mcat = _pick(rng, C.MERCHANT_CATEGORIES, [22, 18, 6, 6, 12, 10, 8, 1, 0.5, 16],
                 [3, 4, 14, 22, 3, 2, 2, 18, 20, 12], p_cat)
    country = _pick(rng, C.COUNTRIES, [45, 15, 10, 8, 8, 4, 4, 3, 2, 1],
                    [20, 8, 5, 5, 7, 12, 8, 10, 13, 12], p_cty)
    hour = _pick(rng, np.arange(24),
                 [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 9, 9, 8, 8, 8, 9, 9, 8, 6, 4, 3, 2],
                 [6, 6, 6, 5, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6],
                 p_hour).astype(int)
    amount = np.where(p_amt, rng.lognormal(4.4, 1.2, n_rows), rng.lognormal(3.5, 1.0, n_rows))
    device = np.where(p_dev, rng.beta(5, 3, n_rows), rng.beta(2, 8, n_rows))
    ip = np.where(p_ip, rng.beta(4, 3, n_rows), rng.beta(2, 9, n_rows))

    if drift:  # simulate a changed customer / fraud landscape for the drift demo
        amount = amount * 1.8
        device = np.where(~is_f, np.clip(device + 0.12, 0, 1), device)
        mcat = np.where(rng.random(n_rows) < 0.30, "crypto", mcat)

    day = rng.integers(0, days, n_rows)
    seconds = day * 86400 + hour * 3600 + rng.integers(0, 60, n_rows) * 60 + rng.integers(0, 60, n_rows)
    ts = pd.Timestamp(start) + pd.to_timedelta(seconds, unit="s")

    df = pd.DataFrame({
        "timestamp": ts,
        "amount": np.round(np.clip(amount, 0.5, None), 2),
        "transaction_type": ttype,
        "merchant_category": mcat,
        "country": country,
        "device_risk_score": np.round(device, 4),
        "ip_risk_score": np.round(ip, 4),
        C.TARGET: y,
    }).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TX{i:08d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write a synthetic dataset to the data folder")
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--fraud-rate", type=float, default=0.008)
    ap.add_argument("--drift", action="store_true", help="generate a drifted dataset for demos")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = generate_synthetic_transactions(a.rows, a.fraud_rate, drift=a.drift, seed=C.RANDOM_STATE + (1 if a.drift else 0))
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = a.out or str(C.DATA_DIR / ("synthetic_drifted.csv" if a.drift else "synthetic_transactions.csv"))
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} rows ({data[C.TARGET].mean():.3%} fraud) to {out}")
