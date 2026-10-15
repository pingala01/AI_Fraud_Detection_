"""Streamlit dashboard. Start from Command Prompt (project folder):  streamlit run app\\dashboard.py"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config as C
from src.data_generation import generate_synthetic_transactions
from src.drift import drift_report
from src.predictor import FraudPredictor

st.set_page_config(page_title="Bank Fraud Detection", layout="wide")
st.title("Bank Fraud Detection Dashboard")
st.caption("Demo built on SYNTHETIC data unless you load your own file. Not real bank data.")


@st.cache_resource
def load_predictor():
    return FraudPredictor(C.MODEL_PATH) if C.MODEL_PATH.exists() else None


predictor = load_predictor()
if predictor is None:
    st.error("No trained model found. In Command Prompt run:  python -m src.train   then refresh this page.")
    st.stop()

st.sidebar.header("Model")
st.sidebar.write(f"**{predictor.name}**")
st.sidebar.write(f"Decision threshold: `{predictor.threshold:.4f}`")
st.sidebar.write(f"Trained: {predictor.trained_at}")

tab1, tab2, tab3, tab4 = st.tabs(["Single transaction", "Batch analysis", "Model performance", "Drift monitoring"])

# ------------------------------------------------------------------ single
with tab1:
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Amount", min_value=0.01, value=250.0)
        d = c1.date_input("Date", value=datetime.now().date())
        t = c1.time_input("Time", value=datetime.now().time().replace(microsecond=0))
        ttype = c2.selectbox("Transaction type", C.TRANSACTION_TYPES, index=2)
        mcat = c2.selectbox("Merchant category", C.MERCHANT_CATEGORIES)
        country = c2.selectbox("Country", C.COUNTRIES)
        dev = c3.slider("Device risk score", 0.0, 1.0, 0.30)
        ip = c3.slider("IP risk score", 0.0, 1.0, 0.30)
        go = st.form_submit_button("Score transaction")
    if go:
        rec = {"amount": amount, "timestamp": datetime.combine(d, t).isoformat(), "transaction_type": ttype,
               "merchant_category": mcat, "country": country, "device_risk_score": dev, "ip_risk_score": ip}
        res = predictor.predict_one(rec)
        a, b, c = st.columns(3)
        a.metric("Fraud score", f"{res['fraud_score']:.3f}")
        b.metric("Decision", "FLAGGED" if res["is_fraud_flagged"] else "Allowed")
        c.metric("Risk level", res["risk_level"])
        try:
            ex = predictor.explain_one(rec, top_k=8)
        except Exception as exc:
            ex = None
            st.info(f"SHAP explanation unavailable ({exc}). Install shap (pip install -r requirements.txt).")
        if ex:
            st.subheader("Why this score? (SHAP)")
            edf = pd.DataFrame(ex["contributions"])
            st.bar_chart(edf.set_index("display")["shap_value"])
            st.caption(f"Positive values push toward fraud. Units: {ex['units']}.")
            st.dataframe(edf[["display", "shap_value", "direction"]], use_container_width=True)

# ------------------------------------------------------------------ batch
with tab2:
    src = st.radio("Data source", ["Generate synthetic demo batch", "Upload CSV"], horizontal=True)
    batch = None
    if src.startswith("Generate"):
        n = st.slider("Rows", 500, 20000, 3000, 500)
        rate = st.slider("Fraud rate in demo batch (%)", 0.5, 10.0, 2.0, 0.5)
        drifted = st.checkbox("Simulate drifted data (changed amounts / merchants)")
        if st.button("Generate and score"):
            batch = generate_synthetic_transactions(n, rate / 100, seed=int(datetime.now().timestamp()) % 10000, drift=drifted)
    else:
        up = st.file_uploader("CSV with columns: " + ", ".join(C.RAW_COLUMNS), type="csv")
        if up is not None:
            batch = pd.read_csv(up)
            missing = [c for c in C.RAW_COLUMNS if c not in batch.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                batch = None
    if batch is not None:
        scored = pd.concat([batch.reset_index(drop=True), predictor.predict_df(batch).reset_index(drop=True)], axis=1)
        st.session_state["batch"] = scored
    scored = st.session_state.get("batch")
    if scored is not None:
        flagged = scored[scored["is_flagged"]]
        a, b, c = st.columns(3)
        a.metric("Transactions", f"{len(scored):,}")
        b.metric("Flagged", f"{len(flagged):,}")
        c.metric("Flag rate", f"{len(flagged) / len(scored):.2%}")
        if C.TARGET in scored.columns:
            y = scored[C.TARGET].astype(int)
            tp = int(((y == 1) & scored["is_flagged"]).sum()); fn = int(((y == 1) & ~scored["is_flagged"]).sum())
            fp = int(((y == 0) & scored["is_flagged"]).sum()); tn = int(((y == 0) & ~scored["is_flagged"]).sum())
            d1, d2, d3 = st.columns(3)
            d1.metric("Recall (this batch)", f"{tp / max(tp + fn, 1):.1%}")
            d2.metric("False positive rate", f"{fp / max(fp + tn, 1):.2%}")
            d3.metric("Precision", f"{tp / max(tp + fp, 1):.1%}")
        st.subheader("Highest-risk transactions")
        st.dataframe(scored.sort_values("raw_score", ascending=False).head(50), use_container_width=True)
        st.download_button("Download flagged transactions (CSV)", flagged.to_csv(index=False), "flagged.csv")
        st.subheader("Score distribution")
        st.bar_chart(pd.cut(scored["risk_score"], 20).value_counts().sort_index().rename(lambda i: f"{i.left:.2f}"))
        if len(flagged):
            st.subheader("Explain a flagged transaction")
            top = flagged.sort_values("raw_score", ascending=False).head(50)
            pick = st.selectbox("Row", top.index.tolist(), format_func=lambda i: f"row {i} (score {top.loc[i, 'risk_score']:.3f})")
            rec = {k: top.loc[pick, k] for k in C.RAW_COLUMNS}
            rec["timestamp"] = str(rec["timestamp"])
            try:
                ex = predictor.explain_one(rec, top_k=8)
                if ex:
                    st.dataframe(pd.DataFrame(ex["contributions"])[["display", "shap_value", "direction"]])
            except Exception as exc:
                st.info(f"SHAP explanation unavailable ({exc}).")

# ------------------------------------------------------------------ performance
with tab3:
    mp = C.REPORT_DIR / "metrics.json"
    if not mp.exists():
        st.info("No reports yet - run training first.")
    else:
        m = json.loads(mp.read_text(encoding="utf-8"))
        st.write(f"Data: {m['data']['source']}")
        st.write(f"Selected model: **{m['selected_model']}**")
        tm, lat = m["selected_test_metrics"], m["latency"]
        st.table(pd.DataFrame([
            {"Target": f"Recall > {m['targets']['recall_gt']:.0%}", "Measured": f"{tm['recall']:.1%}", "Met": m["targets_met"]["recall"]},
            {"Target": f"FPR < {m['targets']['fpr_lt']:.0%}", "Measured": f"{tm['fpr']:.2%}", "Met": m["targets_met"]["fpr"]},
            {"Target": f"Latency < {m['targets']['latency_ms_lt']:.0f} ms (p99)", "Measured": f"{lat['p99_ms']:.1f} ms", "Met": m["targets_met"]["latency_p99"]},
        ]))
        cmp_path = C.REPORT_DIR / "model_comparison.csv"
        if cmp_path.exists():
            st.dataframe(pd.read_csv(cmp_path), use_container_width=True)
        for img in ("pr_curves.png", "shap_importance_bar.png", "shap_summary.png"):
            if (C.REPORT_DIR / img).exists():
                st.image(str(C.REPORT_DIR / img), caption=img)
        if m["skipped_components"]:
            st.warning("Skipped in the last training run:\n\n" + "\n".join(f"- {s}" for s in m["skipped_components"]))

# ------------------------------------------------------------------ drift
with tab4:
    if not C.REFERENCE_PATH.exists():
        st.info("No reference sample found - train a model first.")
    elif st.session_state.get("batch") is None:
        st.info("Score a batch in the 'Batch analysis' tab first (try the drifted-data option).")
    else:
        ref = pd.read_csv(C.REFERENCE_PATH)
        cur = st.session_state["batch"].copy()
        cur["score"] = cur["raw_score"]
        rep = drift_report(ref, cur)
        st.metric("Overall drift status", rep["overall_status"].upper(), f"max PSI {rep['max_psi']:.3f}")
        st.caption(f"PSI < {C.PSI_WARN}: ok | {C.PSI_WARN}-{C.PSI_ALERT}: warning | >= {C.PSI_ALERT}: alert")
        st.dataframe(pd.DataFrame(rep["features"]).T, use_container_width=True)
