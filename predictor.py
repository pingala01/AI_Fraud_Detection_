"""Load the saved model bundle and score transactions (used by API, dashboard and Lambda)."""
import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import add_features
from .scoring import raw_scores


class FraudPredictor:
    def __init__(self, path=None):
        b = joblib.load(path or C.MODEL_PATH)
        self.bundle = b
        self.pre = b["preprocessor"]
        self.model = b["model"]
        self.kind = b["score_kind"]
        self.name = b["model_name"]
        self.threshold = float(b["threshold"])
        self.feature_names = b["feature_names"]
        self.trained_at = b.get("trained_at")
        self._lo, self._hi = b.get("score_lo", 0.0), b.get("score_hi", 1.0)
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1          # avoid thread start-up overhead on single-row requests
        self._explainer = None

    # ---- scoring ---------------------------------------------------------
    def _transform(self, df):
        return self.pre.transform(add_features(df[C.RAW_COLUMNS]))

    def risk_from_raw(self, raw):
        if self.kind == "proba":
            return np.asarray(raw, dtype=float)
        return np.clip((np.asarray(raw) - self._lo) / max(self._hi - self._lo, 1e-12), 0, 1)

    def risk_level(self, raw):
        return "HIGH" if raw >= self.threshold else "MEDIUM" if raw >= 0.5 * self.threshold else "LOW"

    def predict_df(self, df):
        raw = raw_scores(self.kind, self.model, self._transform(df))
        out = pd.DataFrame({"raw_score": raw, "risk_score": self.risk_from_raw(raw)}, index=df.index)
        out["is_flagged"] = out["raw_score"] >= self.threshold
        out["risk_level"] = [self.risk_level(r) for r in raw]
        return out

    def predict_one(self, record: dict) -> dict:
        res = self.predict_df(pd.DataFrame([record])).iloc[0]
        return {
            "fraud_score": float(res["risk_score"]), "raw_score": float(res["raw_score"]),
            "is_fraud_flagged": bool(res["is_flagged"]), "risk_level": res["risk_level"],
            "threshold": self.threshold, "model_name": self.name,
            "score_type": "probability" if self.kind == "proba" else "normalized_anomaly_score",
        }

    # ---- explanations ----------------------------------------------------
    def _get_explainer(self):
        if self._explainer is None:
            from .explain import build_explainer
            self._explainer = build_explainer(self.model, self.name, self.bundle.get("background")) or False
        return self._explainer or None

    def explain_one(self, record: dict, top_k=5):
        ex = self._get_explainer()
        if ex is None:
            return None
        from .explain import shap_values, units_for
        df = pd.DataFrame([record])
        X = self._transform(df)
        sv, base = shap_values(ex, X)
        feats = add_features(df[C.RAW_COLUMNS]).iloc[0]
        order = np.argsort(-np.abs(sv[0]))[:top_k]
        items = []
        for i in order:
            name = self.feature_names[i]
            items.append({"feature": name, "display": self._describe(name, X[0, i], feats),
                          "shap_value": float(sv[0, i]),
                          "direction": "increases_risk" if sv[0, i] > 0 else "decreases_risk"})
        return {"method": ex[0], "units": units_for(self.name), "base_value": base,
                "contributions": items}

    @staticmethod
    def _describe(name, x_value, feats):
        if name in C.NUMERIC_FEATURES:
            v = feats[name]
            return f"{name} = {v:.4g}" if isinstance(v, (int, float, np.number)) else f"{name} = {v}"
        for cat in C.CATEGORICAL_FEATURES:
            if name.startswith(cat + "_"):
                val = name[len(cat) + 1:]
                return f"{cat} = {val}" if x_value > 0.5 else f"{cat} != {val}"
        return name
