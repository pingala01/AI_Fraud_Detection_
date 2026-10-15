"""SHAP explanations. Uses the shap library; falls back to XGBoost's built-in exact tree
contributions if shap's TreeExplainer cannot read the model (version mismatch)."""
import numpy as np

from .scoring import explain_family


def build_explainer(model, model_name, background=None):
    """Return (method, explainer) or None when the model type has no SHAP support here."""
    family = explain_family(model_name)
    if family is None:
        return None
    import shap
    if family == "tree":
        try:
            return ("shap-tree", shap.TreeExplainer(model))
        except Exception:
            if hasattr(model, "get_booster"):
                return ("xgboost-native", model.get_booster())
            raise
    return ("shap-linear", shap.LinearExplainer(model, background))


def shap_values(explainer, X):
    """Return (values[n_rows, n_features], base_value) for the fraud class.
    Units: log-odds for XGBoost / Logistic Regression, probability for Random Forest."""
    method, obj = explainer
    if method == "xgboost-native":
        import xgboost as xgb
        contrib = obj.predict(xgb.DMatrix(X), pred_contribs=True)
        return contrib[:, :-1], float(contrib[0, -1])
    sv = obj.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    base = np.atleast_1d(obj.expected_value)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    base = float(base[1] if len(base) > 1 else base[0])
    return sv, base


def units_for(model_name):
    return "probability" if model_name == "random_forest" else "log-odds"
