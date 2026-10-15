"""Turn a fitted model into one numeric fraud score (higher = more suspicious)."""


def raw_scores(kind, model, X):
    if kind == "proba":                       # Logistic Regression, Random Forest, XGBoost
        return model.predict_proba(X)[:, 1]
    if kind == "isolation_forest":            # anomaly score: higher = more anomalous
        return -model.score_samples(X)
    if kind == "autoencoder":                 # reconstruction error
        return ((X - model.predict(X)) ** 2).mean(axis=1)
    raise ValueError(f"Unknown model kind: {kind}")


def explain_family(model_name):
    if model_name in ("random_forest", "xgboost"):
        return "tree"
    if model_name.startswith("logistic_regression"):
        return "linear"
    return None
