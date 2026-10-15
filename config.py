"""Central configuration. Edit values here or override them with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", ROOT / "reports"))

MODEL_FILENAME = "fraud_model.joblib"
REFERENCE_FILENAME = "reference_sample.csv"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
REFERENCE_PATH = MODEL_DIR / REFERENCE_FILENAME

RANDOM_STATE = 42
TARGET = "is_fraud"

# ---- Fields required by the project guide (see README, "Data") -------------
RAW_COLUMNS = [
    "amount", "timestamp", "transaction_type", "merchant_category",
    "country", "device_risk_score", "ip_risk_score",
]
NUMERIC_FEATURES = [
    "amount", "log_amount", "hour", "day_of_week", "is_night", "is_weekend",
    "device_risk_score", "ip_risk_score",
]
CATEGORICAL_FEATURES = ["transaction_type", "merchant_category", "country"]

TRANSACTION_TYPES = ["PAYMENT", "DEBIT", "TRANSFER", "CASH_OUT", "ONLINE_PURCHASE"]
MERCHANT_CATEGORIES = ["groceries", "restaurants", "travel", "electronics", "fuel",
                       "utilities", "health", "gambling", "crypto", "other"]
COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "AE", "CN", "RU", "NG"]

# ---- Cost-sensitive learning (ILLUSTRATIVE ASSUMPTIONS - replace with real bank figures) ----
COST_FALSE_NEGATIVE = float(os.getenv("COST_FN", 100))  # cost of a missed fraud
COST_FALSE_POSITIVE = float(os.getenv("COST_FP", 5))    # cost of a wrongly blocked/reviewed transaction

# ---- Targets from the project guide ----------------------------------------
TARGET_RECALL = 0.80           # fraud recall must be greater than this
TARGET_FPR = 0.05              # false positive rate must be below this
TARGET_LATENCY_MS = 100.0      # inference latency must be below this
THRESHOLD_FPR_CAP = 0.045      # threshold search stays under this cap (margin below 5%)

# ---- Drift monitoring -------------------------------------------------------
PSI_WARN = 0.10
PSI_ALERT = 0.25
DRIFT_WINDOW = 2000
DRIFT_MIN_ROWS = 200
DRIFT_CHECK_EVERY = 200
