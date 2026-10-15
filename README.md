# Bank Fraud Detection Using Machine Learning

Internship Project 1: real-time, explainable, cost-sensitive fraud detection for a digital bank.
Stack: Python, scikit-learn, XGBoost, SHAP, Streamlit, FastAPI, Docker (+ AWS EC2 / S3 / Lambda, Prometheus, Grafana).

> **Source note.** I only had the text of the requirements, not the PDF file itself. Compare this README with pages 4-5 of your guide and tell me about any differences.

## Verification status (read this first)

| Component | Status |
|---|---|
| Synthetic data, features, Logistic Regression, Random Forest, Isolation Forest, autoencoder, cost-based threshold, drift (PSI/KS), latency measurement, unit tests | **Run and passed** in my sandbox (see `docs/sandbox_results/`) |
| XGBoost + tuning, SMOTE, SHAP, FastAPI server, Streamlit dashboard, Docker, docker-compose, Prometheus/Grafana, S3, Lambda | **Written, syntax-checked, NOT executed.** My sandbox had no internet, so these libraries could not be installed. Expect small fixes on first run; tell me any error text. |
| AWS EC2 / S3 / Lambda | **Not deployed.** Instructions only (`docs/AWS_DEPLOYMENT.md`). |
| Package versions in `requirements*.txt` | Chosen for compatibility from my knowledge; **not install-tested**. |

**Sandbox result (baselines only, XGBoost not available):** best model logistic regression, test recall **49.7%** (target >80%: **not met**), false positive rate **2.44%** (target <5%: met), in-process latency p99 **12.5 ms** (target <100 ms: met). Your run includes tuned XGBoost, so numbers will differ; I make no claim about them. Details and trade-offs: `docs/sandbox_results/RESULTS.md`.
The synthetic data is intentionally noisy; I did not tune it to hit the targets. Real bank data would behave differently.

## Implemented vs optional

- **Implemented (required):** imbalance handling (class weights, SMOTE for one logistic variant, `scale_pos_weight` for XGBoost), Logistic Regression + Random Forest baselines, XGBoost with RandomizedSearchCV, cost-based threshold optimization, SHAP, FastAPI, Streamlit, Docker, drift detection, Prometheus/Grafana config, S3/Lambda/EC2 preparation.
- **Optional/advanced:** Isolation Forest and a shallow autoencoder (scikit-learn `MLPRegressor`, chosen to avoid adding TensorFlow/PyTorch, which are not in the required stack).

## Data

Synthetic, clearly labelled data from `src/data_generation.py` (120,000 transactions over 90 days, 0.8% fraud). It contains every required field: amount, timestamp, transaction type, merchant category, country, device risk score, IP risk score, fraud label (`is_fraud`).
It is **not real bank data**. Public datasets such as the ULB credit-card dataset (anonymised PCA features) or PaySim (simulated mobile money) lack some of these fields (e.g. country, merchant category, device/IP risk), so a mapping would be needed. To use your own CSV: `python -m src.train --data yourfile.csv` with columns `amount, timestamp, transaction_type, merchant_category, country, device_risk_score, ip_risk_score, is_fraud`.

**Cost assumptions** (`src/config.py`, illustrative only): missed fraud = 100, false alert = 5. The threshold minimizes cost while keeping validation FPR <= 4.5%. Models are chosen on validation data and reported on a later, untouched test period (chronological split).

## Folder structure

```
fraud-detection/
  README.md  requirements.txt  requirements-docker.txt  requirements-lambda.txt
  Dockerfile  Dockerfile.dashboard  Dockerfile.lambda  docker-compose.yml  .env.example
  src/        config, data_generation, features, scoring, evaluation, train, predictor, explain, drift, storage
  api/main.py             FastAPI service (+ /metrics, /drift, /reload)
  app/dashboard.py        Streamlit dashboard
  serverless/handler.py   AWS Lambda handler
  scripts/                test_prediction.py, benchmark_api.py, retrain_if_drift.py, *.bat helpers
  monitoring/             prometheus.yml, alert_rules.yml, grafana provisioning + dashboard
  samples/                example transactions (JSON)
  tests/test_core.py
  docs/AWS_DEPLOYMENT.md  docs/sandbox_results/
  data/ models/ reports/  (created/filled by training)
```

## Windows step-by-step (IDLE + Command Prompt)

### 1. Install Python
Install **64-bit Python 3.11 or 3.12** from python.org. Tick **"Add python.exe to PATH"**. IDLE is installed with it. (Avoid 3.13: SHAP may not install.)

### 2. Unzip and open Command Prompt in the project
Unzip `fraud-detection.zip` (e.g. to `C:\fraud-detection`). Press the Windows key, type `cmd`, press Enter, then:
```
cd C:\fraud-detection
python --version
```

### 3. Create a virtual environment and install dependencies
```
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
You should see `(venv)` at the start of the prompt. Reopen this each time with `cd C:\fraud-detection` and `venv\Scripts\activate`. Installation downloads a few hundred MB.
If `pip` fails on a package, copy the red error text and send it to me.

### 4. Opening and editing files in IDLE
In File Explorer right-click a `.py` file (e.g. `src\config.py`) -> **Edit with IDLE**. Or start IDLE -> **File > Open**. Edit, then **Ctrl+S** to save. Typical edits: cost values and targets in `src\config.py`.
This guide never uses IDLE's F5/Run for servers: training, the API and Streamlit are all launched from Command Prompt.

### 5. Train the model
```
python -m src.train
```
This generates synthetic data, trains all models, tunes XGBoost (can take several minutes; use `python -m src.train --quick` for a fast trial), picks a threshold, measures latency and writes `models\fraud_model.joblib` and `reports\RESULTS.md`, `metrics.json`, plots. Read `reports\RESULTS.md` for your real numbers and a list of anything skipped.

### 6. Start the FastAPI server (Command Prompt window 1)
```
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000/docs in a browser to try `POST /predict` interactively (click Try it out). Stop with **Ctrl+C**.

### 7. Test a prediction (Command Prompt window 2)
```
cd C:\fraud-detection
venv\Scripts\activate
python scripts\test_prediction.py
```
or with curl (included in Windows 10/11):
```
curl.exe -X POST "http://127.0.0.1:8000/predict?explain=true" -H "Content-Type: application/json" -d @samples\suspicious_transaction.json
```
(If you set `API_KEY`, add `-H "X-API-Key: yourkey"`.) The response has `fraud_score`, `is_fraud_flagged`, `risk_level`, `latency_ms` and a SHAP `explanation`.
Measure HTTP latency: `python scripts\benchmark_api.py`.

### 8. Launch the Streamlit dashboard (Command Prompt window 3)
```
cd C:\fraud-detection
venv\Scripts\activate
streamlit run app\dashboard.py
```
It opens http://localhost:8501. If Streamlit asks for an email on first run, just press Enter. Tabs: single transaction + SHAP explanation, batch analysis (synthetic or CSV upload), model performance, drift monitoring.

Optional shortcuts: double-click `scripts\1_train.bat`, `2_start_api.bat`, `3_start_dashboard.bat` (they open Command Prompt and run the same commands).

### 9. Run tests
```
python -m pytest -q tests
```

### 10. Docker (when Docker Desktop with WSL2 is installed)
```
docker compose run --rm trainer
docker compose up --build
```
API http://localhost:8000/docs, dashboard http://localhost:8501, Prometheus http://localhost:9090, Grafana http://localhost:3000 (user `admin`, password from `.env` `GRAFANA_ADMIN_PASSWORD`; default `admin` - change it). In Grafana open the "Fraud Detection Monitoring" dashboard. Stop with `docker compose down`.
Training inside Docker guarantees the model file matches the image's library versions. Copy `.env.example` to `.env` first if you want an API key.

## Design summary

- **Latency:** one-row scoring uses a single thread; the API measures its own scoring time (`latency_ms`, Prometheus histogram). SHAP is optional per request (`?explain=true`) so it never slows the default path.
- **Explainability:** SHAP TreeExplainer / LinearExplainer (falls back to XGBoost's exact built-in contributions if SHAP cannot read the model). Global plots are saved in `reports\`. This supports, but does not by itself satisfy, PSD2/PCI DSS expectations.
- **Adapting to new patterns:** drift (PSI per feature and score, KS test) is computed over the last 2,000 requests and exported as Prometheus gauges with alerts (`monitoring\alert_rules.yml`). `scripts\retrain_if_drift.py` retrains on a new labelled file when drift is detected and archives the old model; schedule it with Windows Task Scheduler or cron, then call `POST /reload`. Drift in inputs is detectable immediately; changes in fraud *behaviour* only show up once confirmed labels arrive. Automatic retraining needs a human or champion/challenger check before promotion - not implemented.
- **Security:** optional `X-API-Key` header (set `API_KEY`); no secrets in code; S3 uploads use server-side encryption. Add HTTPS, rate limiting and network restrictions before any real use.

## Limitations

- Synthetic data; results are for demonstration only. The test set has ~150 frauds, so recall has a wide confidence interval (shown in `RESULTS.md`).
- SMOTE on one-hot features creates non-binary values; it is compared empirically, not assumed to help.
- Cross-validation in XGBoost tuning is stratified, not time-ordered (the final evaluation is time-ordered).
- Drift state lives in API memory (resets on restart; per-instance when scaled).

## Troubleshooting

- `ModuleNotFoundError: src` -> run commands from the project folder (`cd C:\fraud-detection`).
- `'uvicorn' is not recognized` -> activate the venv (`venv\Scripts\activate`).
- API returns 503 "Model not loaded" -> run training first, then `POST /reload` or restart.
- Port in use -> use `--port 8001`.
- Pickle/version errors loading the model -> retrain in the same environment that loads it.
