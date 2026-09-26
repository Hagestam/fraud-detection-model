# Real-Time Financial Anomaly Detection Pipeline

A production-oriented reference implementation for binary financial anomaly detection. It combines a supervised XGBoost fraud probability with an Isolation Forest normality score, uses held-out validation data to select the blend and operating threshold, reports untouched test metrics, and returns per-feature SHAP contributions for XGBoost.

**The résumé figures in the request are not results of this repository.** No transaction dataset was present, so the reported 590K rows, 94.2% F1, and sub-50 ms latency have not been reproduced. This project reports measured test metrics and per-request inference time only after you provide and run your actual dataset.

## Data contract

Supply a CSV or Parquet file with one row per transaction. The target column must contain `0` for legitimate and `1` for fraud. Every other included column is treated as an input feature unless explicitly excluded. Pass a timestamp with `--timestamp` to make chronological train/validation/test partitions; that timestamp is used for splitting and excluded from model features. Exclude identifiers and post-outcome fields explicitly so the model cannot memorize IDs or use information unavailable at decision time.

The trainer infers numeric versus categorical features from the loaded data. Numeric values are median-imputed and scaled; missing categorical values receive an explicit sentinel and are one-hot encoded with infrequent categories grouped. Keep target-derived, post-authorization, and otherwise unavailable-at-inference columns out of the feature set. Use domain review to establish that contract; code cannot infer causal availability from column names reliably.

## Train

Python 3.11 or 3.12 is supported. Install from the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
```

Example invocation (replace column names and file path with your actual schema):

```bash
fraud-train --data data/transactions.parquet --target is_fraud --timestamp transaction_time --exclude transaction_id
```

If no trustworthy event timestamp is available, omit `--timestamp`; the trainer then uses stratified random splits and logs a warning. Do not describe that evaluation as a forward-in-time result. The default partition is 70% train, 15% validation, and 15% test. Preprocessing is fit only on training data. XGBoost early stopping and the hybrid-score blend/threshold use validation data. The test partition is scored once at that fixed validation-selected operating point. MLflow records run parameters, test metrics, and the serving artifacts (`MLFLOW_TRACKING_URI` defaults to local `./mlruns`).

The Isolation Forest fits only the legitimate rows in the training partition. Its decision scores are converted to empirical anomaly percentiles using those same legitimate training rows. XGBoost uses class weighting derived from training prevalence. The validation-selected score is `(1 - if_weight) * fraud_probability + if_weight * isolation_forest_score`; set `--if-weight` to lock a chosen weight instead of tuning the 0-to-1 grid. F1 threshold selection optimizes the validation labels and can overfit; choose the operating point based on business costs and monitor precision/recall after deployment.

Artifacts are written to `artifacts/current` by default. The manifest is written last so a partial artifact write is not marked ready. Use an immutable output directory per release and point `MODEL_DIR` at that release for safer deployment/rollback. These artifacts include Python joblib files: load only artifacts produced by a trusted training pipeline and keep dependency versions aligned with the manifest.

## Serve locally

```bash
$env:MODEL_DIR = "artifacts/current"
$env:REDIS_URL = "redis://localhost:6379/0" # Optional; API still works without Redis.
uvicorn fraud_detection.api:app --host 0.0.0.0 --port 8000 --workers 1
```

The request shape is intentionally generic; use the feature names in your trained manifest:

```json
{
  "transaction": {
    "amount": 125.5,
    "merchant_category": "groceries",
    "account_age_days": 420
  }
}
```

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "content-type: application/json" \
  -d '{"transaction":{"amount":125.5,"merchant_category":"groceries","account_age_days":420}}'
```

The response includes XGBoost probability, calibrated Isolation Forest anomaly percentile, their configured hybrid score, thresholded decision, model version, measured uncached inference time, and all transformed-feature SHAP contributions for the XGBoost raw margin. One-hot encoded features map back to their source input field. SHAP explains the XGBoost component; it does not explain the Isolation Forest or the blended score. Unknown input fields return HTTP 422. Missing trained fields are passed to the fitted imputers.

Redis caches deterministic predictions by model version and canonical request body for the configured TTL. Cache failures do not block inference. The cache is an optimization and should not be used as an audit store. Do not put sensitive transaction data in Redis without the access controls, network isolation, and retention policy your deployment requires.

## Docker

Train artifacts first, then:

```bash
docker compose up --build
```

The API mounts `./artifacts` read-only at `/models`, defaults to `/models/current`, runs as a non-root user, and uses Redis as a bounded-memory disposable cache. Change `MODEL_DIR` in `docker-compose.yml` or via Compose environment if you deploy a versioned artifact directory. For multiple API workers or replicas, benchmark memory and CPU use: each process loads its own preprocessor, models, and SHAP explainer. The example deliberately runs one worker per container.

## Evaluation and operations

`manifest.json` contains the split method, selected blend and threshold, test precision, recall, F1, average precision, ROC AUC, and confusion counts, plus library versions. Use average precision and precision/recall at a stated threshold for imbalanced fraud data; accuracy alone is not informative. Compare results to a simple baseline and evaluate over time-based holdouts before making claims. Recalibrate the threshold when fraud prevalence, review capacity, or error costs change.

The API's `inference_ms` covers uncached preprocessing, both models, and SHAP attribution inside the process; it excludes network and queue time. Benchmark p50/p95/p99 under representative concurrency and realistic feature cardinality before claiming a latency SLA. SHAP for every request has real compute and response-size cost, so the sub-50 ms figure is a target to measure on deployment hardware, not a guarantee of this code.

For a load sample, provide at least as many distinct representative requests as the requested sample count in a JSONL file (each line uses the same `{"transaction": {...}}` body as the API), then run `fraud-benchmark --input requests.jsonl --requests 1000 --concurrency 8`. It reports API round-trip and uncached server inference percentiles separately, throughput, model versions, and cache-hit count. Use realistic category cardinality and disable Redis or supply unique requests when measuring uncached inference. This lightweight driver is for a deployment smoke benchmark; use a dedicated load-testing tool for capacity tests.

Health endpoints: `GET /health/live` and `GET /health/ready`. Interactive API docs: `/docs`.
