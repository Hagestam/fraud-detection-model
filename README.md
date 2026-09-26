# Financial Anomaly Detector

Hybrid fraud detection using XGBoost and Isolation Forest. The training pipeline selects the score blend and decision threshold on validation data, evaluates on a held-out test split, and saves a model for the API.

## Requirements

- Python 3.11 or 3.12
- Kaggle account with access to the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection)

## Install

Run these commands from the project folder:

```
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Get the data

Sign in to Kaggle, open the competition page above, and accept its rules. Then download and extract the files:

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
New-Item -ItemType Directory -Force ".\data\ieee-fraud-detection" | Out-Null
.\.venv\Scripts\kaggle.exe auth login
.\.venv\Scripts\kaggle.exe competitions download ieee-fraud-detection -p ".\data\ieee-fraud-detection"
Expand-Archive ".\data\ieee-fraud-detection\ieee-fraud-detection.zip" -DestinationPath ".\data\ieee-fraud-detection" -Force
```

## Train

This example trains from the competition's transaction table. `TransactionDT` is used for chronological splitting, and `TransactionID` is excluded from model features.

```powershell
.\.venv\Scripts\fraud-train.exe --data ".\data\ieee-fraud-detection\train_transaction.csv" --target isFraud --timestamp TransactionDT --exclude TransactionID --output ".\artifacts\ieee-cis-v1"
```

Training metrics and the model manifest are saved with the artifacts. MLflow run data defaults to the local `mlruns` folder. To see all options, run `.\.venv\Scripts\fraud-train.exe --help`.

## Run the API

Start the API using the artifacts created above:

```powershell
$env:MODEL_DIR = ".\artifacts\ieee-cis-v1"
.\.venv\Scripts\uvicorn.exe fraud_detection.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API. Health checks are available at `/health/live` and `/health/ready`; submit predictions to `POST /v1/predict`.

## Docker

Compose expects the model at `artifacts/current`. Train with `--output ".\artifacts\current"` (or update `MODEL_DIR` in `docker-compose.yml` to match your output directory), then run `docker compose up --build` from the project folder. Compose starts the API and Redis.
