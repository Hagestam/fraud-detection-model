from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("fraud_detection.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a leakage-aware hybrid fraud detector.")
    parser.add_argument("--data", type=Path, required=True, help="CSV or Parquet file")
    parser.add_argument("--target", required=True, help="Binary target column; positive class must be 1")
    parser.add_argument("--timestamp", help="Optional event-time column for chronological splitting")
    parser.add_argument("--exclude", action="append", default=[], help="Non-feature column; repeat as needed")
    parser.add_argument("--output", type=Path, default=Path("artifacts/current"))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    parser.add_argument("--experiment", default="financial-anomaly-detection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--if-weight", type=float, default=None, help="Set blend weight; default tunes on validation")
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--max-depth", type=int, default=6)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("--data must point to a .csv or .parquet file")


def make_splits(frame: pd.DataFrame, y: pd.Series, timestamp: str | None, seed: int):
    indices = np.arange(len(frame))
    if timestamp:
        if pd.api.types.is_numeric_dtype(frame[timestamp]):
            times = pd.to_numeric(frame[timestamp], errors="coerce")
            if times.isna().any() or not np.isfinite(times.to_numpy(dtype=float)).all():
                raise ValueError(f"Numeric timestamp column {timestamp!r} contains missing or invalid values")
            time_kind = "numeric event-time"
        else:
            times = pd.to_datetime(frame[timestamp], errors="coerce", utc=True)
            if times.isna().any():
                raise ValueError(f"Timestamp column {timestamp!r} contains missing or invalid values")
            time_kind = "datetime event-time"
        train_cutoff, validation_cutoff = times.quantile([0.70, 0.85]).tolist()
        # Use timestamp cutoffs so transactions sharing an event time never cross partitions.
        parts = (
            indices[(times <= train_cutoff).to_numpy()],
            indices[((times > train_cutoff) & (times <= validation_cutoff)).to_numpy()],
            indices[(times > validation_cutoff).to_numpy()],
        )
        if any(len(part) == 0 for part in parts):
            raise ValueError("Timestamp splits produced an empty partition; dataset needs more distinct event times")
        method = f"chronological 70/15/15 using {time_kind}; timestamp excluded from features"
    else:
        train, remainder = train_test_split(
            indices, test_size=0.30, random_state=seed, stratify=y.to_numpy()
        )
        validation, test = train_test_split(
            remainder, test_size=0.50, random_state=seed, stratify=y.iloc[remainder].to_numpy()
        )
        parts = train, validation, test
        method = "stratified random 70/15/15; supply --timestamp for production-like temporal evaluation"
        logger.warning("No timestamp supplied: %s", method)
    for label, part in zip(("train", "validation", "test"), parts, strict=True):
        if y.iloc[part].nunique() != 2:
            raise ValueError(f"{label} split must contain both classes; revise dataset/split boundaries")
    return parts, method


def build_preprocessor(frame: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric = [name for name in feature_columns if pd.api.types.is_numeric_dtype(frame[name])]
    categorical = [name for name in feature_columns if name not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                                   ("scale", StandardScaler(with_mean=False))]), numeric)
        )
    if categorical:
        transformers.append(
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), categorical)
        )
    if not transformers:
        raise ValueError("No usable feature columns remain after exclusions")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=True)


def anomaly_percentile(iforest: IsolationForest, matrix, reference_scores: np.ndarray) -> np.ndarray:
    # decision_function is higher for more normal rows. The empirical lower-tail rank is anomalousness.
    normality = iforest.decision_function(matrix)
    return 1.0 - np.searchsorted(reference_scores, normality, side="right") / len(reference_scores)


def choose_blend_and_threshold(
    supervised: np.ndarray,
    unsupervised: np.ndarray,
    y: np.ndarray,
    fixed_weight: float | None,
) -> tuple[float, float, float]:
    weights = [fixed_weight] if fixed_weight is not None else np.linspace(0.0, 1.0, 11).tolist()
    best: tuple[float, float, float, float] | None = None
    for weight in weights:
        if weight is None or not 0 <= weight <= 1:
            raise ValueError("--if-weight must be between 0 and 1")
        scores = (1.0 - weight) * supervised + weight * unsupervised
        candidates = np.unique(np.r_[0.0, scores, 1.0])
        # Threshold ties favor higher precision, then the lower Isolation Forest weight.
        for threshold in candidates:
            predictions = scores >= threshold
            f1 = float(f1_score(y, predictions, zero_division=0))
            precision = float(precision_score(y, predictions, zero_division=0))
            rank = (f1, precision, -float(weight), float(threshold))
            if best is None or rank[:3] > (best[0], best[1], -best[2]):
                best = (f1, precision, float(weight), float(threshold))
    assert best is not None
    return best[2], best[3], best[0]


def metric_dict(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_count": int(np.sum(y)),
        "row_count": int(len(y)),
    }


def atomic_joblib(value: Any, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temp, compress=3)
    temp.replace(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if (args.output / "manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite a published model at {args.output}; use a fresh immutable output directory"
        )
    frame = load_data(args.data)
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame.columns.has_duplicates:
        raise ValueError("Dataset contains duplicate column names; rename them before training")
    if args.target not in frame:
        raise ValueError(f"Target column {args.target!r} not found")
    if args.timestamp and args.timestamp not in frame:
        raise ValueError(f"Timestamp column {args.timestamp!r} not found")
    target = frame[args.target]
    if target.isna().any() or set(pd.unique(target)) - {0, 1, False, True}:
        raise ValueError("Target must be complete and binary with values 0/1 (fraud must be 1)")
    y = target.astype("int8")
    if y.nunique() != 2:
        raise ValueError("Both target classes are required")

    excluded = set(args.exclude) | {args.target}
    if args.timestamp:
        excluded.add(args.timestamp)
    missing_exclusions = set(args.exclude) - set(frame.columns)
    if missing_exclusions:
        raise ValueError(f"Excluded columns not found: {sorted(missing_exclusions)}")
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No feature columns remain; check --exclude and --target")
    X = frame[feature_columns].copy()
    numeric_input = X.select_dtypes(include=["number"]).columns
    X[numeric_input] = X[numeric_input].replace([np.inf, -np.inf], np.nan)
    for column in X.select_dtypes(include=["object", "string"]).columns:
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
    (train_idx, validation_idx, test_idx), split_method = make_splits(frame, y, args.timestamp, args.seed)
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[validation_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx].to_numpy(), y.iloc[validation_idx].to_numpy(), y.iloc[test_idx].to_numpy()

    preprocessor = build_preprocessor(X, feature_columns)
    train_matrix = preprocessor.fit_transform(X_train)
    val_matrix = preprocessor.transform(X_val)
    test_matrix = preprocessor.transform(X_test)
    positives, negatives = int(y_train.sum()), int(len(y_train) - y_train.sum())
    if positives == 0:
        raise ValueError("Training partition has no positive examples")
    classifier = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=0.05,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=2.0, scale_pos_weight=negatives / positives,
        early_stopping_rounds=50, n_jobs=max(1, (os.cpu_count() or 2) - 1),
        random_state=args.seed,
    )
    classifier.fit(train_matrix, y_train, eval_set=[(val_matrix, y_val)], verbose=False)
    # Isolation Forest sees only legitimate training rows to learn a normality boundary.
    benign_matrix = train_matrix[y_train == 0]
    iforest = IsolationForest(
        n_estimators=300, max_samples=min(512, len(benign_matrix)), contamination="auto",
        n_jobs=max(1, (os.cpu_count() or 2) - 1), random_state=args.seed,
    ).fit(benign_matrix)
    normal_reference = np.sort(iforest.decision_function(benign_matrix).astype(np.float32))
    val_probability = classifier.predict_proba(val_matrix)[:, 1]
    val_if_score = anomaly_percentile(iforest, val_matrix, normal_reference)
    if_weight, threshold, validation_f1 = choose_blend_and_threshold(
        val_probability, val_if_score, y_val, args.if_weight
    )
    test_probability = classifier.predict_proba(test_matrix)[:, 1]
    test_if_score = anomaly_percentile(iforest, test_matrix, normal_reference)
    test_scores = (1 - if_weight) * test_probability + if_weight * test_if_score
    test_metrics = metric_dict(y_test, test_scores, threshold)
    transformed_names = preprocessor.get_feature_names_out().tolist()
    model_version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{args.seed}"
    manifest = {
        "model_version": model_version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_columns": feature_columns,
        "transformed_feature_names": transformed_names,
        "target_column": args.target,
        "timestamp_column": args.timestamp,
        "threshold": threshold,
        "if_weight": if_weight,
        "split_method": split_method,
        "metrics": {"validation_f1_at_selected_threshold": validation_f1, "test": test_metrics},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                      "xgboost": xgb.__version__, "shap": shap.__version__},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    classifier.save_model(args.output / "xgboost.json")
    atomic_joblib(preprocessor, args.output / "preprocessor.joblib")
    atomic_joblib(iforest, args.output / "isolation_forest.joblib")
    ref_temp = args.output / "normal_if_scores.npy.tmp"
    with ref_temp.open("wb") as stream:
        np.save(stream, normal_reference)
    ref_temp.replace(args.output / "normal_if_scores.npy")
    manifest_path = args.output / "manifest.json"
    manifest_temp = args.output / "manifest.json.tmp"
    manifest_temp.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    manifest_temp.replace(manifest_path)  # Write last: incomplete releases remain unready.

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"hybrid-{model_version}"):
        mlflow.log_params({"seed": args.seed, "split_method": split_method,
                           "feature_count": len(feature_columns), "encoded_feature_count": len(transformed_names),
                           "if_weight": if_weight, "threshold": threshold,
                           "xgb_best_iteration": int(classifier.best_iteration),
                           "scale_pos_weight": negatives / positives})
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items() if isinstance(value, (int, float))})
        mlflow.log_metric("validation_f1_at_selected_threshold", validation_f1)
        mlflow.log_artifacts(str(args.output), artifact_path="serving_bundle")
        mlflow.set_tags({"model_version": model_version, "target": args.target, "model_type": "xgboost_isolation_forest"})
    logger.info("Wrote serving artifacts to %s", args.output.resolve())
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    result = train(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
