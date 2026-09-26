from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap
import xgboost as xgb


class ModelBundle:
    """Loaded inference components. Artifact directories must come from a trusted trainer."""

    def __init__(self, model_dir: Path):
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Model manifest missing: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.model_version = str(self.manifest["model_version"])
        self.feature_columns: list[str] = self.manifest["feature_columns"]
        self.transformed_feature_names: list[str] = self.manifest["transformed_feature_names"]
        self.threshold = float(self.manifest["threshold"])
        self.if_weight = float(self.manifest["if_weight"])
        self.preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        self.iforest = joblib.load(model_dir / "isolation_forest.joblib")
        self.normal_if_scores = np.load(model_dir / "normal_if_scores.npy", mmap_mode="r")
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(model_dir / "xgboost.json")
        self.explainer = shap.TreeExplainer(self.xgb_model)

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(row) - set(self.feature_columns))
        if unknown:
            raise ValueError(f"Unknown feature(s): {', '.join(unknown)}")
        import pandas as pd

        frame = pd.DataFrame([{column: row.get(column) for column in self.feature_columns}])
        transformed = self.preprocessor.transform(frame)
        fraud_probability = float(self.xgb_model.predict_proba(transformed)[0, 1])
        normality_score = float(self.iforest.decision_function(transformed)[0])
        if_percentile = float(np.searchsorted(self.normal_if_scores, normality_score, side="right"))
        if_score = 1.0 - if_percentile / max(1, len(self.normal_if_scores))
        hybrid_score = (1.0 - self.if_weight) * fraud_probability + self.if_weight * if_score

        shap_values = self.explainer.shap_values(transformed)
        values = np.asarray(shap_values)
        if values.ndim == 3:  # SHAP version/model combinations may include an output axis.
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[0]
        else:
            raise RuntimeError(f"Unexpected SHAP output shape: {values.shape}")
        contributions = [
            {"feature": name, "input_feature": raw_name,
             "value": _json_scalar(row.get(raw_name)), "shap_value": float(value)}
            for name, raw_name, value in zip(
                self.transformed_feature_names,
                _raw_feature_names(self.transformed_feature_names, self.feature_columns),
                values,
                strict=True,
            )
        ]
        contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
        return {
            "model_version": self.model_version,
            "fraud_probability": fraud_probability,
            "isolation_forest_score": if_score,
            "hybrid_score": hybrid_score,
            "threshold": self.threshold,
            "is_anomaly": hybrid_score >= self.threshold,
            "explanation": {
                "method": "SHAP TreeExplainer; XGBoost raw-margin contributions",
                "base_value": float(np.asarray(self.explainer.expected_value).reshape(-1)[-1]),
                "features": contributions,
            },
        }


def _raw_feature_names(transformed: list[str], raw: list[str]) -> list[str]:
    """Map one-hot encoded names to their input field, preserving preprocessor output order."""
    mapped = []
    for name in transformed:
        # ColumnTransformer prefixes each field with its transformer name, e.g. numeric__amount.
        base_name = name.split("__", 1)[-1]
        matches = [column for column in raw if base_name == column or base_name.startswith(f"{column}_")]
        mapped.append(max(matches, key=len) if matches else base_name)
    return mapped


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
