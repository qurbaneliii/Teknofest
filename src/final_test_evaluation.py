from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from medical_metrics import compute_medical_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "test_evaluation"
TABLES_DIR = REPORT_DIR / "tables"
FIGURES_DIR = REPORT_DIR / "figures"
PREDICTION_PATH = PROJECT_ROOT / "artifacts" / "predictions" / "final_test_predictions.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "final_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "preprocessors" / "final_preprocessor.pkl"
COLUMNS_PATH = PROJECT_ROOT / "artifacts" / "models" / "final_model_columns.txt"
THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "final_threshold.json"
DECISION_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "final_model_decision.json"


def _test_csvs(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*.csv") if "test" in path.name.lower())


def _schema_row(path: Path, train: pd.DataFrame) -> dict[str, object]:
    frame = pd.read_csv(path)
    train_columns, columns = set(train.columns), set(frame.columns)
    categorical = [column for column in train.columns if train[column].dtype == "object" and column in frame]
    category_differences = sum(
        len(set(frame[column].dropna().astype(str)) - set(train[column].dropna().astype(str)))
        for column in categorical
    )
    return {
        "file_path": str(path.resolve()),
        "dataset_name": path.stem,
        "rows": len(frame),
        "columns": len(frame.columns),
        "has_label": "Label" in columns,
        "has_variant_id": "Variant_ID" in columns,
        "schema_compatible": not (train_columns - columns),
        "missing_columns": "; ".join(sorted(train_columns - columns)),
        "extra_columns": "; ".join(sorted(columns - train_columns)),
        "missingness_rate": float(frame.isna().mean().mean()),
        "categorical_new_levels": int(category_differences),
        "use": "labeled_evaluation" if "Label" in columns else "unlabeled_inference",
    }


def discover_test_data(data_dir: str | Path) -> pd.DataFrame:
    data_path = Path(data_dir)
    train_path = data_path / "YARISMA_TRAIN_MASTER.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training schema reference is missing: {train_path}")
    train = pd.read_csv(train_path)
    return pd.DataFrame([_schema_row(path, train) for path in _test_csvs(data_path)])


def write_discovery_report(audit: pd.DataFrame, data_dir: str | Path) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(TABLES_DIR / "test_schema_audit.csv", index=False)
    lines = ["# Test Data Discovery", "", f"Searched `{Path(data_dir).resolve()}` recursively for CSV files with `test` in the filename.", ""]
    if audit.empty:
        lines.extend(
            [
                "No official test CSV was found.",
                "",
                "Expected names include `YARISMA_TEST_MASTER.csv`, `YARISMA_TEST_KANSER.csv`, `YARISMA_TEST_PAH.csv`, `YARISMA_TEST_CFTR.csv`, `TEST.csv`, or `test.csv` under the supplied data directory.",
                "No test predictions or test metrics were generated, and training files were not repurposed as test data.",
            ]
        )
    else:
        lines.extend([audit.to_markdown(index=False), ""])
    (REPORT_DIR / "test_data_discovery.md").write_text("\n".join(lines), encoding="utf-8")


def load_protected_model() -> tuple[Any, Any, list[str], dict[str, Any], dict[str, Any]]:
    required = [MODEL_PATH, PREPROCESSOR_PATH, COLUMNS_PATH, THRESHOLD_PATH, DECISION_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing protected final-model artifact(s): " + ", ".join(missing))
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    columns = [line.strip() for line in COLUMNS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if float(threshold["threshold"]) != 0.471 or float(decision["threshold"]) != 0.471:
        raise ValueError("Protected final threshold is not 0.471.")
    if decision["model_id"] != "lightgbm_conservative_regularized" or decision.get("calibration") != "none":
        raise ValueError("Protected final decision does not match the approved LightGBM configuration.")
    return model, preprocessor, columns, threshold, decision


def write_loading_audit(success: bool, message: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = "# Final Model Loading Audit\n\n" + ("Status: PASS. " if success else "Status: FAIL. ") + message + "\n"
    (REPORT_DIR / "model_loading_audit.md").write_text(text, encoding="utf-8")


def _score(frame: pd.DataFrame, model: Any, preprocessor: Any, columns: list[str], threshold: float) -> pd.DataFrame:
    engineered = preprocessor.transform(frame.copy())
    matrix = engineered.reindex(columns=columns).replace([np.inf, -np.inf], np.nan).astype(float)
    probability = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    result = pd.DataFrame({
        "row_index": frame.index.to_numpy(),
        "Variant_ID": frame.get("Variant_ID", pd.Series(frame.index.astype(str), index=frame.index)).astype(str).to_numpy(),
        "predicted_probability": probability,
        "predicted_label": (probability >= threshold).astype(int),
        "threshold_used": threshold,
        "uncertainty_flag": np.select(
            [np.abs(probability - threshold) <= 0.05, probability > threshold + 0.05],
            ["uncertain", "confident_pathogenic"],
            default="confident_benign",
        ),
    })
    if "Label" in frame:
        result["true_label"] = frame["Label"].astype(int).to_numpy()
        result["correct_prediction"] = result["true_label"].eq(result["predicted_label"])
        result["error_type"] = np.select(
            [(result.true_label == 0) & (result.predicted_label == 1), (result.true_label == 1) & (result.predicted_label == 0)],
            ["FP", "FN"], default=np.where(result.correct_prediction, "correct", "unknown"),
        )
    return result


def _save_unlabeled_outputs(predictions: pd.DataFrame, train: pd.DataFrame) -> None:
    summary = predictions.groupby("dataset_name", dropna=False).agg(
        rows=("predicted_label", "size"), pathogenic_predictions=("predicted_label", "sum"),
        mean_probability=("predicted_probability", "mean"), uncertain_predictions=("uncertainty_flag", lambda x: int((x == "uncertain").sum())),
    ).reset_index()
    summary["benign_predictions"] = summary.rows - summary.pathogenic_predictions
    summary["uncertain_percentage"] = summary.uncertain_predictions / summary.rows
    summary.to_csv(TABLES_DIR / "unlabeled_test_prediction_summary.csv", index=False)
    margin_rows = []
    for margin in (0.03, 0.05, 0.10):
        uncertain = (predictions.predicted_probability.sub(0.471).abs() <= margin)
        margin_rows.append({"margin": margin, "uncertain_predictions": int(uncertain.sum()), "uncertain_percentage": float(uncertain.mean())})
    pd.DataFrame(margin_rows).to_csv(TABLES_DIR / "unlabeled_test_uncertainty_summary.csv", index=False)
    test_missing = predictions.shape[0] * 0.0
    pd.DataFrame([{"training_mean_missingness": float(train.isna().mean().mean()), "test_prediction_rows": len(predictions), "test_raw_missingness_available": False, "test_missingness_proxy": test_missing}]).to_csv(TABLES_DIR / "test_distribution_shift_summary.csv", index=False)


def _save_labeled_metrics(predictions: pd.DataFrame) -> None:
    rows = []
    for name, group in [("overall", predictions), *list(predictions.groupby("dataset_name"))]:
        metrics = compute_medical_metrics(group.true_label, group.predicted_probability, float(group.threshold_used.iloc[0]))
        rows.append({"evaluation_split": name, **metrics})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES_DIR / "final_test_metrics.csv", index=False)
    metrics.to_csv(TABLES_DIR / "final_test_panel_metrics.csv", index=False)
    overall = metrics.iloc[0]
    pd.DataFrame([{key: overall[key] for key in ("tn", "fp", "fn", "tp")}]).to_csv(TABLES_DIR / "final_test_confusion_matrix.csv", index=False)
    confidence = []
    for name, group in predictions.groupby("uncertainty_flag"):
        if len(group) and group.true_label.nunique() == 2:
            confidence.append({"confidence_group": name, **compute_medical_metrics(group.true_label, group.predicted_probability, float(group.threshold_used.iloc[0]))})
    pd.DataFrame(confidence).to_csv(TABLES_DIR / "final_test_confidence_metrics.csv", index=False)


def _plots(predictions: pd.DataFrame, labeled: bool) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.hist(predictions.predicted_probability, bins=20); plt.xlabel("Predicted probability"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "probability_histogram.png", dpi=160); plt.close()
    predictions.predicted_label.value_counts().sort_index().plot(kind="bar"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "predicted_class_distribution.png", dpi=160); plt.close()
    predictions.uncertainty_flag.value_counts().plot(kind="bar"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "uncertainty_distribution.png", dpi=160); plt.close()
    if not labeled or predictions.true_label.nunique() < 2:
        return
    ConfusionMatrixDisplay.from_predictions(predictions.true_label, predictions.predicted_label); plt.tight_layout(); plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160); plt.close()
    fpr, tpr, _ = roc_curve(predictions.true_label, predictions.predicted_probability); plt.plot(fpr, tpr); plt.tight_layout(); plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=160); plt.close()
    precision, recall, _ = precision_recall_curve(predictions.true_label, predictions.predicted_probability); plt.plot(recall, precision); plt.tight_layout(); plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=160); plt.close()
    frac, mean = calibration_curve(predictions.true_label, predictions.predicted_probability, n_bins=10); plt.plot(mean, frac, marker="o"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=160); plt.close()


def run_final_test_evaluation(data_dir: str | Path) -> dict[str, object]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True); TABLES_DIR.mkdir(parents=True, exist_ok=True)
    audit = discover_test_data(data_dir)
    write_discovery_report(audit, data_dir)
    if audit.empty:
        return {"status": "no_test_files", "predictions": 0, "message": "No official test CSV found; test inference and metrics were not run."}
    try:
        model, preprocessor, columns, threshold_data, decision = load_protected_model()
        write_loading_audit(True, f"Loaded model, preprocessor, {len(columns)} feature columns, threshold 0.471, and final decision metadata.")
    except Exception as exc:
        write_loading_audit(False, f"{type(exc).__name__}: {exc}")
        raise
    started = time.perf_counter(); tracemalloc.start()
    frames = []
    for path in _test_csvs(Path(data_dir)):
        scored = _score(pd.read_csv(path), model, preprocessor, columns, float(threshold_data["threshold"]))
        scored.insert(0, "dataset_name", path.stem); scored["model_id"] = decision["model_id"]; frames.append(scored)
    current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); elapsed = time.perf_counter() - started
    predictions = pd.concat(frames, ignore_index=True); predictions.to_csv(PREDICTION_PATH, index=False)
    labeled = "true_label" in predictions
    if labeled: _save_labeled_metrics(predictions)
    else: _save_unlabeled_outputs(predictions, pd.read_csv(Path(data_dir) / "YARISMA_TRAIN_MASTER.csv"))
    _plots(predictions, labeled)
    pd.DataFrame([{"rows": len(predictions), "seconds": elapsed, "rows_per_second": len(predictions) / elapsed if elapsed else np.nan, "peak_tracemalloc_bytes": peak}]).to_csv(TABLES_DIR / "runtime_test.csv", index=False)
    return {"status": "labeled" if labeled else "unlabeled", "predictions": len(predictions), "prediction_path": str(PREDICTION_PATH)}
