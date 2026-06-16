from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix, f1_score

from teknofest.validation import best_f1_macro_threshold, youden_j_threshold


def _savefig(path: Path, dpi: int = 170) -> None:
    try:
        plt.savefig(path, dpi=dpi)
    except OSError:
        if path.exists():
            path.unlink()
        plt.savefig(path, dpi=dpi)


def threshold_results(y_true, y_score) -> pd.DataFrame:
    f1_thr, _ = best_f1_macro_threshold(y_true, y_score)
    youden_thr, _ = youden_j_threshold(y_true, y_score)
    rows = []
    for name, thr in [("default_0.5", 0.5), ("f1_macro_opt", f1_thr), ("youden_j", youden_thr)]:
        pred = (y_score >= thr).astype(int)
        rows.append({"threshold_name": name, "threshold": float(thr), "f1_macro": float(f1_score(y_true, pred, average="macro"))})
    return pd.DataFrame(rows)


def save_evaluation_figures(y_true, y_score, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Benign", "Pathogenic"])
    plt.yticks([0, 1], ["Benign", "Pathogenic"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title("MASTER OOF confusion matrix")
    plt.tight_layout()
    _savefig(out / "confusion_matrix_master.png")
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    _savefig(out / "roc_curve_master.png")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    _savefig(out / "pr_curve_master.png")
    plt.close()
