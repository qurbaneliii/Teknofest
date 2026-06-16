from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def nb(cells: list) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return notebook


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


COMMON_SETUP = """from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import matplotlib.pyplot as plt
"""


notebooks = {
    "01_data_understanding.ipynb": nb(
        [
            md("# 01 Data Understanding\n\nSchema checks, label balance, panel overlap, and missingness diagnostics."),
            code(COMMON_SETUP),
            code(
                """from data_loading import discover_data_dir, write_data_diagnostics
from teknofest.data_prep import load_datasets, prepare_data

data_dir = discover_data_dir(PROJECT_ROOT)
datasets = load_datasets(data_dir)
prepared = prepare_data(data_dir)
tables_dir = PROJECT_ROOT / "reports" / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)
write_data_diagnostics(data_dir, tables_dir)

summary = pd.read_csv(tables_dir / "dataset_summary.csv")
summary"""
            ),
            code(
                """ax = prepared.master["Label"].value_counts().sort_index().plot(kind="bar", color=["#4c6f82", "#9b5f4f"])
ax.set_xticklabels(["Benign", "Pathogenic"], rotation=0)
ax.set_title("MASTER label distribution")
ax.set_ylabel("Variants")
plt.tight_layout()"""
            ),
            code("""pd.read_csv(tables_dir / "overlap_summary.csv")"""),
            code("""pd.read_csv(tables_dir / "missingness_summary.csv").head(20)"""),
        ]
    ),
    "02_feature_engineering_and_validation.ipynb": nb(
        [
            md("# 02 Feature Engineering And Validation\n\nLeakage-safe feature generation and contamination-aware folds."),
            code(COMMON_SETUP),
            code(
                """from data_loading import discover_data_dir
from teknofest.data_prep import prepare_data
from teknofest.features import FeatureEngineer, detect_binary_al_cols
from teknofest.validation import contamination_aware_folds, fold_summary

prepared = prepare_data(discover_data_dir(PROJECT_ROOT))
flag_cols = detect_binary_al_cols(prepared.master, prepared.al_cols)
engineer = FeatureEngineer(prepared.al_cols, prepared.al_raw, flag_cols)
engineered = engineer.fit_transform(prepared.master)
feature_cols = [c for c in engineered.columns if c not in {"Variant_ID", "Label"}]
len(feature_cols), feature_cols[:20]"""
            ),
            code(
                """folds = contamination_aware_folds(prepared.master["Label"], prepared.master_shared_mask)
diagnostics = fold_summary(prepared.master, folds)
diagnostics"""
            ),
            code(
                """ax = diagnostics.set_index("fold")[["train_size", "val_size"]].plot(kind="bar", color=["#52796f", "#b56576"])
ax.set_title("Fold sizes")
ax.set_ylabel("Rows")
plt.tight_layout()"""
            ),
        ]
    ),
    "03_model_training_and_evaluation.ipynb": nb(
        [
            md("# 03 Model Training And Evaluation\n\nBaselines, main-model outputs, thresholds, and validation figures."),
            code(COMMON_SETUP),
            code(
                """import subprocess

subprocess.run([sys.executable, str(PROJECT_ROOT / "run_pipeline.py"), "--mode", "smoke"], check=True)
pd.read_csv(PROJECT_ROOT / "reports" / "tables" / "baseline_results.csv")"""
            ),
            code("""pd.read_csv(PROJECT_ROOT / "reports" / "tables" / "all_evaluation_metrics.csv").head(20)"""),
            code("""pd.read_csv(PROJECT_ROOT / "reports" / "tables" / "main_model_cv_results.csv").head(20)"""),
            code("""pd.read_csv(PROJECT_ROOT / "reports" / "tables" / "threshold_results.csv")"""),
            code(
                """from IPython.display import Image, display
for name in [
    "correlation_matrix_top_features.png",
    "confusion_matrix_master.png",
    "roc_curve_master.png",
    "pr_curve_master.png",
    "threshold_optimization.png",
    "model_comparison_metrics.png",
    "feature_importance_top30.png",
]:
    path = PROJECT_ROOT / "reports" / "figures" / name
    if path.exists():
        display(Image(filename=str(path)))"""
            ),
        ]
    ),
    "04_explainability_and_report_outputs.ipynb": nb(
        [
            md("# 04 Explainability And Report Outputs\n\nFeature importance, ACMG mapping, panel errors, and final report references."),
            code(COMMON_SETUP),
            code(
                """tables = PROJECT_ROOT / "reports" / "tables"
figures = PROJECT_ROOT / "reports" / "figures"
pd.read_csv(tables / "feature_importance.csv").head(25)"""
            ),
            code("""pd.read_csv(tables / "acmg_feature_mapping.csv")"""),
            code("""pd.read_csv(tables / "panel_generalization_results.csv")"""),
            code("""pd.read_csv(tables / "error_analysis.csv").head(20)"""),
            code(
                """from IPython.display import Image, Markdown, display
if (figures / "feature_importance.png").exists():
    display(Image(filename=str(figures / "feature_importance.png")))
display(Markdown((PROJECT_ROOT / "reports" / "final_model_report_summary.md").read_text(encoding="utf-8")))"""
            ),
        ]
    ),
}


for filename, notebook in notebooks.items():
    nbf.write(notebook, NOTEBOOK_DIR / filename)

print(f"Wrote {len(notebooks)} notebooks to {NOTEBOOK_DIR}")
