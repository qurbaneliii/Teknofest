from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from teknofest_v3.selection import robust_genomics_score


master = {"f1_macro": 0.75, "mcc": 0.55, "pr_auc": 0.85, "roc_auc": 0.82}
panels = [{"evaluation_split": "KANSER", "f1_macro": 0.70, "mcc": 0.48}, {"evaluation_split": "CFTR", "f1_macro": 0.90, "mcc": 0.80}, {"evaluation_split": "PAH", "f1_macro": 0.73, "mcc": 0.50}]
score = robust_genomics_score(master, panels)
assert 0.0 < score < 1.0
print(f"V3 selection-objective validation passed (score={score:.4f})")
