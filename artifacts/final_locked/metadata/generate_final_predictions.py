"""Generate label-free predictions using only the locked final baseline bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from final_inference import _single_bundle_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions without using input labels.")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--locked-artifact-dir", default=ROOT / "artifacts" / "final_locked", type=Path)
    args = parser.parse_args()
    locked = args.locked_artifact_dir
    decision = json.loads((locked / "metadata" / "final_model_decision.json").read_text(encoding="utf-8"))
    if decision["model_id"] != "lightgbm_conservative_regularized" or float(decision["threshold"]) != 0.471:
        raise ValueError("Locked decision is not the approved protected baseline.")
    raw = pd.read_csv(args.input_csv)
    metadata_id = raw.get("Variant_ID", pd.Series(raw.index.astype(str), index=raw.index)).astype(str)
    raw = raw.drop(columns=["Label"], errors="ignore")
    bundle = joblib.load(locked / "model" / "final_model.pkl")
    probability, schema_warnings = _single_bundle_probabilities(raw, bundle)
    threshold = float(decision["threshold"])
    output = pd.DataFrame({"Variant_ID": metadata_id, "probability": probability, "predicted_label": (probability >= threshold).astype(int), "threshold": threshold, "model_id": decision["model_id"]})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Predictions written: {args.output_csv} ({len(output)} rows). Input labels were not used. Schema warnings: {len(schema_warnings)}")


if __name__ == "__main__":
    main()
