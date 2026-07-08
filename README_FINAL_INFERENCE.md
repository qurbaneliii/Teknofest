# Audited Final Inference

The audited candidate is trained only on MASTER variants whose `Variant_ID` is
not present in KANSER, PAH, or CFTR. It does not need a `Label` column for
inference; if one is present it is ignored.

```powershell
python src/final_inference.py --input path/to/official_test.csv --output outputs/submission.csv
```

The default decision is `artifacts/metrics/audited_final_model_decision.json`.
To force a different artifact, pass `--decision path/to/decision.json`.

The normal output includes Variant_ID, probability, binary prediction,
threshold, model identifier, and uncertainty flag. The official organizer
template was not available in this repository. Only after its required columns
are confirmed should you use the compact form:

```powershell
python src/final_inference.py --input path/to/official_test.csv --output outputs/submission.csv --basic-submission
```

This compact form writes `Variant_ID,predicted_label`; rename columns only if
the organizer's published template requires it.
