# Final Inference Readiness

`generate_final_predictions.py` loads only locked model/preprocessing artifacts, retains Variant_ID as output metadata, ignores Label if present, aligns locked feature columns, and writes probability plus thresholded labels. No official test CSV exists locally, so no prediction file was generated. Metrics are unavailable without explicitly supplied labeled local data.
