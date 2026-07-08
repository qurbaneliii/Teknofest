from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from teknofest_v3.features import V3FeatureBuilder


frame = pd.DataFrame({"Variant_ID": ["a", "b", "c", "d"], "Label": [0, 1, 0, 1], "AL_1": [0.0, 0.1, None, 0.2], "EK_1": [1.0, 2.0, 3.0, None], "CAT_1": ["x", "x", "y", "z"], "AA_1": ["A", "R", "X", "C"], "AA_2": ["V", "K", "A", "M"]})
for name in ("v3_safe_minimal", "v3_no_target_encoding", "v3_frequency_heavy", "v3_panel_robust", "v3_aa_heavy", "v3_bio_full"):
    builder = V3FeatureBuilder(name).fit(frame.iloc[:3])
    out = builder.transform(frame.iloc[3:])
    assert len(out) == 1 and "Variant_ID" not in out and "Label" not in out
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out)
print("V3 feature safety validation passed")
