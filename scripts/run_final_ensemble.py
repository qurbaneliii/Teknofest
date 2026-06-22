from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from final_ensembling import run_final_ensembles


def main() -> None:
    result = run_final_ensembles()
    comparison = result["comparison"]
    eligible = comparison[comparison["eligible_to_replace"]]
    print(f"Ensemble comparison: {PROJECT_ROOT / 'reports' / 'tables' / 'final_ensemble_comparison.csv'}")
    print(f"Ensembles evaluated: {len(comparison)}")
    print(f"Ensembles meeting the two-metric gate: {len(eligible)}")


if __name__ == "__main__":
    main()
