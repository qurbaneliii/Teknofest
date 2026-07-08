"""Single, reproducible entrypoint for the audited TEKNOFEST final pipeline."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Regenerate the audited candidate, then package final competition outputs."""
    for script in ("scripts/run_teknofest_readiness_audit.py", "scripts/package_audited_final_outputs.py"):
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    print("Final audited pipeline completed. See outputs/final/TEKNOFEST_MODEL_READINESS_REPORT.md")


if __name__ == "__main__":
    main()
