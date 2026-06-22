from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from final_selection_board import build_final_selection_board, save_final_selection_board


def main() -> None:
    board, selected = build_final_selection_board()
    decision_path = save_final_selection_board(board, selected)
    print(f"Final selection board: {PROJECT_ROOT / 'reports' / 'tables' / 'final_selection_board.csv'}")
    print(f"Selected model: {selected['candidate_id']}")
    print(f"Selected threshold: {float(selected['threshold']):.6f}")
    print(f"Decision artifact: {decision_path}")


if __name__ == "__main__":
    main()
