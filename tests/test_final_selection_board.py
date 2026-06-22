import pandas as pd

from final_selection_board import REFERENCE_ID, select_final_candidate


def _candidate(candidate_id: str, kind: str, utility: float, panel_utility: float, panel_f1: float, panel_mcc: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_kind": kind,
        "mcc": 0.55 if candidate_id == REFERENCE_ID else 0.57,
        "f1_macro": 0.77 if candidate_id == REFERENCE_ID else 0.76,
        "pr_auc": 0.90 if candidate_id == REFERENCE_ID else 0.91,
        "medical_utility_score": utility,
        "clinical_safety_score": 0.74,
        "panel_medical_utility_score": panel_utility,
        "panel_f1_macro": panel_f1,
        "panel_mcc": panel_mcc,
        "fold_medical_utility_score_std": 0.02,
    }


def test_final_selection_preserves_reference_when_challenger_lowers_utility():
    board = pd.DataFrame(
        [
            _candidate(REFERENCE_ID, "current_final_model", 0.775, 0.775, 0.771, 0.582),
            _candidate("ridge_stacking", "ensemble", 0.774, 0.780, 0.780, 0.590),
        ]
    )

    selected_board, selected = select_final_candidate(board)

    assert selected["candidate_id"] == REFERENCE_ID
    rejected = selected_board.loc[selected_board["candidate_id"].eq("ridge_stacking")].iloc[0]
    assert not rejected["eligible_to_replace"]
    assert "lower OOF MedicalUtilityScore" in rejected["rejection_reason"]
