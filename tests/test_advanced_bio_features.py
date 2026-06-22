import pandas as pd

from advanced_bio_features import AdvancedBioFeatureEngineer


def test_advanced_bio_features_add_substitution_evidence_and_aggregates():
    frame = pd.DataFrame(
        {
            "AA_1": ["A", "C"],
            "AA_2": ["V", "R"],
            "AL_1": [0.0, 0.02],
            "AL_2": [0.01, 0.0],
            "EK_1": [0.2, 0.8],
            "EK_2": [-1.0, 2.0],
            "EK_7": [0.5, 6.0],
            "EK_9": [2.0, 8.0],
        }
    )
    transformed = AdvancedBioFeatureEngineer().transform(frame)

    required = {
        "adv_blosum62",
        "adv_grantham_distance",
        "adv_hydrophobicity_delta",
        "adv_computational_pathogenic_count",
        "adv_evidence_entropy_score",
        "adv_al_mean",
        "adv_ek_top3_mean",
        "adv_radical_x_conservation",
    }
    assert required.issubset(transformed.columns)
    assert transformed["adv_grantham_distance"].notna().all()
