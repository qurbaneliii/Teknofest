from teknofest_v3.selection import RobustGenomicsSelector
def base(): return {'master_f1_macro':.7,'master_mcc':.5,'master_pr_auc':.8,'master_roc_auc':.8,'panel_f1_macro':.7,'panel_mcc':.5,'kanser_f1_macro':.7,'kanser_mcc':.5,'worst_panel_f1_macro':.7,'threshold_stability_score':.9,'reproducible':True}
def test_robust_score_computes(): assert RobustGenomicsSelector().compute_robust_genomics_score(base())['final_score']>0
def test_roc_only_improvement_rejected(): assert not RobustGenomicsSelector().check_replacement_criteria({**base(),'master_roc_auc':.9},base())['replacement_allowed']
def test_kansER_worsening_rejected(): assert not RobustGenomicsSelector().check_replacement_criteria({**base(),'master_mcc':.6,'master_f1_macro':.8,'kanser_f1_macro':.6},base())['replacement_allowed']
def test_leakage_flag_rejected(): assert not RobustGenomicsSelector().check_replacement_criteria({**base(),'leakage_suspected':True},base())['replacement_allowed']
def test_panel_worsening_rejected(): assert not RobustGenomicsSelector().check_replacement_criteria({**base(),'master_mcc':.6,'master_f1_macro':.8,'panel_mcc':.4},base())['replacement_allowed']
def test_unreproducible_candidate_rejected(): assert not RobustGenomicsSelector().check_replacement_criteria({**base(),'reproducible':False},base())['replacement_allowed']
def test_two_metric_improvement_can_pass(): assert RobustGenomicsSelector().check_replacement_criteria({**base(),'master_mcc':.6,'master_f1_macro':.8},base())['replacement_allowed']
