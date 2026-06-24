import pandas as pd
from teknofest_v3.validation import *
def test_stratified_folds_preserve_labels(): assert len(make_master_stratified_folds([0,1]*10))==5
def test_threshold_stability_computation(): assert compute_threshold_stability([.4,.5])['count']==2
def test_worst_panel_metric_computation(): assert compute_worst_panel_metrics(pd.DataFrame({'panel':['a','b'],'f1_macro':[.8,.7],'mcc':[.6,.5]}))['worst_panel_by_mcc']=='b'
def test_identifier_leakage_check(): assert not validate_no_identifier_leakage(['Variant_ID'])
def test_target_leakage_check(): assert not validate_no_target_leakage(['Label'])
def test_contamination_filter_returns_status(): assert 'status' in contamination_aware_validation_filter()
