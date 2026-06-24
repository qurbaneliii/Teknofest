import numpy as np
from teknofest_v3.metrics import *
def test_binary_confusion_counts(): assert binary_confusion_counts([0,0,1,1],[0,1,0,1])==(1,1,1,1)
def test_compute_binary_metrics_basic(): assert compute_binary_metrics([0,1],[.1,.9])['tp']==1
def test_safe_auc_single_class_does_not_crash(): assert np.isnan(safe_roc_auc([1,1],[.2,.8]))
def test_mcc_safe_on_degenerate_case(): assert safe_mcc([0,1],[1,1])==0
def test_threshold_predictions(): assert threshold_predictions([.4,.5],.5).tolist()==[0,1]
def test_bootstrap_metric_ci_runs(): assert 'mean' in bootstrap_metric_ci([0,1],[.1,.9],safe_roc_auc,10)
