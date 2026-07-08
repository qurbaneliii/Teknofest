from __future__ import annotations
import numpy as np
from sklearn.metrics import *
def threshold_predictions(p,threshold=.5): return (np.asarray(p)>=threshold).astype(int)
def binary_confusion_counts(y,p): return tuple(confusion_matrix(y,p,labels=[0,1]).ravel())
def safe_roc_auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
def safe_pr_auc(y,p): return float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan
def safe_mcc(y,p): return float(matthews_corrcoef(y,p)) if len(np.unique(p))>1 else 0.
def safe_log_loss(y,p):
 try:return float(log_loss(y,np.clip(p,1e-7,1-1e-7)))
 except ValueError:return np.nan
def compute_binary_metrics(y_true,y_prob=None,y_pred=None,threshold=.5):
 y=np.asarray(y_true); prob=np.asarray(y_prob) if y_prob is not None else None; pred=threshold_predictions(prob,threshold) if y_pred is None else np.asarray(y_pred);tn,fp,fn,tp=binary_confusion_counts(y,pred);return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'specificity':tn/(tn+fp) if tn+fp else np.nan,'f1':f1_score(y,pred,zero_division=0),'f1_macro':f1_score(y,pred,average='macro',zero_division=0),'f1_weighted':f1_score(y,pred,average='weighted',zero_division=0),'mcc':safe_mcc(y,pred),'roc_auc':safe_roc_auc(y,prob) if prob is not None else np.nan,'pr_auc':safe_pr_auc(y,prob) if prob is not None else np.nan,'brier_score':brier_score_loss(y,prob) if prob is not None else np.nan,'log_loss':safe_log_loss(y,prob) if prob is not None else np.nan,'tn':tn,'fp':fp,'fn':fn,'tp':tp,'false_positive_rate':fp/(fp+tn) if fp+tn else np.nan,'false_negative_rate':fn/(fn+tp) if fn+tp else np.nan,'positive_predictive_value':tp/(tp+fp) if tp+fp else np.nan,'negative_predictive_value':tn/(tn+fn) if tn+fn else np.nan}
def bootstrap_metric_ci(y,p,metric_fn,n_bootstrap=100,seed=42):
 rng=np.random.default_rng(seed);y=np.asarray(y);p=np.asarray(p);v=[]
 for _ in range(n_bootstrap):
  i=rng.integers(0,len(y),len(y));x=metric_fn(y[i],p[i]);
  if np.isfinite(x):v.append(x)
 return {'lower':np.quantile(v,.025) if v else np.nan,'mean':np.mean(v) if v else np.nan,'upper':np.quantile(v,.975) if v else np.nan}
