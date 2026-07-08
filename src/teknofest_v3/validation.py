from sklearn.model_selection import StratifiedKFold
import numpy as np
def make_master_stratified_folds(y,n_splits=5,seed=42): return list(StratifiedKFold(n_splits,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
def make_repeated_master_folds(y,n_splits=5,seeds=(42,2026,777)): return [(s,*f) for s in seeds for f in make_master_stratified_folds(y,n_splits,s)]
def compute_threshold_stability(t):
 a=np.asarray(t);return {'count':len(a),'mean':a.mean(),'median':np.median(a),'std':np.std(a,ddof=1),'q1':np.quantile(a,.25),'q3':np.quantile(a,.75),'iqr':np.quantile(a,.75)-np.quantile(a,.25),'min':a.min(),'max':a.max()}
def compute_panel_metrics(df,panel_col='panel'):
 return df.groupby(panel_col)[['f1_macro','mcc']].mean().reset_index()
def compute_worst_panel_metrics(df):
 f=df.loc[df.f1_macro.idxmin()];m=df.loc[df.mcc.idxmin()];return {'worst_panel_by_f1_macro':f.iloc[0],'worst_panel_f1_macro':f.f1_macro,'worst_panel_by_mcc':m.iloc[0],'worst_panel_mcc':m.mcc}
def contamination_aware_validation_filter(*args,**kwargs): return {'status':'not_applied_requires_overlap_metadata','limitation':'caller must provide overlap metadata'}
def validate_no_identifier_leakage(x): return not any('variant_id' in str(c).lower() or str(c).lower().endswith('_id') for c in x)
def validate_no_target_leakage(x): return not any(any(k in str(c).lower() for k in ['label','target','outcome']) for c in x)
