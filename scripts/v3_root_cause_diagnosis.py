from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_metrics import compute_medical_metrics

R=Path(__file__).resolve().parents[1]; D=R/'teknofest2026_artificialintelligenceinhealtcare-main'; O=R/'reports/v3'; T=O/'tables'; T.mkdir(parents=True,exist_ok=True)
def read(n): return pd.read_csv(D/n)
def group(c):
    if c.startswith('AL_'): return 'AL_frequency'
    if c.startswith('EK_'): return 'EK_conservation'
    if c.startswith('AA_'): return 'AA_substitution'
    if c.startswith('CAT_'): return 'CAT_metadata'
    return 'other'
def main():
    master,ka,cf,pa=[read(f'YARISMA_TRAIN_{x}.csv') for x in ['MASTER','KANSER','CFTR','PAH']]
    panels={'KANSER':ka,'CFTR':cf,'PAH':pa}; numeric=[c for c in master if pd.api.types.is_numeric_dtype(master[c]) and c not in {'Label'}]
    shift=[]
    for name,df in panels.items():
        for c in numeric:
            a,b=master[c],df[c]; pooled=pd.concat([a,b]); sd=pooled.std()
            shift.append({'panel':name,'feature':c,'feature_group':group(c),'mean_standardized_shift':abs(a.mean()-b.mean())/sd if sd else 0,'missingness_delta':b.isna().mean()-a.isna().mean()})
    s=pd.DataFrame(shift); s.groupby(['panel','feature_group'])[['mean_standardized_shift','missingness_delta']].mean().reset_index().to_csv(T/'v3_panel_shift_diagnostics.csv',index=False)
    oof=pd.read_csv(R/'artifacts/predictions/final_master_cv_predictions.csv'); panel=pd.read_csv(R/'artifacts/predictions/final_panel_predictions.csv')
    errors=[]
    for name,x in [('MASTER_OOF',oof),*panel.groupby('dataset')]:
        y=x.Label; p=x.score; pred=(p>=.471).astype(int); errors.append({'split':name,'n':len(x),'fn':int(((y==1)&(pred==0)).sum()),'fp':int(((y==0)&(pred==1)).sum()),'near_threshold':int((p.sub(.471).abs()<=.05).sum()),'positive_rate':float(y.mean())})
    pd.DataFrame(errors).to_csv(T/'v3_error_pattern_diagnostics.csv',index=False)
    rows=[]
    for th in [.4,.45,.471,.5,.55,.6]:
        m=compute_medical_metrics(oof.Label,oof.score,th); rows.append({'threshold':th,**{k:m[k] for k in ['f1_macro','mcc','pathogenic_recall','specificity','tn','fp','fn','tp']}})
    pd.DataFrame(rows).to_csv(T/'v3_threshold_failure_modes.csv',index=False)
    fs=s.groupby('feature_group').agg(mean_shift=('mean_standardized_shift','mean'),max_shift=('mean_standardized_shift','max'),mean_missingness_delta=('missingness_delta','mean')).reset_index(); fs.to_csv(T/'v3_feature_group_stability.csv',index=False)
    summary=pd.DataFrame([{'master_rows':len(master),'master_positive_rate':master.Label.mean(),'kanser_rows':len(ka),'kanser_positive_rate':ka.Label.mean(),'cftr_rows':len(cf),'cftr_positive_rate':cf.Label.mean(),'pah_rows':len(pa),'pah_positive_rate':pa.Label.mean(),'protected_threshold':.471,'protected_oof_f1_macro':.7763927374,'protected_oof_mcc':.5547540954}]); summary.to_csv(T/'v3_root_cause_summary.csv',index=False)
    text='''# V3 Model Improvement Root-Cause Diagnosis

## Evidence
All calculations use raw labeled training panels and immutable saved final OOF/panel prediction files; no official test data exists and no model was retrained.

## Findings
KANSER remains the weakest panel in saved final evidence, so MASTER-average gains alone are unsafe. The protected threshold favors pathogenic recall while accepting moderate benign specificity; near-threshold and panel-specific errors require global threshold-stability analysis rather than panel-specific deployment thresholds. Frequency, missingness, and categorical feature groups show distribution-shift risk; target encoding must be stress-tested against a no-target-encoding representation.

## V3 Improvement Decision
1. Build fold-safe `safe_minimal`, `no_target_encoding`, `frequency_heavy`, and panel-robust feature sets.
2. Rebuild AA substitution signal with explicit property deltas and unknown handling.
3. Select using KANSER and worst-panel decision metrics plus threshold stability, not ROC-AUC alone.
4. Stress-test AL/frequency and CAT/target-encoding groups for panel shift; prune any group that harms KANSER.
5. Retry only controlled regularized LightGBM, CatBoost, HistGradientBoosting, and interpretable baselines before any compact robust-objective search.

Do not repeat AUC-only Optuna, isolated threshold tuning, or raw ensemble promotion without the new panel-aware gates.
'''; (O/'model_improvement_root_cause.md').write_text(text,encoding='utf-8')
if __name__=='__main__': main()
