from __future__ import annotations
import argparse,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from teknofest_v3.data import load_available_datasets
from teknofest_v3.validation import make_master_stratified_folds
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',default='reports/v3/tables');p.add_argument('--n-splits',type=int,default=5);p.add_argument('--seeds',default='42,2026,777');p.add_argument('--sample-only',action='store_true');a=p.parse_args();d=load_available_datasets(a.data_dir);x,y,_,_=d['MASTER'];rows=[{'fold':i,'train_n':len(tr),'validation_n':len(va),'validation_positive_rate':float(y.iloc[va].mean())} for i,(tr,va) in enumerate(make_master_stratified_folds(y,a.n_splits))];o=Path(a.output_dir);o.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(o/'v3_validation_plan.csv',index=False);pd.DataFrame(rows).to_csv(o/'v3_fold_balance_summary.csv',index=False);pd.DataFrame([{'panel':n,'rows':len(v[0]),'positive_rate':float(v[1].mean())} for n,v in d.items() if v[1] is not None]).to_csv(o/'v3_panel_label_summary.csv',index=False);Path('reports/v3/validation_design_v3.md').write_text('# V3 Validation Design\n\nMASTER CV, panel-unique KANSER protection, worst-panel metrics, threshold stability, and leakage checks are required before any V3 selection. No V3 model trained.\n')
if __name__=='__main__':main()
