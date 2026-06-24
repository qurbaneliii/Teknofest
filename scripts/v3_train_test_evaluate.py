from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from teknofest_v3.data import load_available_datasets
from teknofest_v3.preprocessing import V3FeatureBuilder
from teknofest_v3.metrics import compute_binary_metrics
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--test-size',type=float,default=.2);p.add_argument('--random-state',type=int,default=42);p.add_argument('--feature-sets',default='v3_safe_minimal,v3_no_target_encoding');p.add_argument('--models',default='logistic_regression,extratrees');a=p.parse_args(); d=load_available_datasets(a.data_dir);x,y,_,_=d['MASTER'];tr,te=train_test_split(range(len(x)),test_size=a.test_size,stratify=y,random_state=a.random_state);a.output_dir.mkdir(parents=True,exist_ok=True);pred=a.output_dir/'../..'/'artifacts_placeholder'; rows=[]
 for fs in a.feature_sets.split(','):
  b=V3FeatureBuilder(fs);xt=b.fit_transform(x.iloc[tr],y.iloc[tr]);xe=b.transform(x.iloc[te]);
  for name,model in [('logistic_regression',LogisticRegression(max_iter=500)),('extratrees',ExtraTreesClassifier(n_estimators=100,random_state=42,n_jobs=4))]:
   t=time.time();model.fit(xt,y.iloc[tr]);prob=model.predict_proba(xe)[:,1];m=compute_binary_metrics(y.iloc[te],prob,threshold=.5);rows.append({'model_id':name,'feature_set':fs,'n_features':xt.shape[1],'train_seconds':time.time()-t,**m});pd.DataFrame({'row_index':list(te),'Label':y.iloc[te].values,'probability':prob}).to_csv(a.output_dir/f'{name}_{fs}_local_test_predictions.csv',index=False)
 pd.DataFrame(rows).to_csv(a.output_dir/'model_train_test_metrics.csv',index=False);pd.DataFrame([{'train_n':len(tr),'local_test_n':len(te),'train_positive_rate':y.iloc[tr].mean(),'local_test_positive_rate':y.iloc[te].mean()}]).to_csv(a.output_dir/'train_test_split_summary.csv',index=False);print(pd.DataFrame(rows)[['model_id','feature_set','f1_macro','mcc']].to_string(index=False))
if __name__=='__main__':main()
