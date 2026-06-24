from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from teknofest_v3.data import load_available_datasets
from teknofest_v3.preprocessing import V3FeatureBuilder
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--feature-set',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--sample-only',action='store_true');p.add_argument('--max-rows',type=int,default=None);p.add_argument('--write-artifacts',choices=['yes','no'],default='yes');a=p.parse_args()
 if not a.data_dir.exists(): raise FileNotFoundError(f'Data directory not found: {a.data_dir.resolve()}')
 ds=load_available_datasets(a.data_dir)
 if 'MASTER' not in ds: raise FileNotFoundError('MASTER CSV is required for V3 feature fitting.')
 x,y,_,_=ds['MASTER']; x=x.head(a.max_rows) if a.sample_only and a.max_rows else x; y=y.loc[x.index] if y is not None else None;b=V3FeatureBuilder(a.feature_set); z=b.fit_transform(x,y); a.output_dir.mkdir(parents=True,exist_ok=True); z.to_csv(a.output_dir/f'{a.feature_set}_MASTER_features.csv',index=False)
 for n,(px,_,_,_) in ds.items():
  if n!='MASTER': b.transform(px.head(a.max_rows) if a.sample_only and a.max_rows else px).to_csv(a.output_dir/f'{a.feature_set}_{n}_features.csv',index=False)
 s=Path('artifacts/v3/schemas');s.mkdir(parents=True,exist_ok=True);(s/f'{a.feature_set}_feature_schema.json').write_text(json.dumps(b.get_feature_groups(),indent=2))
 t=Path('reports/v3/tables');t.mkdir(parents=True,exist_ok=True);pd.DataFrame([{'feature_set':a.feature_set,'source':'sample_real_data' if a.sample_only else 'real_data','rows':len(z),'features':z.shape[1]}]).to_csv(t/f'{a.feature_set}_feature_summary.csv',index=False);pd.DataFrame([{'feature':k,'feature_group':v,'source':'sample_real_data'} for k,v in b.get_feature_groups().items()]).to_csv(t/'v3_feature_schema.csv',index=False);pd.DataFrame([{'feature_group':v,'count':list(b.get_feature_groups().values()).count(v),'source':'sample_real_data'} for v in set(b.get_feature_groups().values())]).to_csv(t/'v3_feature_group_summary.csv',index=False);pd.DataFrame([{'feature_set':a.feature_set,'features':z.shape[1],'source':'sample_real_data'}]).to_csv(t/'v3_feature_set_summary.csv',index=False);Path(f'reports/v3/{a.feature_set}_feature_build_log.md').write_text(f'# V3 feature build\n\nRows: {len(z)}; features: {z.shape[1]}. No model trained.\n');print(z.shape)
if __name__=='__main__':main()
