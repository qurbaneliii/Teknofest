from __future__ import annotations
from pathlib import Path
import pandas as pd
def infer_dataset_group(path):
 n=Path(path).stem.upper(); return next((x for x in ['MASTER','KANSER','CFTR','PAH'] if x in n),'UNKNOWN')
def infer_label_column(df): return 'Label' if 'Label' in df.columns else None
def split_features_labels_metadata(df,label_col=None):
 label_col=label_col or infer_label_column(df); meta=df[[c for c in ['Variant_ID'] if c in df]].copy(); y=df[label_col].copy() if label_col else None; return df.drop(columns=[c for c in ['Variant_ID',label_col] if c],errors='ignore'),y,meta
def find_competition_files(data_dir): return sorted(Path(data_dir).rglob('*.csv'))
def load_dataset(path,dataset_name=None):
 df=pd.read_csv(path); x,y,m=split_features_labels_metadata(df); return x,y,m,dataset_name or infer_dataset_group(path)
def load_available_datasets(data_dir): return {infer_dataset_group(p):load_dataset(p) for p in find_competition_files(data_dir) if infer_dataset_group(p)!='UNKNOWN'}
