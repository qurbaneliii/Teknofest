from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from .bio_features import aa_features
from .features import numeric_summary
from .schema import infer_schema
from .utils import load_config
class V3FeatureBuilder:
 def __init__(self,feature_set='v3_safe_minimal',config=None): self.feature_set=feature_set; self.config=config or load_config(feature_set); self.medians={};self.lower={};self.upper={};self.names=[];self.groups={}
 def fit(self,X_train,y_train=None,metadata_train=None):
  s=infer_schema(X_train); self.raw=s.loc[s.is_allowed_model_input&s.dtype.str.contains('int|float'),'original_column'].tolist(); z=X_train.reindex(columns=self.raw).apply(pd.to_numeric,errors='coerce');q=self.config.get('clipping_quantiles',[0.01,0.99]);self.medians=z.median().to_dict();self.lower=z.quantile(q[0]).to_dict();self.upper=z.quantile(q[1]).to_dict(); self.names=list(self._engineer(X_train).columns);self.groups={c:('AL_frequency' if c.startswith('AL_') or c.startswith('al_') else 'EK_conservation' if c.startswith('EK_') or c.startswith('ek_') else 'AA_substitution' if c.startswith('aa_') else 'missingness' if 'missing' in c else 'baseline_raw') for c in self.names};return self
 def _engineer(self,X):
  z=X.reindex(columns=self.raw).apply(pd.to_numeric,errors='coerce').clip(self.lower,self.upper,axis=1).fillna(self.medians); out=z.copy(); out=pd.concat([out,numeric_summary(X,'AL_','al'),numeric_summary(X,'EK_','ek')],axis=1);out['row_missing_count']=X.isna().sum(axis=1);out['row_missing_rate']=X.isna().mean(axis=1)
  if self.config.get('allow_aa_features'): out=pd.concat([out,aa_features(X)],axis=1)
  return out.replace([np.inf,-np.inf],np.nan).fillna(0.)
 def transform(self,X,metadata=None): return self._engineer(X).reindex(columns=self.names,fill_value=0.).astype(float)
 def fit_transform(self,X_train,y_train=None,metadata_train=None): return self.fit(X_train,y_train,metadata_train).transform(X_train)
 def get_feature_names(self): return self.names
 def get_feature_schema(self): return self.groups
 def get_feature_groups(self): return self.groups
 def save(self,path): joblib.dump(self,path)
 @staticmethod
 def load(path): return joblib.load(path)
