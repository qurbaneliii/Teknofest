from __future__ import annotations
import pandas as pd
def numeric_summary(df,prefix,name):
 c=[x for x in df if x.startswith(prefix)]; z=df.reindex(columns=c).apply(pd.to_numeric,errors='coerce'); return pd.DataFrame({f'{name}_missing_count':z.isna().sum(axis=1),f'{name}_mean':z.mean(axis=1),f'{name}_max':z.max(axis=1),f'{name}_min':z.min(axis=1),f'{name}_std':z.std(axis=1).fillna(0)},index=df.index)
