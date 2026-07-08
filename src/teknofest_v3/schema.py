from __future__ import annotations
import pandas as pd
def infer_schema(df):
 rows=[]
 for c in df:
  ident=c.lower()=='variant_id' or ('id' in c.lower() and c!='Label'); label=c=='Label'; group='AL_frequency' if c.startswith('AL_') else 'EK_conservation' if c.startswith('EK_') else 'AA_substitution' if c.startswith('AA_') else 'CAT_metadata' if c.startswith('CAT_') else 'other'; rows.append({'original_column':c,'inferred_feature_group':group,'dtype':str(df[c].dtype),'missingness_rate':float(df[c].isna().mean()),'unique_count':int(df[c].nunique(dropna=True)),'is_identifier_like':ident,'is_label':label,'is_allowed_model_input':not ident and not label,'reason_if_excluded':'identifier_or_label' if ident or label else ''})
 return pd.DataFrame(rows)
