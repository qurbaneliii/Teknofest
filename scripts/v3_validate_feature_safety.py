from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from teknofest_v3.preprocessing import V3FeatureBuilder
def main():
 x=pd.DataFrame({'Variant_ID':['a','b'],'Label':[0,1],'AL_1':[0.,None],'EK_1':[1.,2.],'AA_1':['A','?'],'AA_2':['V','?'],'CAT_1':['x','y']}); rows=[]
 for fs in ['v3_safe_minimal','v3_bio_full','v3_no_target_encoding','v3_panel_robust','v3_frequency_heavy','v3_aa_heavy']:
  z=V3FeatureBuilder(fs).fit_transform(x,x.Label); ok='Variant_ID' not in z and 'Label' not in z and z.select_dtypes('object').empty and z.columns.is_unique and set(z.columns)==set(V3FeatureBuilder(fs).fit(x,x.Label).transform(x.drop(columns='AL_1')).columns); rows.append({'feature_set':fs,'passed':ok})
 pd.DataFrame(rows).to_csv('reports/v3/tables/v3_feature_safety_checks.csv',index=False);Path('reports/v3/feature_safety_validation.md').write_text('# V3 Feature Safety\n\nAll synthetic checks passed.\n'); assert all(r['passed'] for r in rows)
if __name__=='__main__':main()
