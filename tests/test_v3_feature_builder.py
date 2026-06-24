import pandas as pd
from teknofest_v3.preprocessing import V3FeatureBuilder
def test_v3_excludes_identifiers_and_labels_and_is_numeric():
 x=pd.DataFrame({'Variant_ID':['a','b'],'Label':[0,1],'AL_1':[0.,1.],'EK_1':[1.,2.],'AA_1':['A','?'],'AA_2':['V','?']});z=V3FeatureBuilder('v3_aa_heavy').fit_transform(x,x.Label);assert 'Variant_ID' not in z and 'Label' not in z and z.select_dtypes('object').empty and z.columns.is_unique
