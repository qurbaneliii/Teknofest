from __future__ import annotations
import numpy as np
import pandas as pd
AA={'A':(89.1,1.8,0),'V':(117.1,4.2,0),'G':(75.1,-0.4,0),'P':(115.1,-1.6,0),'C':(121.2,2.5,0),'D':(133.1,-3.5,-1),'E':(147.1,-3.5,-1),'K':(146.2,-3.9,1),'R':(174.2,-4.5,1),'H':(155.2,-3.2,1),'F':(165.2,2.8,0),'Y':(181.2,-1.3,0),'W':(204.2,-0.9,0),'I':(131.2,4.5,0),'L':(131.2,3.8,0),'M':(149.2,1.9,0),'N':(132.1,-3.5,0),'Q':(146.2,-3.5,0),'S':(105.1,-0.8,0),'T':(119.1,-0.7,0)}
def aa_features(frame: pd.DataFrame) -> pd.DataFrame:
 r=frame.get('AA_1',pd.Series(index=frame.index,dtype=object)).astype(str).str.upper().str[0]; a=frame.get('AA_2',pd.Series(index=frame.index,dtype=object)).astype(str).str.upper().str[0]; ok=r.isin(AA)&a.isin(AA); out=pd.DataFrame(index=frame.index); out['aa_parse_success']=ok.astype(float); out['aa_unknown_ref']=(~r.isin(AA)).astype(float); out['aa_unknown_alt']=(~a.isin(AA)).astype(float)
 for n,i in [('molecular_weight_delta',0),('hydrophobicity_delta',1),('charge_change',2)]: out[n]=[(AA[x][i]-AA[y][i]) if x in AA and y in AA else 0. for x,y in zip(a,r)]
 out['proline_involved']=((r=='P')|(a=='P')).astype(float); out['glycine_involved']=((r=='G')|(a=='G')).astype(float); out['cysteine_involved']=((r=='C')|(a=='C')).astype(float); out['amino_acid_property_distance']=out[['molecular_weight_delta','hydrophobicity_delta','charge_change']].abs().sum(axis=1); return out
