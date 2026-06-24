from __future__ import annotations
import pandas as pd
class RobustGenomicsSelector:
 weights={'master_f1_macro':.15,'master_mcc':.15,'master_pr_auc':.10,'master_roc_auc':.07,'panel_f1_macro':.13,'panel_mcc':.13,'kanser_f1_macro':.10,'kanser_mcc':.10,'worst_panel_f1_macro':.04,'threshold_stability_score':.03}
 def compute_robust_genomics_score(self,m):
  raw=sum(w*float(m.get(k,0) or 0) for k,w in self.weights.items());pen=(.1 if m.get('leakage_suspected') else 0)+(.1 if m.get('reproducible') is False else 0);return {'raw_score':raw,'penalties':pen,'final_score':raw-pen,'component_scores':self.weights,'notes':[]}
 def check_replacement_criteria(self,c,b):
  keys=['master_f1_macro','master_mcc','master_pr_auc','panel_f1_macro','panel_mcc','kanser_f1_macro','kanser_mcc','worst_panel_f1_macro'];imp=[k for k in keys if c.get(k,-1)>b.get(k,-1)];r=[]
  if self.compute_robust_genomics_score(c)['final_score']<=self.compute_robust_genomics_score(b)['final_score']:r+=['score_not_improved']
  if len(imp)<2:r+=['fewer_than_two_primary_improvements']
  if c.get('kanser_f1_macro',0)<b.get('kanser_f1_macro',0)-.01:r+=['kanser_f1_worsened']
  if c.get('kanser_mcc',0)<b.get('kanser_mcc',0)-.015:r+=['kanser_mcc_worsened']
  if c.get('panel_f1_macro',0)<b.get('panel_f1_macro',0)-.01 or c.get('panel_mcc',0)<b.get('panel_mcc',0)-.015:r+=['panel_worsened']
  if c.get('leakage_suspected') or c.get('reproducible') is False:r+=['safety_failure']
  return {'improved_components':imp,'replacement_allowed':not r,'rejection_reasons':r}
 def compare_to_baseline(self,c,b):return {**self.check_replacement_criteria(c,b),'candidate_score':self.compute_robust_genomics_score(c),'baseline_score':self.compute_robust_genomics_score(b)}
 def make_selection_board(self,rows):return pd.DataFrame(rows)
