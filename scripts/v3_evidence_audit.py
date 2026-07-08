from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v3"
SKIP = {".git"}

def rel(p: Path) -> str: return p.relative_to(ROOT).as_posix()
def status(path: str) -> tuple[str,str,str]:
    if path.startswith(".codex_backups/") or "__pycache__" in path or path.endswith(".pyc"): return "temporary_or_cache","safe_delete_candidate","yes"
    if path.startswith("teknofest2026_") or path.startswith("tests/") or path in {"README.md","requirements.txt"}: return "raw_data" if path.endswith(".csv") else "test","protected_final","no"
    if path.startswith("artifacts/predictions/"): return "prediction_artifact","protected_final" if "/final_" in path else "legacy_useful","no" if "/final_" in path else "maybe"
    if any(x in path for x in ["artifacts/models/final_","artifacts/preprocessors/final_","artifacts/predictions/final_","artifacts/metrics/final_"]): return "model_artifact","protected_final","no"
    if path.startswith("src/teknofest/") or path in {"run_pipeline.py","scripts/run_model_performance_improvement.py"}: return "script","active_core","no"
    if path.startswith("reports/master_prompt/") or "optuna" in path.lower() or "model_zoo" in path.lower() or "ensemble" in path.lower(): return "report","rejected_experiment","yes"
    ext=Path(path).suffix.lower(); cat={".py":"script",".ipynb":"notebook",".csv":"metric_table",".json":"metric_table",".md":"report",".png":"figure",".jpg":"figure",".jpeg":"figure",".svg":"figure",".pkl":"model_artifact",".joblib":"model_artifact",".sqlite":"model_artifact",".pdf":"official_document",".docx":"official_document"}.get(ext,"unknown")
    return cat,"legacy_useful","maybe"

def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True); files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.parts)]
    inv=[]
    for p in files:
        path=rel(p); cat,st,arc=status(path); inv.append({"file_path":path,"file_type":p.suffix.lower(),"file_size":p.stat().st_size,"modified_time":p.stat().st_mtime,"category":cat,"evidence_role":"repository evidence","active_or_legacy_status":st,"can_be_archived_later":arc,"protected_do_not_touch":"yes" if st=="protected_final" else "no","reason":"heuristic audit; source-specific maps govern selection","related_pipeline_stage":"final" if "final" in path else "experiment"})
    inventory=pd.DataFrame(inv); inventory.to_csv(OUT/'evidence_inventory.csv',index=False)
    metric=[]
    for p in [ROOT/'reports/tables/final_medical_metric_comparison.csv',ROOT/'reports/tables/final_model_selection_table.csv',ROOT/'reports/tables/model_zoo_metrics.csv',ROOT/'reports/tables/final_ensemble_comparison.csv',ROOT/'reports/tables/optuna_before_after_comparison.csv']:
        if not p.exists(): continue
        try: df=pd.read_csv(p)
        except Exception: continue
        for _,r in df.iterrows():
            metric.append({"file_path":rel(p),"metric_context":r.get('evaluation_split',r.get('model_id','table_row')),"model_name":r.get('model_name',r.get('model_id',r.get('ensemble_id','unknown'))),"threshold":r.get('threshold'),'split':r.get('evaluation_split','unknown'),"dataset_or_panel":r.get('evaluation_split','unknown'),**{k:r.get(k) for k in ['roc_auc','pr_auc','accuracy','balanced_accuracy','precision','recall','specificity','f1','f1_macro','mcc','tn','fp','fn','tp']},"source_priority":1 if p.name=='final_medical_metric_comparison.csv' else 2,"verified_from_predictions":"yes" if p.name=='final_medical_metric_comparison.csv' else "unknown","selected_as_final":r.get('selected_as_final','unknown'),"notes":"saved audit preferred"})
    pd.DataFrame(metric).to_csv(OUT/'metric_source_map.csv',index=False)
    figs=[]
    for p in files:
        if p.suffix.lower() in {'.png','.jpg','.jpeg','.svg'}:
            n=p.name.lower(); kind=next((x for x in ['roc','pr','confusion','threshold','calibration','feature_importance','shap','panel'] if x in n),'unknown'); figs.append({"file_path":rel(p),"figure_type":kind,"related_stage":"final" if 'final' in n else 'experiment',"source_data_file":"unknown","readable_quality":"unknown","suitable_for_report":"maybe","suitable_for_presentation":"maybe","active_or_legacy_status":status(rel(p))[1],"notes":"verify visually before reuse"})
    pd.DataFrame(figs).to_csv(OUT/'figure_source_map.csv',index=False)
    code=[]
    for p in files:
        if p.suffix=='.py':
            t=p.read_text(encoding='utf-8',errors='ignore'); code.append({"file_path":rel(p),"purpose":(next((x.strip('# ').strip() for x in t.splitlines() if x.strip().startswith('#')),"module"))[:120],"imports_internal_modules":"yes" if re.search(r'from (teknofest|src|final_)|import (teknofest|final_)',t) else "no","writes_outputs":"yes" if re.search(r'to_csv|write_text|joblib.dump',t) else "no","reads_inputs":"yes" if re.search(r'read_csv|joblib.load|open\(',t) else "no","active_in_pipeline":"yes" if rel(p) in {'run_pipeline.py','scripts/run_model_performance_improvement.py'} or rel(p).startswith('src/teknofest/') else "unknown","called_by_readme":"unknown","called_by_run_pipeline":"yes" if p.name in (ROOT/'run_pipeline.py').read_text(errors='ignore') else "unknown","related_phase":"final","safe_to_archive_later":"no" if status(rel(p))[1]=='active_core' else "maybe","notes":"static audit"})
    pd.DataFrame(code).to_csv(OUT/'code_dependency_map.csv',index=False)
    notebooks=[]
    for p in files:
        if p.suffix=='.ipynb':
            try: n=nbformat.read(p,as_version=4); outputs=any(c.cell_type=='code' and c.get('outputs') for c in n.cells)
            except Exception: outputs=False
            notebooks.append({"file_path":rel(p),"purpose":"notebook","execution_status":"unknown","outputs_present":"yes" if outputs else "no","duplicates_script_logic":"unknown","active_or_legacy_status":status(rel(p))[1],"safe_to_archive_later":status(rel(p))[2],"notes":"inspect before archival"})
    pd.DataFrame(notebooks).to_csv(OUT/'notebook_inventory.csv',index=False)
    arts=inventory[inventory.category.isin(['model_artifact','prediction_artifact'])].copy(); arts.rename(columns={'category':'artifact_type','protected_do_not_touch':'protected_do_not_touch'},inplace=True); arts['model_id']='lightgbm_conservative_regularized' ; arts['used_by_final_pipeline']=arts.file_path.str.contains('final_').map({True:'yes',False:'unknown'}); arts['related_metrics_file']='reports/tables/final_medical_metric_comparison.csv'; arts['notes']='inventory'; arts[['file_path','artifact_type','model_id','used_by_final_pipeline','protected_do_not_touch','related_metrics_file','notes']].to_csv(OUT/'artifact_inventory.csv',index=False)
    decision=json.loads((ROOT/'artifacts/metrics/final_model_decision.json').read_text()); lines=["# V3 Evidence Map","",f"Protected final: `{decision['model_id']}`, threshold `{decision['threshold']}`, calibration `{decision['calibration']}`.","","Saved prediction audit is the priority source. Official test CSV/labels are absent; no official test metrics exist.","","# Inconsistencies","","- **High**: `reports/master_prompt/` AUC-only Optuna and candidate outputs are not final selection evidence. Action: archive later.","- **Medium**: archived FP/FN case row counts differ from selected OOF confusion counts, as documented in `final_model_selection_decision.md`. Action: use only qualitative error analysis.","- **Medium**: pipeline-generated legacy reports may describe pre-selection LightGBM states. Action: prioritize saved prediction audits and final decision JSON."]
    (OUT/'01_evidence_map.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); (OUT/'inconsistency_report.md').write_text('\n'.join(lines[6:])+'\n',encoding='utf-8')
    summary=f"# Phase 1 Audit Summary\n\nFiles scanned: {len(files)}. Python: {len(code)}. Notebooks: {len(notebooks)}. Figures: {len(figs)}. Metric-source rows: {len(metric)}.\n\nConfirmed final model: `{decision['model_id']}` at `{decision['threshold']}`. Next: Phase 2 cleanup planning only after human review of the inventory.\n"; (OUT/'phase1_audit_summary.md').write_text(summary,encoding='utf-8')
if __name__=='__main__': main()
