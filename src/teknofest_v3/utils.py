from __future__ import annotations
from pathlib import Path
import yaml
def load_config(feature_set):
 p=Path(__file__).resolve().parents[2]/'configs'/'v3'/f'{feature_set}.yaml'; return yaml.safe_load(p.read_text())
