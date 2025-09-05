from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np, pandas as pd, os, time, json
from collections import Counter

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
                             mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from validators import BasicDataValidationAgent

# --- Utility functions and AgentResult class ---
def _as_series(y):
    return y if isinstance(y, pd.Series) else pd.Series(y)

def _task_type(y: pd.Series) -> str:
    y = _as_series(y)
    if pd.api.types.is_numeric_dtype(y):
        uniq = pd.unique(y.dropna())
        if len(uniq) == 2 and set(map(float, uniq)).issubset({0.0, 1.0}): return "binary"
        if len(uniq) <= 50 and all(float(v).is_integer() for v in uniq):  return "multiclass"
        return "regression"
    return "multiclass"

def _min_class_count(y):
    yS = _as_series(y).dropna()
    uniq, counts = np.unique(yS, return_counts=True)
    return int(counts.min()) if len(counts) else 0

def _cv_for_task(y, cv_splits=5, random_state=42):
    task = _task_type(y)
    if task in ("binary","multiclass"):
        mcc = _min_class_count(y)
        n = max(2, min(cv_splits, mcc)) if mcc >= 2 else 2
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=random_state) if mcc>=2 else KFold(n_splits=n, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

def _finding(sev, msg, meta=None):
    return {"severity": sev, "message": msg, "meta": meta or {}}

@dataclass
class AgentResult:
    agent: str
    score: float
    findings: List[Dict[str,Any]]
    metrics: Dict[str,Any]
    def to_dict(self): return asdict(self)

# --- Data Integrity Agent ---
def data_integrity_agent(df: pd.DataFrame, label_col: Optional[str]=None) -> AgentResult:
    """Delegate dataset-level checks to BasicDataValidationAgent and wrap results into AgentResult."""
    validator = BasicDataValidationAgent()
    metrics = validator.validate_dataset(df)

    # Convert the validator's findings into the AgentResult findings schema
    findings = []
    # missingness warning if any column > 30% missing
    miss_top = metrics.get("missing_values", {}).get("percentages", {})
    if miss_top:
        # find max missing
        max_col = max(miss_top.items(), key=lambda kv: kv[1])
        if max_col[1] > 30.0:
            findings.append(_finding("warning", f"High missingness in '{max_col[0]}' ({max_col[1]:.1f}%)."))

    # constant columns info
    consts = [c for c, stats in (metrics.get("distributions") or {}).items() if stats.get("std") in (0.0, None)]
    if consts:
        findings.append(_finding("info", f"{len(consts)} constant columns detected.", {"columns": consts}))

    # duplicates detection using simple df check if available
    try:
        dups = int(df.duplicated().sum())
        if dups:
            findings.append(_finding("warning", f"Found {dups} duplicate rows."))
            metrics["duplicate_rows"] = dups
    except Exception:
        pass

    # label checks (best-effort)
    if label_col and label_col in df.columns:
        y = df[label_col]
        nun = int(pd.Series(y).nunique(dropna=True))
        metrics["label_unique_values"] = nun
        if nun == 1:
            findings.append(_finding("error", f"Label '{label_col}' has only one unique value."))
        if nun > 200 and not pd.api.types.is_numeric_dtype(y):
            findings.append(_finding("warning", f"Label '{label_col}' seems high-cardinality ({nun})."))

    # derive a simple score from findings
    score = 100.0
    if any(f["severity"]=="error" for f in findings): score -= 40
    if any(f["severity"]=="warning" for f in findings): score -= 20
    return AgentResult("data_integrity", score, findings, metrics)

def sampling_agent(df: pd.DataFrame, label_col: Optional[str]=None, train_frac=0.75, random_state=7) -> AgentResult:
    findings, metrics = [], {}
    if label_col and label_col in df.columns:
        y = df[label_col]
        if not pd.api.types.is_numeric_dtype(y): y = pd.Categorical(y).codes
        from sklearn.model_selection import train_test_split
        strat = y if _min_class_count(y) >= 2 else None
        _, _, ytr, yte = train_test_split(df.drop(columns=[label_col]), y, test_size=1-train_frac, random_state=random_state, stratify=strat)
        from collections import Counter
        base, trc, tec = Counter(y), Counter(ytr), Counter(yte)

        def _ratio(cnter):
            tot = sum(int(v) for v in cnter.values()) or 1
            return {int(k): (int(v)/tot) for k, v in cnter.items()}

        r_all, r_te = _ratio(base), _ratio(tec)
        drifts = {int(k): abs(r_all.get(int(k),0.0) - r_te.get(int(k),0.0)) for k in set(r_all)|set(r_te)}
        mx = max(drifts.values()) if drifts else 0.0

        metrics.update({
            "class_counts_all": {int(k): int(v) for k, v in base.items()},
            "class_counts_train": {int(k): int(v) for k, v in trc.items()},
            "class_counts_test": {int(k): int(v) for k, v in tec.items()},
            "test_class_ratio_drift": float(mx)
        })
        if mx > 0.10: findings.append(_finding("warning", f"Class ratio drift > 0.10 between full data and test ({mx:.2f})."))
    else:
        findings.append(_finding("info", "No label provided; sampling checks limited."))
    score = 100.0 - (15.0 if any(f["severity"]=="warning" for f in findings) else 0.0)
    return AgentResult("sampling", score, findings, metrics)


def consistency_agent(df: pd.DataFrame) -> AgentResult:
    num_cols = df.select_dtypes(include=[np.number]).columns
    negatives = {}
    for c in num_cols:
        if any(tok in c.lower() for tok in ["amount","income","balance","score","term","tenure","credit"]):
            neg = float((df[c] < 0).mean())
            if neg > 0: negatives[c] = neg
    findings = []
    if negatives:
        findings.append(_finding("warning", "Found negative values in non-negative-like fields.", {"columns": negatives}))
    return AgentResult("consistency", 100.0 - (10.0 if negatives else 0.0), findings, {"negatives_suspect": negatives})

# ... other agents may be added below ...
