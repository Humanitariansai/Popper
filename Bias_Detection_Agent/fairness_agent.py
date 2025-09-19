from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import numpy as np
import pandas as pd

def _to_series(x, name: str = "x") -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reset_index(drop=True)
    return pd.Series(x, name=name)

def _safe_rate(numer: int, denom: int) -> float:
    if denom == 0:
        return np.nan
    return numer / denom

def _selection_rate(y_pred: pd.Series, positive_label=1) -> float:
    return _safe_rate(int((y_pred == positive_label).sum()), len(y_pred))

def _tpr(y_true: pd.Series, y_pred: pd.Series, positive_label=1) -> float:
    mask_pos = (y_true == positive_label)
    if mask_pos.sum() == 0:
        return np.nan
    return _safe_rate(int(((y_pred == positive_label) & mask_pos).sum()), int(mask_pos.sum()))

def _accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return _safe_rate(int((y_true == y_pred).sum()), len(y_true))

@dataclass
class FairnessThresholds:
    dp_gap_max: float = 0.1
    eo_gap_max: float = 0.1
    di_min_ratio: float = 0.8

class AlgorithmicFairnessAgent:
    def __init__(self, favorable_label: int | str = 1,
                 protected_attributes: Optional[Dict[str, Iterable]] = None,
                 privileged_groups: Optional[Dict[str, Iterable]] = None):
        self.favorable_label = favorable_label
        self.protected_df: Optional[pd.DataFrame] = None
        self.privileged_groups = privileged_groups or {}
        if protected_attributes is not None:
            self.set_protected_attributes(protected_attributes)

    def set_protected_attributes(self, protected_attributes: Dict[str, Iterable]):
        self.protected_df = pd.DataFrame({k: list(v) for k, v in protected_attributes.items()}).reset_index(drop=True)

    def demographic_parity(self, y_pred, attribute: str) -> pd.DataFrame:
        y_pred = _to_series(y_pred, name="y_pred")
        groups = self._get_attribute_series(attribute)
        rows = []
        for g, idx in groups.groupby(groups).groups.items():
            sel_rate = _selection_rate(y_pred.iloc[list(idx)], self.favorable_label)
            rows.append({"attribute": attribute, "group": g, "selection_rate": sel_rate, "count": len(idx)})
        df = pd.DataFrame(rows)
        max_rate, min_rate = df["selection_rate"].max(), df["selection_rate"].min()
        df["gap_vs_max"] = max_rate - df["selection_rate"]
        df["gap_vs_min"] = df["selection_rate"] - min_rate
        ref = self._get_reference_group(attribute, df)
        ref_rate = df.loc[df["group"] == ref, "selection_rate"].values[0]
        df["diff_vs_ref"] = df["selection_rate"] - ref_rate
        df["ratio_vs_ref"] = df["selection_rate"] / ref_rate if ref_rate not in (0, np.nan) else np.nan
        return df.sort_values("selection_rate", ascending=False).reset_index(drop=True)

    def equal_opportunity(self, y_true, y_pred, attribute: str) -> pd.DataFrame:
        y_true = _to_series(y_true, name="y_true")
        y_pred = _to_series(y_pred, name="y_pred")
        groups = self._get_attribute_series(attribute)
        rows = []
        for g, idx in groups.groupby(groups).groups.items():
            tpr = _tpr(y_true.iloc[list(idx)], y_pred.iloc[list(idx)], self.favorable_label)
            rows.append({"attribute": attribute, "group": g, "tpr": tpr, "count": len(idx)})
        df = pd.DataFrame(rows)
        max_tpr, min_tpr = df["tpr"].max(), df["tpr"].min()
        df["gap_vs_max"] = max_tpr - df["tpr"]
        df["gap_vs_min"] = df["tpr"] - min_tpr
        ref = self._get_reference_group(attribute, df)
        ref_tpr = df.loc[df["group"] == ref, "tpr"].values[0]
        df["diff_vs_ref"] = df["tpr"] - ref_tpr
        df["ratio_vs_ref"] = df["tpr"] / ref_tpr if ref_tpr not in (0, np.nan) else np.nan
        return df.sort_values("tpr", ascending=False).reset_index(drop=True)

    def disparate_impact(self, y_pred, attribute: str) -> pd.DataFrame:
        y_pred = _to_series(y_pred, name="y_pred")
        groups = self._get_attribute_series(attribute)
        ref = self._get_reference_group(attribute)
        rates = {g: _selection_rate(y_pred.iloc[list(idx)], self.favorable_label) for g, idx in groups.groupby(groups).groups.items()}
        ref_rate = rates.get(ref, np.nan)
        other = {g: r for g, r in rates.items() if g != ref}
        di_ratios = {g: (r / ref_rate if ref_rate not in (0, np.nan) else np.nan) for g, r in other.items()}
        worst_group, worst_ratio = (None, np.nan)
        if di_ratios:
            worst_group, worst_ratio = min(di_ratios.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=np.inf)))
        return pd.DataFrame({
            "attribute": [attribute],
            "reference_group": [ref],
            "reference_rate": [ref_rate],
            "worst_group": [worst_group],
            "worst_group_rate": [rates.get(worst_group, np.nan)],
            "di_ratio": [worst_ratio],
        })

    def intersectional_bias(self, y_true, y_pred, attributes: Optional[List[str]] = None) -> pd.DataFrame:
        if self.protected_df is None:
            raise ValueError("Protected attributes not set. Call set_protected_attributes().")
        y_true = _to_series(y_true, name="y_true")
        y_pred = _to_series(y_pred, name="y_pred")
        attrs = attributes or list(self.protected_df.columns)
        df = self.protected_df[attrs].copy()
        combo = df.astype(str).agg("|".join, axis=1)
        rows = []
        for g, idx in combo.groupby(combo).groups.items():
            sel = _selection_rate(y_pred.iloc[list(idx)], self.favorable_label)
            tpr = _tpr(y_true.iloc[list(idx)], y_pred.iloc[list(idx)], self.favorable_label)
            rows.append({"intersection_group": g, "selection_rate": sel, "tpr": tpr, "count": len(idx)})
        result = pd.DataFrame(rows).sort_values(["selection_rate", "tpr"], ascending=False).reset_index(drop=True)
        if not result.empty:
            for col in ["selection_rate", "tpr"]:
                best, worst = result[col].max(), result[col].min()
                result[f"{col}_gap_vs_best"] = best - result[col]
                result[f"{col}_gap_vs_worst"] = result[col] - worst
        return result

    def evaluate_fairness_criteria(self, y_true, y_pred, thresholds: FairnessThresholds, attribute: str) -> pd.DataFrame:
        y_true = _to_series(y_true, name="y_true")
        y_pred = _to_series(y_pred, name="y_pred")
        dp_df = self.demographic_parity(y_pred, attribute)
        eo_df = self.equal_opportunity(y_true, y_pred, attribute)
        di_df = self.disparate_impact(y_pred, attribute)
        dp_gap = dp_df["selection_rate"].max() - dp_df["selection_rate"].min()
        eo_gap = eo_df["tpr"].max() - eo_df["tpr"].min()
        di_ratio = float(di_df["di_ratio"].iloc[0])
        acc = _accuracy(y_true, y_pred)
        return pd.DataFrame({
            "attribute": [attribute],
            "dp_gap": [dp_gap],
            "eo_gap": [eo_gap],
            "di_ratio": [di_ratio],
            "accuracy": [acc],
            "pass_dp": [dp_gap <= thresholds.dp_gap_max],
            "pass_eo": [eo_gap <= thresholds.eo_gap_max],
            "pass_di": [di_ratio >= thresholds.di_min_ratio],
        })

    def assess_tradeoffs(self, scores, y_true, attribute: str, thresholds: FairnessThresholds, threshold_grid=None) -> pd.DataFrame:
        scores = _to_series(scores, name="score")
        y_true = _to_series(y_true, name="y_true")
        import numpy as np
        if threshold_grid is None:
            threshold_grid = np.linspace(0.05, 0.95, 19)
        rows = []
        for t in threshold_grid:
            y_pred = (scores >= t).astype(int)
            ev = self.evaluate_fairness_criteria(y_true, y_pred, thresholds, attribute)
            rows.append({"threshold": t, **ev.iloc[0].to_dict()})
        return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    def _get_attribute_series(self, attribute: str) -> pd.Series:
        if self.protected_df is None or attribute not in self.protected_df.columns:
            raise ValueError(f"Attribute '{attribute}' not found. Set protected attributes first.")
        return self.protected_df[attribute].reset_index(drop=True)

    def _get_reference_group(self, attribute: str, df_metrics: Optional[pd.DataFrame] = None):
        if attribute in self.privileged_groups and self.privileged_groups[attribute] is not None:
            return self.privileged_groups[attribute]
        if df_metrics is not None and "count" in df_metrics.columns:
            return df_metrics.sort_values("count", ascending=False)["group"].iloc[0]
        ser = self._get_attribute_series(attribute)
        return ser.mode().iloc[0]
