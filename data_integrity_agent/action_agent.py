import pandas as pd
import numpy as np
from data_integrity_agent import BasicDataIntegrityAgent

class DataRepairAgent:
    def __init__(self):
        self.validation_agent = BasicDataIntegrityAgent()
    
    def auto_repair_dataset(self, df):
        issues = self.validation_agent.validate_dataset(df)
        repaired_df = df.copy()
        
        if 'missing_values' in issues:
            repaired_df = self._fix_missing_values(repaired_df, issues['missing_values'])
        
        if 'outliers' in issues:
            repaired_df = self._handle_outliers(repaired_df, issues['outliers'])
        
        return repaired_df, self._generate_repair_report(issues)
    
    def _fix_missing_values(self, df, details):
        for col, count in details['counts'].items():
            if count > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        return df
    
    def _handle_outliers(self, df, details):
        for col, info in details.items():
            if info['count'] > 0 and df[col].dtype in ['int64', 'float64']:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
                df[col] = df[col].clip(lower, upper)
        return df
    
    def _generate_repair_report(self, issues):
        return {
            'repairs_applied': ['missing_values', 'outliers'],
            'original_issues': issues,
            'repair_timestamp': pd.Timestamp.now()
        }
