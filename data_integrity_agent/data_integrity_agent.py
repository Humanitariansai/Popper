import pandas as pd
import numpy as np
from typing import Dict, List, Any

class BasicDataIntegrityAgent:
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main validation method"""
        results = {
            'missing_values': self._check_missing_values(df),
            'outliers': self._check_outliers(df),
            'distributions': self._analyze_distributions(df),
            'summary': self._generate_summary(df)
        }
        return results
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values and their impact"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'total_missing': missing_counts.sum(),
            'total_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100
        }
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify outliers using IQR method"""
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            outliers[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100
            }
        
        return outliers
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distributions"""
        distributions = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            distributions[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew()
            }
        
        return distributions
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall summary"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }
