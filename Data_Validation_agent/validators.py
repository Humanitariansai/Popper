import pandas as pd
import numpy as np

class BasicDataValidationAgent:
    """Lightweight dataset validator that returns JSON-serializable results."""
    def __init__(self):
        pass

    def _check_missing_values(self, df: pd.DataFrame) -> dict:
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        return {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'total_missing': int(missing_counts.sum()),
            'total_percentage': float((missing_counts.sum() / (len(df) * len(df.columns))) * 100) if len(df) and len(df.columns) else 0.0
        }

    def _check_outliers(self, df: pd.DataFrame) -> dict:
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = int(len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]))
            outliers[col] = {
                'count': outlier_count,
                'percentage': float((outlier_count / len(df)) * 100) if len(df) else 0.0
            }
        return outliers

    def _analyze_distributions(self, df: pd.DataFrame) -> dict:
        distributions = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            distributions[col] = {
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'skewness': float(df[col].skew()) if not pd.isna(df[col].skew()) else None
            }
        return distributions

    def _generate_summary(self, df: pd.DataFrame) -> dict:
        return {
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
            'categorical_columns': int(len(df.select_dtypes(include=['object', 'category']).columns))
        }

    def _convert_to_serializable(self, obj):
        # recursive conversion for numpy/pandas types
        if isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_serializable(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj

    def validate_dataset(self, df: pd.DataFrame) -> dict:
        results = {
            'missing_values': self._check_missing_values(df),
            'outliers': self._check_outliers(df),
            'distributions': self._analyze_distributions(df),
            'summary': self._generate_summary(df)
        }
        return self._convert_to_serializable(results)
