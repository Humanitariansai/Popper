"""
Data Type Detection and Schema Validation Module
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

class DataType(Enum):
    """Supported data types"""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"

@dataclass
class ColumnInfo:
    """Information about a column's data type and characteristics"""
    name: str
    detected_type: DataType
    confidence: float
    sample_values: List[Any]
    unique_count: int
    null_count: int
    patterns: List[str]
    range_info: Optional[Dict[str, Any]] = None
    format_info: Optional[Dict[str, Any]] = None

class DataTypeDetector:
    """Detects data types and validates schemas"""
    
    def __init__(self):
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': re.compile(r'^\+?1?\d{9,15}$'),
            'url': re.compile(r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'),
            'date': re.compile(r'^\d{4}-\d{2}-\d{2}$'),
            'datetime': re.compile(r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}'),
            'currency': re.compile(r'^[\$€£¥]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?$'),
            'percentage': re.compile(r'^\d+(?:\.\d+)?%$'),
            'boolean': re.compile(r'^(true|false|yes|no|1|0|t|f|y|n)$', re.IGNORECASE)
        }
        
        self.date_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
    
    def detect_column_types(self, df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, ColumnInfo]:
        """Detect data types for all columns in the dataframe"""
        column_info = {}
        
        for column in df.columns:
            column_info[column] = self._detect_single_column_type(df[column], sample_size)
        
        return column_info
    
    def _detect_single_column_type(self, series: pd.Series, sample_size: int) -> ColumnInfo:
        """Detect data type for a single column"""
        
        # Get sample data
        sample_data = series.dropna().head(sample_size)
        if len(sample_data) == 0:
            return ColumnInfo(
                name=series.name,
                detected_type=DataType.UNKNOWN,
                confidence=0.0,
                sample_values=[],
                unique_count=0,
                null_count=series.isnull().sum(),
                patterns=[]
            )
        
        # Check for boolean type
        if self._is_boolean(sample_data):
            return self._create_column_info(series, DataType.BOOLEAN, 0.95)
        
        # Check for numeric types
        if self._is_numeric(sample_data):
            numeric_type, confidence = self._detect_numeric_type(sample_data)
            return self._create_column_info(series, numeric_type, confidence)
        
        # Check for date/datetime types
        if self._is_date(sample_data):
            date_type, confidence = self._detect_date_type(sample_data)
            return self._create_column_info(series, date_type, confidence)
        
        # Check for specific string patterns
        pattern_type, confidence = self._detect_pattern_type(sample_data)
        if pattern_type != DataType.STRING:
            return self._create_column_info(series, pattern_type, confidence)
        
        # Check for categorical
        if self._is_categorical(sample_data):
            return self._create_column_info(series, DataType.CATEGORICAL, 0.8)
        
        # Default to string
        return self._create_column_info(series, DataType.STRING, 0.7)
    
    def _is_boolean(self, sample_data: pd.Series) -> bool:
        """Check if data represents boolean values"""
        boolean_pattern = self.patterns['boolean']
        boolean_count = sum(1 for val in sample_data if boolean_pattern.match(str(val)))
        return boolean_count / len(sample_data) > 0.8
    
    def _is_numeric(self, sample_data: pd.Series) -> bool:
        """Check if data represents numeric values"""
        numeric_count = 0
        for val in sample_data:
            try:
                float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
            except ValueError:
                continue
        
        return numeric_count / len(sample_data) > 0.8
    
    def _detect_numeric_type(self, sample_data: pd.Series) -> Tuple[DataType, float]:
        """Detect specific numeric type (integer, float, currency, percentage)"""
        
        # Check for percentage
        percentage_count = sum(1 for val in sample_data if self.patterns['percentage'].match(str(val)))
        if percentage_count / len(sample_data) > 0.5:
            return DataType.PERCENTAGE, 0.9
        
        # Check for currency
        currency_count = sum(1 for val in sample_data if self.patterns['currency'].match(str(val)))
        if currency_count / len(sample_data) > 0.5:
            return DataType.CURRENCY, 0.9
        
        # Check if all values are integers
        integer_count = 0
        for val in sample_data:
            try:
                float_val = float(str(val).replace(',', ''))
                if float_val.is_integer():
                    integer_count += 1
            except ValueError:
                continue
        
        if integer_count / len(sample_data) > 0.9:
            return DataType.INTEGER, 0.95
        else:
            return DataType.FLOAT, 0.9
    
    def _is_date(self, sample_data: pd.Series) -> bool:
        """Check if data represents date values"""
        date_count = 0
        for val in sample_data:
            if self._is_valid_date(str(val)):
                date_count += 1
        
        return date_count / len(sample_data) > 0.5
    
    def _detect_date_type(self, sample_data: pd.Series) -> Tuple[DataType, float]:
        """Detect if data is date or datetime"""
        datetime_count = 0
        for val in sample_data:
            if self.patterns['datetime'].match(str(val)):
                datetime_count += 1
        
        if datetime_count / len(sample_data) > 0.3:
            return DataType.DATETIME, 0.9
        else:
            return DataType.DATE, 0.85
    
    def _detect_pattern_type(self, sample_data: pd.Series) -> Tuple[DataType, float]:
        """Detect specific string patterns (email, phone, url)"""
        
        # Check for email
        email_count = sum(1 for val in sample_data if self.patterns['email'].match(str(val)))
        if email_count / len(sample_data) > 0.7:
            return DataType.EMAIL, 0.95
        
        # Check for phone
        phone_count = sum(1 for val in sample_data if self.patterns['phone'].match(str(val)))
        if phone_count / len(sample_data) > 0.7:
            return DataType.PHONE, 0.9
        
        # Check for URL
        url_count = sum(1 for val in sample_data if self.patterns['url'].match(str(val)))
        if url_count / len(sample_data) > 0.7:
            return DataType.URL, 0.9
        
        return DataType.STRING, 0.7
    
    def _is_categorical(self, sample_data: pd.Series) -> bool:
        """Check if data represents categorical values"""
        unique_ratio = sample_data.nunique() / len(sample_data)
        return unique_ratio < 0.1 and sample_data.nunique() < 50
    
    def _is_valid_date(self, value: str) -> bool:
        """Check if a string represents a valid date"""
        for fmt in self.date_formats:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False
    
    def _create_column_info(self, series: pd.Series, data_type: DataType, confidence: float) -> ColumnInfo:
        """Create ColumnInfo object with additional metadata"""
        
        sample_values = series.dropna().head(10).tolist()
        unique_count = series.nunique()
        null_count = series.isnull().sum()
        
        # Extract patterns
        patterns = self._extract_patterns(series)
        
        # Get range info for numeric types
        range_info = None
        if data_type in [DataType.INTEGER, DataType.FLOAT, DataType.CURRENCY, DataType.PERCENTAGE]:
            range_info = self._get_range_info(series)
        
        # Get format info for date types
        format_info = None
        if data_type in [DataType.DATE, DataType.DATETIME]:
            format_info = self._get_format_info(series)
        
        return ColumnInfo(
            name=series.name,
            detected_type=data_type,
            confidence=confidence,
            sample_values=sample_values,
            unique_count=unique_count,
            null_count=null_count,
            patterns=patterns,
            range_info=range_info,
            format_info=format_info
        )
    
    def _extract_patterns(self, series: pd.Series) -> List[str]:
        """Extract common patterns from the data"""
        patterns = []
        sample_data = series.dropna().head(100)
        
        # Check for common patterns
        for pattern_name, pattern in self.patterns.items():
            matches = sum(1 for val in sample_data if pattern.match(str(val)))
            if matches / len(sample_data) > 0.3:
                patterns.append(pattern_name)
        
        return patterns
    
    def _get_range_info(self, series: pd.Series) -> Dict[str, Any]:
        """Get range information for numeric columns"""
        numeric_data = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_data) == 0:
            return {}
        
        return {
            'min': float(numeric_data.min()),
            'max': float(numeric_data.max()),
            'mean': float(numeric_data.mean()),
            'std': float(numeric_data.std()),
            'q25': float(numeric_data.quantile(0.25)),
            'q75': float(numeric_data.quantile(0.75))
        }
    
    def _get_format_info(self, series: pd.Series) -> Dict[str, Any]:
        """Get format information for date columns"""
        sample_data = series.dropna().head(50)
        formats = []
        
        for val in sample_data:
            for fmt in self.date_formats:
                try:
                    datetime.strptime(str(val), fmt)
                    formats.append(fmt)
                    break
                except ValueError:
                    continue
        
        if formats:
            most_common_format = max(set(formats), key=formats.count)
            return {
                'detected_format': most_common_format,
                'format_confidence': formats.count(most_common_format) / len(formats)
            }
        
        return {}

class SchemaValidator:
    """Validates data against expected schemas"""
    
    def __init__(self):
        self.detector = DataTypeDetector()
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataframe against expected schema"""
        
        detected_types = self.detector.detect_column_types(df)
        validation_results = {
            'overall_valid': True,
            'column_validations': {},
            'schema_violations': [],
            'type_mismatches': [],
            'missing_columns': [],
            'extra_columns': []
        }
        
        # Check for missing columns
        expected_columns = set(expected_schema.keys())
        actual_columns = set(df.columns)
        
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        
        validation_results['missing_columns'] = list(missing_columns)
        validation_results['extra_columns'] = list(extra_columns)
        
        if missing_columns:
            validation_results['overall_valid'] = False
        
        # Validate each column
        for column, expected_type in expected_schema.items():
            if column not in df.columns:
                continue
            
            column_validation = self._validate_column(
                df[column], 
                expected_type, 
                detected_types.get(column)
            )
            
            validation_results['column_validations'][column] = column_validation
            
            if not column_validation['valid']:
                validation_results['overall_valid'] = False
                validation_results['type_mismatches'].append({
                    'column': column,
                    'expected': expected_type,
                    'detected': detected_types[column].detected_type.value if detected_types[column] else 'unknown'
                })
        
        return validation_results
    
    def _validate_column(self, series: pd.Series, expected_type: str, detected_info: Optional[ColumnInfo]) -> Dict[str, Any]:
        """Validate a single column against expected type"""
        
        validation = {
            'valid': True,
            'issues': [],
            'detected_type': detected_info.detected_type.value if detected_info else 'unknown',
            'confidence': detected_info.confidence if detected_info else 0.0
        }
        
        # Type compatibility check
        if detected_info and not self._is_type_compatible(expected_type, detected_info.detected_type):
            validation['valid'] = False
            validation['issues'].append(f"Type mismatch: expected {expected_type}, detected {detected_info.detected_type.value}")
        
        # Additional validation based on type
        if expected_type == 'email':
            validation.update(self._validate_email_column(series))
        elif expected_type == 'date':
            validation.update(self._validate_date_column(series))
        elif expected_type in ['integer', 'float']:
            validation.update(self._validate_numeric_column(series, expected_type))
        
        return validation
    
    def _is_type_compatible(self, expected: str, detected: DataType) -> bool:
        """Check if detected type is compatible with expected type"""
        
        compatibility_map = {
            'integer': [DataType.INTEGER, DataType.FLOAT],
            'float': [DataType.FLOAT, DataType.INTEGER],
            'string': [DataType.STRING, DataType.CATEGORICAL, DataType.EMAIL, DataType.PHONE, DataType.URL],
            'boolean': [DataType.BOOLEAN],
            'date': [DataType.DATE, DataType.DATETIME],
            'datetime': [DataType.DATETIME, DataType.DATE],
            'email': [DataType.EMAIL],
            'phone': [DataType.PHONE],
            'url': [DataType.URL],
            'categorical': [DataType.CATEGORICAL, DataType.STRING]
        }
        
        return detected in compatibility_map.get(expected, [])
    
    def _validate_email_column(self, series: pd.Series) -> Dict[str, Any]:
        """Validate email column"""
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        invalid_emails = []
        
        for idx, val in series.items():
            if pd.notna(val) and not email_pattern.match(str(val)):
                invalid_emails.append({'index': idx, 'value': val})
        
        return {
            'invalid_emails': invalid_emails,
            'valid': len(invalid_emails) == 0
        }
    
    def _validate_date_column(self, series: pd.Series) -> Dict[str, Any]:
        """Validate date column"""
        invalid_dates = []
        date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
        
        for idx, val in series.items():
            if pd.notna(val):
                valid_date = False
                for fmt in date_formats:
                    try:
                        datetime.strptime(str(val), fmt)
                        valid_date = True
                        break
                    except ValueError:
                        continue
                
                if not valid_date:
                    invalid_dates.append({'index': idx, 'value': val})
        
        return {
            'invalid_dates': invalid_dates,
            'valid': len(invalid_dates) == 0
        }
    
    def _validate_numeric_column(self, series: pd.Series, expected_type: str) -> Dict[str, Any]:
        """Validate numeric column"""
        invalid_numbers = []
        
        for idx, val in series.items():
            if pd.notna(val):
                try:
                    num_val = float(str(val).replace(',', ''))
                    if expected_type == 'integer' and not float(num_val).is_integer():
                        invalid_numbers.append({'index': idx, 'value': val, 'reason': 'Not an integer'})
                except ValueError:
                    invalid_numbers.append({'index': idx, 'value': val, 'reason': 'Not a number'})
        
        return {
            'invalid_numbers': invalid_numbers,
            'valid': len(invalid_numbers) == 0
        }
    
    def suggest_schema(self, df: pd.DataFrame) -> Dict[str, str]:
        """Suggest a schema based on detected types"""
        detected_types = self.detector.detect_column_types(df)
        
        schema = {}
        for column, info in detected_types.items():
            if info.confidence > 0.7:
                schema[column] = info.detected_type.value
            else:
                schema[column] = 'string'  # Default to string for low confidence
        
        return schema
