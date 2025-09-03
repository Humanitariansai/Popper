import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from data_integrity_agent import BasicDataIntegrityAgent

class LLMEnhancedDataIntegrityAgent(BasicDataIntegrityAgent):
    """
    LLM-enhanced data integrity agent that uses a LLM to analyze data and generate validation strategies.
    """
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self._configure_llm(api_key)
        self.validation_history = []
    
    def _configure_llm(self, api_key: Optional[str] = None):
        """Configure the LLM"""
        load_dotenv()
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Set it in .env file or pass as parameter.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def intelligent_validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ Main intelligent validation method using LLM orchestration """
        # LLM analyzes data and decides validation strategy
        strategy = self._analyze_data_and_plan_strategy(df)
        
        # Run validations based on LLM's decision
        validation_results = self._execute_validation_strategy(df, strategy)
        
        # Convert numpy/pandas types to JSON-serializable types
        serializable_results = self._convert_to_serializable(validation_results)
        
        # LLM interprets results and generates insights
        insights = self._interpret_validation_results(serializable_results, df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(serializable_results, insights)

        return {
            'validation_results': serializable_results,
            'llm_insights': insights,
            'strategy_used': strategy,
            'recommendations': recommendations,
            'data_quality_score': insights.get('data_quality_score', 0)
        }
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from markdown-wrapped responses"""
        import re
        
        # Remove markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # If no markdown blocks, try to find JSON object
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # If still no match, return the original text
        return response_text

    def _analyze_data_and_plan_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """LLM analyzes data and decides which validations to run"""
        
        prompt = f"""
        You are an expert data quality analyst. Analyze this dataset and create a validation strategy.
        
        Dataset Info:
        - Columns: {list(df.columns)}
        - Data types: {df.dtypes.to_dict()}
        - Shape: {df.shape}
        - Sample data: {df.head(3).to_dict()}
        
        Available validations:
        - missing_values: Check for null/NaN values
        - outliers: Detect statistical outliers using IQR method
        - distributions: Analyze data distributions and statistics
        - format_validation: Check data format consistency
        - range_validation: Verify values are within expected ranges
        - consistency_checks: Look for logical contradictions
        
        Return a JSON object with:
        {{
            "validations_to_run": ["list", "of", "validations"],
            "priority_order": ["ordered", "by", "importance"],
            "custom_thresholds": {{"outlier_threshold": 0.05}},
            "reasoning": "explanation of why these validations were chosen"
        }}

        Do not wrap the JSON in ```json or ```.
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            if not response.text or response.text.strip() == "":
                print("LLM returned empty response")
                raise ValueError("Empty response from LLM")
            
            # Extract JSON from potential markdown wrapping
            json_text = self._extract_json_from_response(response.text)
            strategy = json.loads(json_text)
            return strategy
        except Exception as e:
            print(f"LLM strategy generation failed: {e}")
            # Fallback to basic validations
            return {
                "validations_to_run": ["missing_values", "outliers", "distributions"],
                "priority_order": ["missing_values", "outliers", "distributions"],
                "custom_thresholds": {},
                "reasoning": "Fallback to basic validations due to LLM error"
            }
    
    def _execute_validation_strategy(self, df: pd.DataFrame, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validations based on LLM's strategy"""
        
        results = {}
        validations_to_run = strategy.get('validations_to_run', [])
        
        for validation in validations_to_run:
            if validation == 'missing_values':
                results['missing_values'] = self._check_missing_values(df)
            elif validation == 'outliers':
                results['outliers'] = self._check_outliers(df)
            elif validation == 'distributions':
                results['distributions'] = self._analyze_distributions(df)
            elif validation == 'format_validation':
                results['format_validation'] = self._validate_formats(df)
            elif validation == 'range_validation':
                results['range_validation'] = self._validate_ranges(df)
            elif validation == 'consistency_checks':
                results['consistency_checks'] = self._check_consistency(df)
        
        results['summary'] = self._generate_summary(df)
        
        return results
    
    def _interpret_validation_results(self, results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """LLM interprets validation results and generates insights"""
        
        prompt = f"""
        You are a data quality expert. Analyze these validation results and provide insights.
        
        Validation Results: {json.dumps(results, indent=2)}
        Dataset Info: {df.shape} shape, {list(df.columns)} columns
        
        Provide a JSON response with:
        {{
            "critical_issues": ["list", "of", "critical", "problems"],
            "moderate_issues": ["list", "of", "moderate", "concerns"],
            "data_quality_score": 85,
            "key_insights": ["list", "of", "important", "findings"],
            "risk_assessment": "high/medium/low risk assessment",
            "explanation": "detailed explanation of findings"
        }}
        
        Score data quality from 0-100 where:
        - 90-100: Excellent quality
        - 70-89: Good quality with minor issues
        - 50-69: Moderate quality with significant issues
        - 0-49: Poor quality requiring immediate attention

        Do not wrap the JSON in ```json or ```.
        """
        
        try:
            response = self.model.generate_content(prompt)
            insights = json.loads(self._extract_json_from_response(response.text))
            return insights
        except Exception as e:
            print(f"LLM insights generation failed: {e}")
            return {
                "critical_issues": ["LLM analysis failed"],
                "moderate_issues": [],
                "data_quality_score": 50,
                "key_insights": ["Fallback analysis due to LLM error"],
                "risk_assessment": "unknown",
                "explanation": "LLM analysis failed, using fallback"
            }
    
    def _generate_recommendations(self, results: Dict[str, Any], insights: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        
        prompt = f"""
        Based on these validation results and insights, generate specific, actionable recommendations.
        
        Results: {json.dumps(results, indent=2)}
        Insights: {json.dumps(insights, indent=2)}
        
        Provide a JSON array of recommendations:
        [
            "Specific action item 1",
            "Specific action item 2",
            "Specific action item 3"
        ]
        
        Make recommendations:
        - Specific and actionable
        - Prioritized by impact
        - Include both quick fixes and long-term improvements

        Do not wrap the JSON in ```json or ```.
        """
        
        try:
            response = self.model.generate_content(prompt)
            recommendations = json.loads(self._extract_json_from_response(response.text))
            return recommendations
        except Exception as e:
            print(f"LLM recommendations generation failed: {e}")
            return ["LLM analysis failed - review validation results manually"]
    
    def _validate_formats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data formats (email, date, etc.)"""
        format_issues = {}
        
        for col in df.columns:
            if 'email' in col.lower():
                format_issues[col] = self._validate_email_format(df[col])
            elif 'date' in col.lower():
                format_issues[col] = self._validate_date_format(df[col])
        
        return format_issues
    
    def _validate_email_format(self, series: pd.Series) -> Dict[str, Any]:
        """Validate email format"""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        invalid_emails = []
        for idx, value in series.items():
            if pd.notna(value) and not re.match(email_pattern, str(value)):
                invalid_emails.append({'index': idx, 'value': value})
        
        return {
            'invalid_count': len(invalid_emails),
            'invalid_percentage': (len(invalid_emails) / len(series)) * 100,
            'invalid_entries': invalid_emails
        }
    
    def _validate_date_format(self, series: pd.Series) -> Dict[str, Any]:
        """Validate date format"""
        from datetime import datetime
        
        invalid_dates = []
        for idx, value in series.items():
            if pd.notna(value):
                try:
                    datetime.strptime(str(value), '%Y-%m-%d')
                except ValueError:
                    invalid_dates.append({'index': idx, 'value': value})
        
        return {
            'invalid_count': len(invalid_dates),
            'invalid_percentage': (len(invalid_dates) / len(series)) * 100,
            'invalid_entries': invalid_dates
        }
    
    def _validate_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate value ranges"""
        range_issues = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if 'age' in col.lower():
                range_issues[col] = self._validate_age_range(df[col])
            elif 'score' in col.lower():
                range_issues[col] = self._validate_score_range(df[col])
        
        return range_issues
    
    def _validate_age_range(self, series: pd.Series) -> Dict[str, Any]:
        """Validate age values (0-150)"""
        invalid_ages = []
        for idx, value in series.items():
            if pd.notna(value) and (value < 0 or value > 150):
                invalid_ages.append({'index': idx, 'value': value})
        
        return {
            'invalid_count': len(invalid_ages),
            'invalid_percentage': (len(invalid_ages) / len(series)) * 100,
            'invalid_entries': invalid_ages
        }
    
    def _validate_score_range(self, series: pd.Series) -> Dict[str, Any]:
        """Validate score values (0-100)"""
        invalid_scores = []
        for idx, value in series.items():
            if pd.notna(value) and (value < 0 or value > 100):
                invalid_scores.append({'index': idx, 'value': value})
        
        return {
            'invalid_count': len(invalid_scores),
            'invalid_percentage': (len(invalid_scores) / len(series)) * 100,
            'invalid_entries': invalid_scores
        }
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for logical consistency issues"""
        consistency_issues = {}
        
        # Check for contradictory values
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_values = df[col].value_counts()
                if len(unique_values) == 1 and len(df) > 1:
                    consistency_issues[col] = f"All values are identical: {unique_values.index[0]}"
        
        return consistency_issues
