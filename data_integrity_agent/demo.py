#!/usr/bin/env python3
"""
Quick demo of all Data Integrity Agent features
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
from cli import DataIntegrityCLI
from config import get_config
from visualization import DataIntegrityVisualizer
from type_detector import DataTypeDetector
from llm_enhanced_agent import LLMEnhancedDataIntegrityAgent
from action_agent import DataRepairAgent

def create_demo_data():
    """Create sample data with various issues"""
    np.random.seed(42)
    
    data = {
        'id': range(1, 51),
        'age': np.random.normal(35, 10, 50),
        'email': ['user' + str(i) + '@email.com' for i in range(25)] + 
                ['invalid_email' + str(i) for i in range(25)],
        'score': np.random.normal(75, 15, 50),
        'date': ['2023-01-' + f'{i:02d}' for i in range(1, 26)] + 
                ['invalid_date' + str(i) for i in range(25)]
    }
    
    df = pd.DataFrame(data)
    
    # Add some issues
    df.loc[0, 'age'] = 150  # Outlier
    df.loc[10:15, 'age'] = np.nan  # Missing values
    df.loc[20:25, 'email'] = np.nan  # Missing values
    df.loc[30:35, 'score'] = np.nan  # More missing values
    df.loc[40, 'score'] = 200  # Another outlier
    
    return df

def show_data_comparison(original_df, repaired_df, issues):
    """Show before/after comparison of data"""
    print("\n=== DATA REPAIR COMPARISON ===")
    
    # Handle both LLM agent results (nested) and basic agent results (flat)
    if 'validation_results' in issues:
        # LLM agent result structure
        validation_data = issues['validation_results']
        print(f"   LLM Quality Score: {issues.get('data_quality_score', 'N/A')}/100")
    else:
        # Basic agent result structure
        validation_data = issues
    
    print("\n📊 BEFORE REPAIR:")
    if 'missing_values' in validation_data:
        missing_info = validation_data['missing_values']
        if 'total_missing' in missing_info:
            print(f"   Missing values: {missing_info['total_missing']}")
        else:
            print(f"   Missing values: {missing_info.get('counts', {}).get('age', 0) + missing_info.get('counts', {}).get('score', 0)}")
    
    if 'outliers' in validation_data:
        outlier_count = sum(info.get('count', 0) for info in validation_data['outliers'].values())
        print(f"   Outliers: {outlier_count}")
    
    print("\n🔧 AFTER REPAIR:")
    # Check what was actually fixed
    missing_after = repaired_df.isnull().sum().sum()
    print(f"   Missing values: {missing_after}")
    
    # Simple outlier check for repaired data
    outlier_count = 0
    for col in ['age', 'score']:
        if col in repaired_df.columns and repaired_df[col].dtype in ['int64', 'float64']:
            q1, q3 = repaired_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            outliers = ((repaired_df[col] < lower) | (repaired_df[col] > upper)).sum()
            outlier_count += outliers
    
    print(f"   Outliers: {outlier_count}")
    
    print("\n📈 SAMPLE COMPARISON:")
    print("   Original age column (first 10 rows):")
    print(f"   {original_df['age'].head(10).tolist()}")
    print("   Repaired age column (first 10 rows):")
    print(f"   {repaired_df['age'].head(10).tolist()}")

def main():
    print("=== Data Integrity Agent Demo ===\n")
    
    # Create demo data
    df = create_demo_data()
    df.to_csv('demo_data.csv', index=False)
    print("✅ Created demo_data.csv")
    
    # Test configuration
    config = get_config()
    print(f"✅ Configuration loaded - Missing values threshold: {config.missing_values['threshold_percentage']}%")
    
    # Test type detection
    detector = DataTypeDetector()
    column_info = detector.detect_column_types(df)
    print("✅ Data types detected:")
    for col, info in column_info.items():
        print(f"   {col}: {info.detected_type.value} (confidence: {info.confidence:.2f})")
    
    # Test LLM agent
    try:
        agent = LLMEnhancedDataIntegrityAgent()
        results = agent.intelligent_validate_dataset(df)
        print(f"✅ LLM validation complete - Quality Score: {results['data_quality_score']}/100")
    except Exception as e:
        print(f"⚠️  LLM validation failed: {e}")
        # Fallback to basic validation
        from data_integrity_agent import BasicDataIntegrityAgent
        basic_agent = BasicDataIntegrityAgent()
        results = basic_agent.validate_dataset(df)
        print(f"✅ Basic validation complete")
    
    # Test data repair agent
    print("\n🔧 TESTING DATA REPAIR AGENT")
    repair_agent = DataRepairAgent()
    repaired_df, repair_report = repair_agent.auto_repair_dataset(df)
    
    # Show repair results
    show_data_comparison(df, repaired_df, results)
    
    # Save repaired data
    repaired_df.to_csv('demo_data_repaired.csv', index=False)
    print("✅ Created demo_data_repaired.csv")
    
    # Test visualization with both original and repaired data
    visualizer = DataIntegrityVisualizer()
    
    # Create report for original data
    visualizer.create_comprehensive_report(results, df, 'demo_report_original.html')
    print("✅ Created demo_report_original.html")
    
    # Create report for repaired data
    repair_results = repair_agent.validation_agent.validate_dataset(repaired_df)
    visualizer.create_comprehensive_report(repair_results, repaired_df, 'demo_report_repaired.html')
    print("✅ Created demo_report_repaired.html")
    
    # Test CLI
    cli = DataIntegrityCLI()
    print("✅ CLI interface ready")
    
    print("\n=== Demo Complete ===")
    print("Files created:")
    print("- demo_data.csv (original data with issues)")
    print("- demo_data_repaired.csv (repaired data)")
    print("- demo_report_original.html (original data visualization)")
    print("- demo_report_repaired.html (repaired data visualization)")
    print("\nTry running: python cli.py validate demo_data.csv --verbose")

if __name__ == "__main__":
    main()