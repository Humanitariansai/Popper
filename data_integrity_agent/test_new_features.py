#!/usr/bin/env python3
"""
Test script for new Data Integrity Agent features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Import our new modules
from cli import DataIntegrityCLI
from config import get_config, get_config_manager
from visualization import DataIntegrityVisualizer
from type_detector import DataTypeDetector, SchemaValidator

def create_test_data():
    """Create test data with various data types and quality issues"""
    
    np.random.seed(42)
    
    data = {
        'id': range(1, 1001),
        'age': np.random.normal(35, 10, 1000),
        'email': [
            'john@email.com', 'invalid_email', 'jane@email.com', None, 'bob@email.com',
            'test@test.com', 'not_an_email', 'user@domain.org', '', 'admin@company.com'
        ] * 100,
        'score': np.random.normal(75, 15, 1000),
        'date_joined': [
            '2023-01-15', '2023-02-20', 'invalid_date', '2023-03-10', '2023-04-05',
            '2023-05-12', 'not_a_date', '2023-06-18', '2023-07-22', '2023-08-30'
        ] * 100,
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'is_active': np.random.choice([True, False, 'yes', 'no', 1, 0], 1000),
        'price': np.random.uniform(10, 1000, 1000),
        'phone': [
            '+1234567890', '123-456-7890', 'invalid_phone', '+1-234-567-8900',
            '1234567890', 'not_a_phone', '+44-20-7946-0958', '555-0123', '555-0124', '555-0125'
        ] * 100
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[100:150, 'age'] = np.nan
    df.loc[200:220, 'email'] = np.nan
    
    # Add some outliers
    df.loc[0, 'age'] = 150
    df.loc[1, 'score'] = 150
    df.loc[5, 'age'] = -5
    df.loc[6, 'score'] = 150
    
    return df

def test_configuration():
    """Test configuration system"""
    print("=" * 60)
    print("TESTING CONFIGURATION SYSTEM")
    print("=" * 60)
    
    # Get configuration
    config = get_config()
    print(f"Missing values threshold: {config.missing_values['threshold_percentage']}%")
    print(f"Outlier method: {config.outliers['method']}")
    print(f"LLM enabled: {config.llm['enabled']}")
    
    # Test configuration
    config_manager = get_config_manager()
    print("Configuration system working correctly")
    
    # Test configuration validation
    issues = config_manager.validate_config()
    if issues:
        print(f"Configuration issues: {issues}")
    else:
        print("Configuration is valid!")
    
    print()

def test_type_detection():
    """Test data type detection"""
    print("=" * 60)
    print("TESTING DATA TYPE DETECTION")
    print("=" * 60)
    
    df = create_test_data()
    detector = DataTypeDetector()
    
    # Detect types
    column_info = detector.detect_column_types(df)
    
    print("Detected data types:")
    for column, info in column_info.items():
        print(f"  {column}: {info.detected_type.value} (confidence: {info.confidence:.2f})")
        if info.patterns:
            print(f"    Patterns: {info.patterns}")
        if info.range_info:
            print(f"    Range: {info.range_info['min']:.2f} - {info.range_info['max']:.2f}")
    
    # Test schema validation
    validator = SchemaValidator()
    expected_schema = {
        'id': 'integer',
        'age': 'integer',
        'email': 'email',
        'score': 'float',
        'date_joined': 'date',
        'category': 'categorical',
        'is_active': 'boolean',
        'price': 'float',
        'phone': 'phone'
    }
    
    validation_results = validator.validate_schema(df, expected_schema)
    print(f"\nSchema validation overall valid: {validation_results['overall_valid']}")
    
    if validation_results['type_mismatches']:
        print("Type mismatches:")
        for mismatch in validation_results['type_mismatches']:
            print(f"  {mismatch['column']}: expected {mismatch['expected']}, detected {mismatch['detected']}")
    
    # Suggest schema
    suggested_schema = validator.suggest_schema(df)
    print(f"\nSuggested schema: {suggested_schema}")
    
    print()

def test_visualization():
    """Test visualization features"""
    print("=" * 60)
    print("TESTING VISUALIZATION FEATURES")
    print("=" * 60)
    
    df = create_test_data()
    
    # Create sample validation results
    results = {
        'data_quality_score': 75,
        'validation_results': {
            'missing_values': {
                'total_missing': 72,
                'total_percentage': 1.8,
                'percentages': {
                    'age': 5.1,
                    'email': 2.0,
                    'score': 0.0,
                    'date_joined': 0.0
                }
            },
            'outliers': {
                'age': {'count': 9, 'percentage': 0.9},
                'score': {'count': 10, 'percentage': 1.0}
            }
        },
        'recommendations': [
            'Fix email format issues in email column',
            'Investigate age outliers (values: 150, -5)',
            'Consider data type conversion for is_active column'
        ]
    }
    
    visualizer = DataIntegrityVisualizer()
    
    # Create comprehensive report
    report_path = visualizer.create_comprehensive_report(
        results, df, 'test_report.html', "Test Data Integrity Report"
    )
    print(f"Comprehensive report created: {report_path}")
    
    # Create missing values heatmap
    heatmap_path = visualizer.create_missing_values_heatmap(df, 'missing_heatmap.html')
    print(f"Missing values heatmap created: {heatmap_path}")
    
    # Create outlier analysis
    outlier_path = visualizer.create_outlier_analysis(df, results, 'outlier_analysis.html')
    print(f"Outlier analysis created: {outlier_path}")
    
    # Create distribution plots
    dist_path = visualizer.create_distribution_plots(df, 'distributions.html')
    print(f"Distribution plots created: {dist_path}")
    
    # Create correlation heatmap
    corr_path = visualizer.create_correlation_heatmap(df, 'correlation.html')
    print(f"Correlation heatmap created: {corr_path}")
    
    # Create interactive dashboard
    dashboard_path = visualizer.create_interactive_dashboard(results, df, 'dashboard.html')
    print(f"Interactive dashboard created: {dashboard_path}")
    
    print()

def test_cli():
    """Test CLI functionality"""
    print("=" * 60)
    print("TESTING CLI FUNCTIONALITY")
    print("=" * 60)
    
    # Save test data
    df = create_test_data()
    df.to_csv('test_data.csv', index=False)
    
    # Test CLI help
    cli = DataIntegrityCLI()
    
    print("CLI help output:")
    cli.parser.print_help()
    
    print("\nTesting list-formats command:")
    cli.run(['list-formats'])
    
    print("\nTesting basic validation:")
    cli.run(['validate', 'test_data.csv', '--output', 'test_results.json'])
    
    print("\nTesting with verbose output:")
    cli.run(['validate', 'test_data.csv', '--verbose'])
    
    print()

def main():
    """Run all tests"""
    print("DATA INTEGRITY AGENT - NEW FEATURES TEST")
    print("=" * 60)
    
    try:
        test_configuration()
        test_type_detection()
        test_visualization()
        test_cli()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Clean up test files
        test_files = ['test_data.csv', 'test_results.json']
        for file in test_files:
            if Path(file).exists():
                Path(file).unlink()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
