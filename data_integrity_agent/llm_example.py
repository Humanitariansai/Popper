import traceback
import pandas as pd
import numpy as np
from llm_enhanced_agent import LLMEnhancedDataIntegrityAgent

def create_complex_sample_data():
    """Create sample dataset with various data quality issues"""
    np.random.seed(42)
    
    data = {
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
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[100:150, 'age'] = np.nan
    df.loc[200:220, 'email'] = np.nan
    
    # Add some outliers
    df.loc[0, 'age'] = 150
    df.loc[1, 'score'] = 150
    
    # Add some range violations
    df.loc[5, 'age'] = -5
    df.loc[6, 'score'] = 150
    
    return df

def main():
    # Create sample data
    df = create_complex_sample_data()
    
    print("Sample Data Preview:")
    print(df.head())
    print("\n" + "="*50 + "\n")
    
    # Initialize LLM-enhanced agent
    try:
        agent = LLMEnhancedDataIntegrityAgent()
        
        # Run intelligent validation
        print("Running LLM-Enhanced Data Validation...")
        results = agent.intelligent_validate_dataset(df, domain="customer_data")
        
        # Display results
        print("\n" + "="*50)
        print("LLM-ENHANCED VALIDATION RESULTS")
        print("="*50)
        
        # Strategy used
        print(f"\nValidation Strategy:")
        print(f"Validations run: {results['strategy_used']['validations_to_run']}")
        print(f"Reasoning: {results['strategy_used']['reasoning']}")
        
        # Data quality score
        print(f"\nData Quality Score: {results['data_quality_score']}/100")
        
        # LLM insights
        insights = results['llm_insights']
        print(f"\nCritical Issues:")
        for issue in insights.get('critical_issues', []):
            print(f"  • {issue}")
        
        print(f"\nModerate Issues:")
        for issue in insights.get('moderate_issues', []):
            print(f"  • {issue}")
        
        print(f"\nKey Insights:")
        for insight in insights.get('key_insights', []):
            print(f"  • {insight}")
        
        # Recommendations
        print(f"\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # Risk assessment
        print(f"\nRisk Assessment: {insights.get('risk_assessment', 'Unknown')}")
        
        # Detailed validation results
        print(f"\nDetailed Validation Results:")
        validation_results = results['validation_results']
        
        if 'missing_values' in validation_results:
            missing = validation_results['missing_values']
            print(f"  Missing Values: {missing['total_missing']} total ({missing['total_percentage']:.2f}%)")
        
        if 'outliers' in validation_results:
            outliers = validation_results['outliers']
            for col, info in outliers.items():
                print(f"  {col} outliers: {info['count']} ({info['percentage']:.2f}%)")
        
        if 'format_validation' in validation_results:
            format_issues = validation_results['format_validation']
            for col, info in format_issues.items():
                print(f"  {col} format issues: {info['invalid_count']} ({info['invalid_percentage']:.2f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
