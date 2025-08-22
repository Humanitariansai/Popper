import pandas as pd
import numpy as np
from basic_data_integrity_agent import BasicDataIntegrityAgent

def create_sample_data():
    """Create sample dataset with some data quality issues"""
    np.random.seed(42)
    
    data = {
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'score': np.random.normal(75, 15, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[100:150, 'age'] = np.nan
    df.loc[200:220, 'income'] = np.nan
    
    # Add some outliers
    df.loc[0, 'age'] = 150
    df.loc[1, 'income'] = 1000000
    
    return df

def main():
    # Create sample data
    df = create_sample_data()
    
    # Initialize agent
    agent = BasicDataIntegrityAgent()
    
    # Run validation
    results = agent.validate_dataset(df)
    
    # Print results
    print("Data Integrity Validation Results:")
    print("=" * 40)
    
    print(f"\nDataset Summary:")
    print(f"Total rows: {results['summary']['total_rows']}")
    print(f"Total columns: {results['summary']['total_columns']}")
    print(f"Numeric columns: {results['summary']['numeric_columns']}")
    print(f"Categorical columns: {results['summary']['categorical_columns']}")
    
    print(f"\nMissing Values:")
    print(f"Total missing: {results['missing_values']['total_missing']}")
    print(f"Total percentage: {results['missing_values']['total_percentage']:.2f}%")
    
    print(f"\nOutliers:")
    for col, info in results['outliers'].items():
        print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")

if __name__ == "__main__":
    main()
