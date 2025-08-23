# Data Integrity Agent

A simple agent for validating dataset quality, completeness, and representativeness.

## Purpose
- Detect missing values and assess their impact
- Identify outliers using statistical methods  
- Analyze data distributions
- Generate validation reports with recommendations

## Usage
```python
from data_integrity_agent import BasicDataIntegrityAgent

agent = BasicDataIntegrityAgent()
report = agent.validate_dataset(df)
print(report)
```
