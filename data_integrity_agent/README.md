# Data Integrity Agent

A simple agent for validating dataset quality, completeness, and representativeness.

## Purpose
- Detect missing values and assess their impact
- Identify outliers using statistical methods  
- Analyze data distributions
- Generate validation reports with recommendations

## Two Versions Available

### 1. Basic Agent (No LLM)
```python
from data_integrity_agent import BasicDataIntegrityAgent

agent = BasicDataIntegrityAgent()
report = agent.validate_dataset(df)
print(report)
```

### 2. LLM-Enhanced Agent (Intelligent)
```python
from llm_enhanced_agent import LLMEnhancedDataIntegrityAgent

agent = LLMEnhancedDataIntegrityAgent()
results = agent.intelligent_validate_dataset(df, domain="customer_data")
print(results['data_quality_score'])
print(results['recommendations'])
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For LLM-enhanced version, set your Google API key:
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

3. Run examples:
```bash
# Basic agent
python example.py

# LLM-enhanced agent
python llm_example.py
```

## How It Works

### Basic Agent
- Runs all validation functions statically
- Returns raw validation results
- No intelligent decision-making

### LLM-Enhanced Agent
1. **LLM Analysis**: Analyzes data and decides validation strategy
2. **Smart Execution**: Runs only relevant validations
3. **Intelligent Interpretation**: LLM explains results and their impact
4. **Actionable Output**: Provides specific recommendations for improvement
