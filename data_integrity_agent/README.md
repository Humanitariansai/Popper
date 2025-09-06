# Data Integrity Agent

A modular system for validating, analyzing, and repairing dataset quality issues. The system includes basic validation functions, LLM-enhanced intelligent analysis, and automated data repair capabilities.

## Project Status

This is a functional prototype with agent-like capabilities. It demonstrates intelligent decision-making and autonomous actions but lacks persistent memory, continuous operation, and complex goal planning that would characterize a full autonomous agent.

## Architecture Overview

The system consists of three main components:

- **Basic Agent**: Static validation functions for data quality assessment
- **LLM-Enhanced Agent**: Uses Google Gemini for intelligent validation strategy and insights
- **Action Agent**: Automatically repairs common data quality issues


## Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Google Generative AI API key (for LLM features)

## Setup Instructions

### 1. Environment Setup

```bash
# Navigate to project directory
cd data_integrity_agent

# Install dependencies
pip install -r requirements.txt
```

### 2. LLM Configuration (Optional)

For LLM-enhanced features, create a `.env` file:

```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

To obtain a Google Generative AI API key:
1. Go to Google AI Studio (ai.google.dev)
2. Create a new project or select existing
3. Navigate to "Get API key"
4. Generate and copy the key

### 3. Dependencies

Core dependencies (see `requirements.txt`):
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.20.0` - Numerical operations  
- `google-generativeai>=0.3.0` - LLM integration
- `python-dotenv>=0.19.0` - Environment variables

Extended dependencies (installed automatically):
- `tqdm`, `colorama` - CLI progress and colors
- `openpyxl`, `pyarrow`, `tables` - File format support
- `PyYAML` - Configuration files
- `plotly` - Interactive visualizations

## Usage

### Quick Start

```bash
# Run comprehensive demo
python demo.py

# Basic validation
python cli.py validate your_data.csv

# LLM-enhanced validation
python cli.py validate your_data.csv --llm

# Batch processing
python cli.py validate-batch data_folder/ --output results/
```

### Programmatic Usage

**Basic Validation:**
```python
from data_integrity_agent import BasicDataIntegrityAgent
import pandas as pd

agent = BasicDataIntegrityAgent()
df = pd.read_csv('your_data.csv')
results = agent.validate_dataset(df)
```

**LLM-Enhanced Validation:**
```python
from llm_enhanced_agent import LLMEnhancedDataIntegrityAgent

agent = LLMEnhancedDataIntegrityAgent()
results = agent.intelligent_validate_dataset(df)
print(f"Quality Score: {results['data_quality_score']}/100")
```

**Data Repair:**
```python
from action_agent import DataRepairAgent

repair_agent = DataRepairAgent()
repaired_df, report = repair_agent.auto_repair_dataset(df)
```

## Component Details

### Basic Agent (`data_integrity_agent.py`)

**Purpose**: Core validation functionality without LLM dependencies.

**Key Methods**:
- `validate_dataset(df)`: Main validation entry point
- `_check_missing_values(df)`: Identifies null/NaN values
- `_check_outliers(df)`: Uses IQR method for outlier detection
- `_analyze_distributions(df)`: Statistical distribution analysis
- `_convert_to_serializable(obj)`: Handles numpy/pandas JSON serialization

**Output**: Dictionary with validation results for missing values, outliers, distributions, and summary statistics.

### LLM-Enhanced Agent (`llm_enhanced_agent.py`)

**Purpose**: Intelligent validation using Google Gemini LLM for strategy and insights.

**Key Methods**:
- `intelligent_validate_dataset(df)`: Main LLM-orchestrated validation
- `_analyze_data_and_plan_strategy(df)`: LLM decides validation approach
- `_execute_validation_strategy(df, strategy)`: Runs selected validations
- `_interpret_validation_results(results, df)`: LLM analyzes findings
- `_generate_recommendations(results, insights)`: Creates actionable suggestions

**LLM Integration**:
- Uses Google Gemini-1.5-flash model
- Handles JSON extraction from markdown-wrapped responses
- Fallback to basic validation if LLM fails
- Constrained to predefined validation menu (not fully autonomous)

**Output**: Nested structure with validation results, LLM insights, quality score, and recommendations.

### Action Agent (`action_agent.py`)

**Purpose**: Automated data repair capabilities.

**Key Methods**:
- `auto_repair_dataset(df)`: Main repair orchestration
- `_fix_missing_values(df, details)`: Mean/mode imputation
- `_handle_outliers(df, details)`: IQR-based capping
- `_generate_repair_report(issues)`: Documents repair actions

**Repair Strategies**:
- Missing values: Mean for numeric, mode for categorical
- Outliers: IQR capping (Q1 - 1.5*IQR, Q3 + 1.5*IQR)

### Configuration System (`config.py`)

**Purpose**: Centralized configuration management with YAML support.

**Key Components**:
- `ValidationConfig`: Dataclass for all settings
- `ConfigManager`: Loads/saves/validates configuration
- Global configuration functions: `get_config()`, `update_config()`

**Configuration Sections**:
- Missing values thresholds
- Outlier detection parameters
- Distribution analysis settings
- Format validation patterns
- Range validation limits
- LLM model settings

### CLI Interface (`cli.py`)

**Purpose**: Command-line interface with colored output and progress bars.

**Commands**:
- `validate`: Single file validation
- `validate-batch`: Multiple file processing
- `list-formats`: Show supported file types

**Features**:
- Colored output with status indicators
- Progress bars for batch operations
- Multiple output formats (JSON, HTML, text)
- Automatic fallback from LLM to basic validation
- Supports CSV, Excel, JSON, Parquet, HDF5, Pickle

### Type Detection (`type_detector.py`)

**Purpose**: Automatic data type inference with confidence scoring.

**Detected Types**:
- Numeric: integer, float
- Text: string, categorical
- Temporal: date, datetime
- Structured: email, phone, URL
- Financial: currency, percentage

**Key Classes**:
- `DataTypeDetector`: Infers column types
- `SchemaValidator`: Validates against expected schemas
- `ColumnInfo`: Type information with confidence

### Visualization (`visualization.py`)

**Purpose**: Interactive charts and HTML reports using Plotly.

**Report Types**:
- Missing values heatmaps
- Outlier analysis plots
- Distribution histograms
- Correlation matrices
- Interactive dashboards

**Key Methods**:
- `create_comprehensive_report()`: Full HTML report
- `create_missing_values_heatmap()`: Visualization of missing data
- `create_outlier_analysis()`: Box plots and scatter plots

## Current Limitations

### Agent-like Qualities (Present)
- Intelligent decision-making (LLM chooses validation strategy)
- Autonomous actions (auto-repairs data without human intervention)
- Goal-oriented behavior (works toward data quality improvement)
- Multi-step reasoning (validate → analyze → recommend → repair)

### Missing Agent Qualities
- **No persistent memory**: Doesn't learn from previous validations
- **No continuous operation**: Only runs when invoked
- **No proactive behavior**: Doesn't monitor data sources
- **Constrained decision-making**: LLM chooses from predefined validation menu
- **No complex goal planning**: Follows simple validate→repair workflow

## Common Issues & Solutions

### 1. LLM JSON Parsing Errors
**Problem**: "Expecting value: line 1 column 1 (char 0)"
**Solution**: The system includes `_extract_json_from_response()` to handle markdown-wrapped JSON responses.

### 2. Numpy Serialization Errors
**Problem**: "Object of type int64 is not JSON serializable"
**Solution**: Both agents include `_convert_to_serializable()` methods to handle numpy types.


### 3. Dependency Conflicts
**Problem**: numpy/pandas compatibility issues
**Solution**: Use specific compatible versions:
```bash
pip install numpy==1.24.3 pandas==1.5.3
```

## Configuration Examples

### Basic Configuration (`config.yaml`)
```yaml
missing_values:
  threshold_percentage: 5.0
  critical_threshold: 20.0

outliers:
  method: iqr
  iqr_multiplier: 1.5
  
llm:
  model: gemini-1.5-flash
  temperature: 0.1
```

### Programmatic Configuration
```python
from config import update_config

update_config({
    'outliers': {'iqr_multiplier': 2.0},
    'llm': {'temperature': 0.2}
})
```

## Testing
- `demo.py` serves as integration test
- Each component has basic validation
- Manual testing through CLI interface

## Future Development

### Immediate Improvements
1. **Remove LLM constraints**: Allow custom validation logic generation
2. **Add persistent storage**: Store validation history and patterns
3. **Implement learning**: Improve strategies based on past results

### Long-term Enhancements
1. **Continuous monitoring**: Watch data sources for changes
2. **Goal decomposition**: Break complex objectives into subtasks
3. **Multi-agent coordination**: Integration with other system components
4. **Advanced repair strategies**: ML-based imputation and correction
