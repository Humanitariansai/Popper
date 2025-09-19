# 🤖 Algorithmic Bias Detection Agent

A comprehensive bias detection and fairness analysis system that combines algorithmic fairness evaluation, representation bias assessment, and AI-powered insights to help organizations build more equitable machine learning systems.

## 🌟 Features

### 🎯 4-Step Comprehensive Analysis Workflow

1. **📊 Configuration & Data Management**
2. **🔄 Fairness Report Generation** 
3. **🤖 LLM-Powered Analysis**
4. **🌍 Representation Bias Assessment**

### 🔬 Advanced Capabilities

- **Multiple Fairness Metrics**: Demographic Parity, Equal Opportunity, Disparate Impact
- **15+ Diversity Indices**: Shannon, Simpson, Hill Numbers, Gini-Simpson, and more
- **Population Benchmarking**: Compare against real US Census data
- **Temporal Tracking**: Monitor bias and diversity trends over time
- **AI Integration**: GROQ LLM for intelligent analysis and recommendations
- **Interactive Dashboard**: Streamlit-powered web interface
- **Automated Reporting**: Generate detailed JSON reports for compliance

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- GROQ API key for LLM features

### Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd Bias_Detection_Agent
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy the example environment file
cp .env.example .env
# Edit .env file and add your GROQ API key
```

5. **Run the application**
```bash
streamlit run app/app_streamlit.py
```

6. **Open your browser** to http://localhost:8501

## 📁 Project Structure

```
Bias_Detection_Agent/
├── app/
│   └── app_streamlit.py          # Main Streamlit dashboard
├── scripts/
│   ├── train.py                  # Model training script
│   ├── evaluate_fairness.py      # Fairness evaluation
│   └── tradeoff_sweep.py         # Bias-accuracy tradeoff analysis
├── reports/                      # Generated analysis reports
│   ├── model.pkl                 # Trained models
│   ├── fairness_summary.md       # Human-readable summaries
│   ├── representation_*.png      # Visualization charts
│   └── diversity_tracking.json   # Detailed tracking reports
├── data/
│   └── adult.csv                 # Default dataset (UCI Adult/Census)
├── fairness_agent.py             # Core fairness analysis engine
├── fairness_llm_agent.py         # LLM integration for insights
├── Representation_Bias_Agent.py  # Representation bias analysis
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎛️ Usage Guide

### Step 1: Configure Analysis

1. **Upload Dataset** (optional - uses adult.csv by default)
   - Supported format: CSV
   - Automatic data validation and preprocessing

2. **Set Parameters**
   - Target column (e.g., "income")
   - Positive label (e.g., ">50K")
   - Protected attributes (e.g., "gender,race")
   - Fairness thresholds

3. **Update Configuration**
   - Dynamic config.yaml updates
   - Persistent settings across sessions

### Step 2: Generate Fairness Reports

1. **Click "🚀 Generate Latest Fairness Reports"**
2. **Automated Pipeline Execution**:
   - Model training with scikit-learn
   - Fairness metric calculation
   - Bias-accuracy tradeoff analysis
3. **Real-time Verdicts**: Pass/Fail status for each protected attribute

### Step 3: LLM Analysis

1. **AI-Powered Insights**: Click "Analyze Fairness Metrics with LLM"
2. **Comprehensive Analysis**:
   - Key findings summary
   - Bias identification and severity assessment
   - Actionable recommendations
   - Threshold appropriateness evaluation
3. **Interactive Chat**: Ask follow-up questions via sidebar

### Step 4: Representation Bias Analysis

1. **Dataset Representation Assessment**: Click "🌍 Analyze Dataset Representation"
2. **Comprehensive Evaluation**:
   - **Demographic Distribution**: Group counts and proportions
   - **Population Benchmarking**: Compare vs. US Census data
   - **Inclusion Gap Analysis**: Identify missing demographic groups
   - **Advanced Diversity Metrics**: 15+ mathematical indices
   - **Temporal Tracking**: Monitor trends over time
   - **Historical Context**: Dataset limitations and considerations

## 📊 Fairness Metrics

### Core Metrics
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equal Opportunity**: Equal true positive rates across groups  
- **Disparate Impact**: Ratio of positive rates between groups

### Thresholds (Configurable)
- Demographic Parity Gap: ≤ 0.1 (default)
- Equal Opportunity Gap: ≤ 0.1 (default)
- Disparate Impact Ratio: ≥ 0.8 (default)

## 🎲 Diversity Metrics

### Basic Indices
- **Shannon Diversity (H')**: Information-theoretic measure
- **Simpson Diversity**: Probability that two individuals differ
- **Berger-Parker Dominance**: Proportion of most abundant group

### Advanced Indices
- **Hill Numbers (H0, H1, H2)**: True diversity measures
- **Gini-Simpson Index**: Alternative Simpson formulation
- **Inverse Simpson**: Reciprocal of Simpson index
- **Rényi Diversity**: Generalized entropy measure
- **Pielou's Evenness (J')**: Distribution uniformity
- **Theil Index**: Inequality measure

### Temporal Tracking
- **Trend Analysis**: Increasing/Decreasing/Stable patterns
- **Volatility Measurement**: Diversity fluctuation quantification
- **Subgroup Comparison**: Cross-category diversity analysis

## 🤖 LLM Integration

### Powered by GROQ
- **Fast Inference**: Sub-second response times
- **Expert Analysis**: Trained on fairness and ML literature
- **Contextual Insights**: Considers your specific configuration

### Capabilities
- Fairness metric interpretation
- Bias severity assessment
- Improvement recommendations
- Interactive Q&A sessions
- Technical explanation in plain language

## 📈 Output Reports

### Fairness Reports
- `reports/criteria_[attribute].csv`: Detailed metric calculations
- `reports/fairness_summary.md`: Human-readable summary
- `reports/tradeoff_[attribute].png`: Bias-accuracy visualizations

### Representation Reports
- `reports/representation_analysis.png`: Diversity visualizations
- `reports/representation_report.json`: Complete analysis data
- `reports/diversity_tracking.json`: Temporal tracking data

### Model Artifacts
- `reports/model.pkl`: Trained scikit-learn model
- `reports/test_set.csv`: Held-out test data

## ⚙️ Configuration

### config.yaml Structure
```yaml
data_path: "data/adult.csv"
target: "income"
positive_label: ">50K"
protected_attributes:
  gender:
    label: "Gender"
    privileged_group: "Male"
  race:
    label: "Race"
    privileged_group: "White"
privileged_groups:
  gender: "Male"
  race: "White"
fairness_thresholds:
  dp_gap_max: 0.1
  eo_gap_max: 0.1
  di_min_ratio: 0.8
test_size: 0.2
random_state: 42
```

### Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key_here
```

## 🔧 Advanced Usage

### Custom Datasets

1. **Upload via UI**: Use the file uploader in the sidebar
2. **Place in data/ folder**: Reference in config.yaml
3. **Format Requirements**:
   - CSV format with headers
   - Binary or categorical target variable
   - Protected attributes as categorical columns

### Extending Protected Attributes

Add new attributes to the configuration:
```python
# In the UI or config.yaml
protected_attributes = "gender,race,age,education"
```

### Custom Fairness Thresholds

Adjust based on your domain requirements:
- **Strict**: dp_gap_max=0.05, eo_gap_max=0.05, di_min_ratio=0.9
- **Moderate**: dp_gap_max=0.1, eo_gap_max=0.1, di_min_ratio=0.8
- **Relaxed**: dp_gap_max=0.2, eo_gap_max=0.2, di_min_ratio=0.7

### Programmatic Usage

```python
from fairness_agent import AlgorithmicFairnessAgent
from Representation_Bias_Agent import RepresentationBiasAgent

# Fairness analysis
agent = AlgorithmicFairnessAgent(data)
results = agent.evaluate_fairness_criteria(y_true, y_pred, thresholds, "gender")

# Representation analysis
rep_agent = RepresentationBiasAgent()
rep_agent.load_dataset("data/your_data.csv", ["gender", "race"])
report = rep_agent.generate_representation_report()
```

## 🧪 Testing

### Run Unit Tests
```bash
python -m pytest tests/
```

### Validate Installation
```bash
python -c "from fairness_agent import AlgorithmicFairnessAgent; print('✅ Installation successful')"
```

### Test with Sample Data
```bash
python scripts/train.py
python scripts/evaluate_fairness.py
```

## 📊 Example Results

### Adult Dataset Analysis
- **Dataset**: 48,842 samples, 14 features
- **Protected Attributes**: Gender, Race
- **Key Findings**:
  - Gender: Moderate bias (Male advantage in income prediction)
  - Race: Significant bias (White advantage, minority under-representation)
  - Diversity: Low (Shannon H' = 0.64 for gender, 0.55 for race)

### Recommendations
- Implement bias mitigation techniques
- Increase data collection for underrepresented groups
- Regular monitoring and reporting
- Stakeholder engagement on fairness definitions

## 🛠️ Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

**GROQ API Issues**
```bash
# Check API key in .env file
cat .env
# Verify API key validity
```

**Streamlit Not Loading**
```bash
# Check port availability
lsof -i :8501
# Try different port
streamlit run app/app_streamlit.py --server.port 8502
```

**Model Training Failures**
```bash
# Check data format
python -c "import pandas as pd; print(pd.read_csv('data/adult.csv').info())"
# Verify target column exists
```

### Performance Optimization

- **Large Datasets**: Use sampling for initial analysis
- **Memory Issues**: Reduce batch sizes in config
- **Speed**: Enable Streamlit caching for repeated analyses

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/Humanitariansai/Popper.git
cd Bias_Detection_Agent
pip install -e .
pre-commit install
```

### Areas for Contribution
- Additional fairness metrics
- New bias detection algorithms
- Enhanced visualizations
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Adult dataset
- **GROQ** for fast LLM inference
- **Streamlit** for the interactive dashboard framework
- **scikit-learn** for machine learning utilities
- **Fairness research community** for metric definitions and best practices

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Humanitariansai/Popper/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Humanitariansai/Popper/discussions)
- **Email**: [humanitarian.sai@gmail.com](mailto:humanitarian.sai@gmail.com)

## 🔄 Version History

### v1.0.0 (Current)
- ✅ Complete fairness analysis pipeline
- ✅ Advanced diversity tracking
- ✅ LLM integration
- ✅ Interactive Streamlit dashboard
- ✅ Comprehensive reporting
- ✅ Population benchmarking

### Roadmap
- 🔮 Additional bias mitigation techniques
- 🔮 Real-time monitoring dashboard
- 🔮 Integration with MLOps pipelines
- 🔮 Extended dataset support
- 🔮 Advanced visualization features

