
# Data Validation Agent

## 📌 Overview

The Data Validation Agent is a Python-based tool designed to automate data validation, quality checks, and model validation reporting. It ensures datasets meet required standards and produces detailed validation reports with visualizations such as confusion matrices, ROC/PR curves, calibration plots, and feature importance analysis.

The agent supports:
- Automated dataset validation with custom rules.
- Model performance evaluation and visualization.
- Robustness and calibration analysis.
- Exportable validation reports (HTML + plots).

## 📂 Project Structure
```
Data_Validation_Agent/
│── main.py                        # Entry point to run the validation agent
│── validators.py                  # Validation logic & data quality checks
│── popper_agents.py               # Core agent logic for processing/validation
│── report_utils.py                # Report generation utilities
│── popper_visualisation.py        # Visualization functions (ROC, PR, etc.)
│── loan_data_set.csv              # Sample dataset for testing
│── datavalidation_popper.ipynb    # Jupyter notebook demo
│── validation_report_outputs/     # Generated reports & plots
│     ├── confusion_matrix.png
│     ├── calibration_curve.png
│     ├── feature_importance.png
│     ├── robustness_curve.png
│     ├── roc_pr_roc.png
│     ├── roc_pr_pr.png
│     └── validation_report.html
```

## ⚙️ Installation

Clone the repository:
```sh
git clone https://github.com/yourusername/Data_Validation_Agent.git
cd Data_Validation_Agent
```

Create a virtual environment & install dependencies:
```sh
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```
(If requirements.txt is missing, generate one using `pip freeze > requirements.txt`.)

## 🚀 Usage

Run the Validation Agent:
```sh
python main.py
```
This will:
- Load the dataset (default: loan_data_set.csv)
- Perform data validation & quality checks
- Generate model validation reports
- Save results in validation_report_outputs/

### Explore with Notebook
You can also open the Jupyter notebook:
```sh
jupyter notebook "datavalidation_popper (5).ipynb"
```

## 📊 Example Outputs
- Confusion Matrix
- ROC & PR Curves
- Calibration Curve
- Feature Importance
- Robustness Curve
- Full HTML Report

All outputs are stored in the `validation_report_outputs/` directory.

## 🛠️ Customization
- Modify `validators.py` to add or change validation rules.
- Replace `loan_data_set.csv` with your own dataset.
- Adjust visualization settings in `popper_visualisation.py`.

