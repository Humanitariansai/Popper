# Bias Detection Agents — Algorithmic Fairness Project (with CSV)

This project evaluates fairness using Demographic Parity, Equal Opportunity, Disparate Impact (80% rule), and Intersectional Bias, plus decision-threshold trade‑off sweeps. **Includes `data/adult.csv`**.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train
python scripts/train.py --config config.yaml

# Fairness evaluation
python scripts/evaluate_fairness.py --config config.yaml

# Threshold trade-offs
python scripts/tradeoff_sweep.py --config config.yaml

# Optional dashboard
streamlit run app/app_streamlit.py
```
Artifacts in `reports/`: model, metrics CSVs, summary, and sweep plots.
