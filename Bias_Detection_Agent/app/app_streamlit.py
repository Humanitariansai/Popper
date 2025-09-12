
import streamlit as st
import pandas as pd
import joblib
import subprocess
import io
import csv
from fairness_agent import AlgorithmicFairnessAgent, FairnessThresholds
from fairness_llm_agent import GroqAgent

# --- Example: LLM Analysis of Current Fairness Metrics (for direct output demonstration) ---
if st.button("Show Example LLM Analysis Output"):
    metrics_text = '''
---
Gender Fairness:
attribute,dp_gap,eo_gap,di_ratio,accuracy,pass_dp,pass_eo,pass_di\ngender,0.1601,0.1435,0.1330,0.7883,False,False,False
---
Race Fairness:
attribute,dp_gap,eo_gap,di_ratio,accuracy,pass_dp,pass_eo,pass_di\nrace,0.1078,0.0846,0.5319,0.7883,False,True,False
---
Intersectional Fairness:\nintersection_group,selection_rate,tpr,count,selection_rate_gap_vs_best,selection_rate_gap_vs_worst,tpr_gap_vs_best,tpr_gap_vs_worst\nMale|Other,0.2727,0.2857,44,0.0,0.2642,0.1143,0.1524\nMale|Asian-Pac-Islander,0.2613,0.3768,199,0.0114,0.2527,0.0232,0.2435\nMale|White,0.1861,0.3516,5697,0.0867,0.1775,0.0484,0.2182\nMale|Amer-Indian-Eskimo,0.1739,0.4,69,0.0988,0.1653,0.0,0.2667\nMale|Black,0.1317,0.3838,501,0.1410,0.1232,0.0162,0.2505\nFemale|Amer-Indian-Eskimo,0.0370,0.3333,27,0.2357,0.0285,0.0667,0.2\nFemale|Asian-Pac-Islander,0.0357,0.3077,112,0.2370,0.0271,0.0923,0.1744\nFemale|White,0.0268,0.2115,2615,0.2460,0.0182,0.1885,0.0782\nFemale|Other,0.0263,0.3333,38,0.2464,0.0178,0.0667,0.2\n'''
    try:
        agent = GroqAgent()
        prompt = f"Analyze the following fairness metrics and provide a summary, highlight any fairness concerns, and suggest possible actions.\n{metrics_text}"
        llm_output = agent.ask(prompt, system_prompt="You are a fairness and data science expert. Be concise and actionable.")
        st.markdown("#### Example LLM Analysis Output")
        st.write(llm_output)
    except Exception as e:
        st.error(f"LLM analysis failed: {e}")

# --- Real-time Fairness Report Generation ---
st.header("Real-time Fairness Report Generation")
if st.button("Generate Latest Fairness Reports (Run Scripts)"):
    with st.spinner("Running fairness evaluation and tradeoff sweep scripts..."):
        try:
            subprocess.run(["python", "scripts/train.py", "--config", "config.yaml"], check=True)
            subprocess.run(["python", "scripts/evaluate_fairness.py", "--config", "config.yaml"], check=True)
            subprocess.run(["python", "scripts/tradeoff_sweep.py", "--config", "config.yaml"], check=True)
            st.success("Reports generated! Scroll down to analyze with LLM.")
        except subprocess.CalledProcessError as e:
            st.error(f"Script failed: {e}")

# --- LLM Analysis of Fairness Metrics ---
st.header("LLM Analysis of Fairness Metrics")
metrics_files = [
    ("Gender Fairness", "reports/criteria_gender.csv"),
    ("Race Fairness", "reports/criteria_race.csv"),
    ("Intersectional Fairness", "reports/intersectional.csv")
]
metrics_text = ""
for label, path in metrics_files:
    try:
        with open(path, "r") as f:
            content = f.read()
        metrics_text += f"\n---\n{label}:\n{content}\n"
    except Exception as e:
        metrics_text += f"\n---\n{label}:\n[Could not load: {e}]\n"

if st.button("Analyze Fairness Metrics with LLM"):
    try:
        agent = GroqAgent()
        prompt = f"Analyze the following fairness metrics and provide a summary, highlight any fairness concerns, and suggest possible actions.\n{metrics_text}"
        llm_output = agent.ask(prompt, system_prompt="You are a fairness and data science expert. Be concise and actionable.")
        st.markdown("#### LLM Analysis Output")
        st.write(llm_output)
    except Exception as e:
        st.error(f"LLM analysis failed: {e}")

st.set_page_config(page_title="Algorithmic Fairness Dashboard", layout="wide")
st.title("Algorithmic Fairness Dashboard")

# --- LLM Agent Chat UI ---
st.sidebar.header("LLM Fairness Agent Chat")
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []
llm_query = st.sidebar.text_area("Ask the Fairness LLM Agent a question:")
if st.sidebar.button("Ask LLM") and llm_query.strip():
    try:
        agent = GroqAgent()
        response = agent.ask(llm_query, system_prompt="You are a helpful fairness and data science assistant. Answer clearly and concisely.")
        st.session_state.llm_history.append((llm_query, response))
    except Exception as e:
        st.session_state.llm_history.append((llm_query, f"[Error: {e}]") )
if st.session_state.llm_history:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### LLM Agent Chat History")
    for q, r in reversed(st.session_state.llm_history[-5:]):
        st.sidebar.markdown(f"**Q:** {q}")
        st.sidebar.markdown(f"**A:** {r}")

st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload CSV (optional; defaults to config data)", type=["csv"])
target_col = st.sidebar.text_input("Target column", value="income")
positive_label = st.sidebar.text_input("Positive label", value=">50K")
prot_attrs = st.sidebar.text_input("Protected attributes (comma-separated)", value="gender,race")
priv_gender = st.sidebar.text_input("Privileged group for gender (optional)", value="Male")
priv_race = st.sidebar.text_input("Privileged group for race (optional)", value="White")

dp_gap_max = st.sidebar.number_input("DP gap max", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
eo_gap_max = st.sidebar.number_input("EO gap max", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
di_min_ratio = st.sidebar.number_input("DI min ratio", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

model_file = st.sidebar.text_input("Model path (joblib/pkl)", value="reports/model.pkl")
run_btn = st.sidebar.button("Run Fairness Analysis")

# Default data from config path if no upload
df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv("data/adult.csv")
    except Exception:
        st.info("No CSV uploaded and default data/adult.csv not found.")
        df = None

if df is not None:
    st.write("Preview", df.head())

if run_btn and df is not None:
    try:
        model = joblib.load(model_file)
    except Exception as e:
        st.error(f"Could not load model at {model_file}: {e}")
        st.stop()

    y_true = (df[target_col] == positive_label).astype(int) if df[target_col].dtype == object else df[target_col]
    X = df.drop(columns=[target_col])

    try:
        scores = model.predict_proba(X)[:,1]
        y_pred = (scores >= 0.5).astype(int)
    except Exception:
        pred = model.predict(X)
        y_pred = (pred == positive_label).astype(int) if getattr(pred, "dtype", None) == object else pred
        scores = y_pred

    prot_list = [s.strip() for s in prot_attrs.split(",") if s.strip()]
    prot_map = {a: (X[a] if a in X.columns else df[a]) for a in prot_list}

    # Build privileged_groups dict dynamically based on prot_list
    priv_groups = {}
    for attr in prot_list:
        if attr == "gender":
            priv_groups["gender"] = priv_gender or None
        elif attr == "race":
            priv_groups["race"] = priv_race or None

    agent = AlgorithmicFairnessAgent(
        favorable_label=1 if y_true.max()==1 else positive_label,
        protected_attributes=prot_map,
        privileged_groups=priv_groups
    )
    thr = FairnessThresholds(dp_gap_max=dp_gap_max, eo_gap_max=eo_gap_max, di_min_ratio=di_min_ratio)

    st.subheader("Results")
    for attr in prot_list:
        st.markdown(f"### Attribute: `{attr}`")
        dp = agent.demographic_parity(y_pred, attr)
        eo = agent.equal_opportunity(y_true, y_pred, attr)
        di = agent.disparate_impact(y_pred, attr)
        ev = agent.evaluate_fairness_criteria(y_true, y_pred, thr, attr)

        st.write("Demographic Parity", dp)
        st.write("Equal Opportunity", eo)
        st.write("Disparate Impact (worst)", di)
        st.write("Criteria Evaluation", ev)

    st.markdown("### Intersectional (all protected attributes)")
    inter = agent.intersectional_bias(y_true, y_pred, prot_list)
    st.write(inter)

    st.success("Done.")
