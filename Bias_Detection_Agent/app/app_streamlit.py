import streamlit as st

# This must be the first Streamlit command
st.set_page_config(page_title="Algorithmic Fairness Dashboard", layout="wide")

import pandas as pd
import joblib
import subprocess
import io
import csv
import sys
import os
import yaml

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness_agent import AlgorithmicFairnessAgent, FairnessThresholds
from fairness_llm_agent import GroqAgent

def update_config_yaml(data_path, target_col, positive_label, prot_attrs, priv_gender, priv_race, dp_gap_max, eo_gap_max, di_min_ratio):
    """Update config.yaml with new parameters"""
    config = {
        "data_path": data_path,
        "target": target_col,
        "positive_label": positive_label,
        "protected_attributes": {},
        "privileged_groups": {},
        "test_size": 0.2,
        "random_state": 42,
        "fairness_thresholds": {
            "dp_gap_max": dp_gap_max,
            "eo_gap_max": eo_gap_max,
            "di_min_ratio": di_min_ratio
        }
    }
    
    # Parse protected attributes - scripts expect simple key-value mapping
    prot_list = [attr.strip() for attr in prot_attrs.split(",") if attr.strip()]
    for attr in prot_list:
        # Scripts expect the value to be the column name
        config["protected_attributes"][attr] = attr
    
    # Set privileged groups
    if "gender" in config["protected_attributes"] and priv_gender:
        config["privileged_groups"]["gender"] = priv_gender
    if "race" in config["protected_attributes"] and priv_race:
        config["privileged_groups"]["race"] = priv_race
    
    # Write to config.yaml
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

st.title("Algorithmic Fairness Dashboard")

# --- Quick Start Guide ---
st.markdown("## 🚀 Quick Start Guide")
st.markdown("""
**Step 1:** 📊 Configure your analysis in the sidebar (upload data, set parameters, adjust thresholds)  
**Step 2:** 🔄 Generate fairness reports by clicking the button in the main area below  
**Step 3:** 🤖 Analyze results with LLM for insights and recommendations  
**Step 4:** 🌍 Assess representation bias and demographic diversity
""")
st.markdown("---")

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

# Button to update config with current parameters
if st.sidebar.button("Update Config with Current Parameters"):
    try:
        # Use existing data path from config or default
        try:
            with open("config.yaml", "r") as f:
                existing_config = yaml.safe_load(f)
            current_data_path = existing_config.get("data_path", "data/adult.csv")
        except:
            current_data_path = "data/adult.csv"
        
        updated_config = update_config_yaml(
            data_path=current_data_path,
            target_col=target_col,
            positive_label=positive_label,
            prot_attrs=prot_attrs,
            priv_gender=priv_gender,
            priv_race=priv_race,
            dp_gap_max=dp_gap_max,
            eo_gap_max=eo_gap_max,
            di_min_ratio=di_min_ratio
        )
        st.sidebar.success("✅ Config updated!")
    except Exception as e:
        st.sidebar.error(f"Failed to update config: {e}")

run_btn = st.sidebar.button("Run Fairness Analysis")
retrain_btn = st.sidebar.button("🔄 Train New Model")

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
    st.sidebar.markdown("---")

# Default data from config path if no upload
df = None
uploaded_data_path = None

if uploaded is not None:
    # Save uploaded file to data directory
    os.makedirs("data", exist_ok=True)
    uploaded_data_path = f"data/{uploaded.name}"
    
    # Save the uploaded file
    with open(uploaded_data_path, "wb") as f:
        f.write(uploaded.getbuffer())
    
    df = pd.read_csv(uploaded_data_path)
    st.success(f"✅ Uploaded dataset saved to: {uploaded_data_path}")
    
    # Update config.yaml with the new dataset and parameters
    try:
        updated_config = update_config_yaml(
            data_path=uploaded_data_path,
            target_col=target_col,
            positive_label=positive_label,
            prot_attrs=prot_attrs,
            priv_gender=priv_gender,
            priv_race=priv_race,
            dp_gap_max=dp_gap_max,
            eo_gap_max=eo_gap_max,
            di_min_ratio=di_min_ratio
        )
        st.success("✅ Config.yaml updated with new dataset and parameters!")
        
        # Show updated config
        with st.expander("View Updated Config"):
            st.code(yaml.dump(updated_config, default_flow_style=False), language="yaml")
            
    except Exception as e:
        st.error(f"Failed to update config.yaml: {e}")
        
else:
    # Try to load from existing config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        data_path = config.get("data_path", "data/adult.csv")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.info(f"📊 Using dataset from config: {data_path}")
        else:
            st.warning(f"Dataset not found at {data_path}. Please upload a dataset.")
    except Exception as e:
        try:
            # Try default dataset
            df = pd.read_csv("data/adult.csv")
            st.info("📊 Using default dataset: data/adult.csv")
        except Exception as e2:
            st.error(f"Could not read config: {e}")

# --- Step 2: Generate Fairness Reports ---
st.markdown("### 📊 Step 2: Generate Fairness Reports")
st.info("🚀 **Click the button below to generate fresh fairness reports with your current configuration**")
st.header("Real-time Fairness Report Generation")
if st.button("🚀 Generate Latest Fairness Reports (Run Scripts)"):
    with st.spinner("Updating configuration and running scripts..."):
        try:
            # ALWAYS update config with current UI parameters before running scripts
            try:
                if uploaded is not None:
                    current_data_path = uploaded_data_path
                else:
                    try:
                        with open("config.yaml", "r") as f:
                            existing_config = yaml.safe_load(f)
                        current_data_path = existing_config.get("data_path", "data/adult.csv")
                    except:
                        current_data_path = "data/adult.csv"
                
                updated_config = update_config_yaml(
                    data_path=current_data_path,
                    target_col=target_col,
                    positive_label=positive_label,
                    prot_attrs=prot_attrs,
                    priv_gender=priv_gender,
                    priv_race=priv_race,
                    dp_gap_max=dp_gap_max,
                    eo_gap_max=eo_gap_max,
                    di_min_ratio=di_min_ratio
                )
                st.success("✅ Configuration updated with current parameters!")
                
                # Run the scripts in sequence
                st.write("Running train.py...")
                result_train = subprocess.run([sys.executable, "scripts/train.py", "--config", "config.yaml"], 
                                            capture_output=True, text=True, cwd=".")
                if result_train.returncode == 0:
                    st.success("✅ Model training completed!")
                    st.text(result_train.stdout[-500:])  # Show last 500 chars
                else:
                    st.error(f"❌ Training failed: {result_train.stderr}")
                    st.text(result_train.stdout)
                
                st.write("Running evaluate_fairness.py...")
                result_eval = subprocess.run([sys.executable, "scripts/evaluate_fairness.py", "--config", "config.yaml"], 
                                           capture_output=True, text=True, cwd=".")
                if result_eval.returncode == 0:
                    st.success("✅ Fairness evaluation completed!")
                    st.text(result_eval.stdout[-500:])  # Show last 500 chars
                else:
                    st.error(f"❌ Evaluation failed: {result_eval.stderr}")
                    st.text(result_eval.stdout)
                
                st.write("Running tradeoff_sweep.py...")
                result_tradeoff = subprocess.run([sys.executable, "scripts/tradeoff_sweep.py", "--config", "config.yaml"], 
                                               capture_output=True, text=True, cwd=".")
                if result_tradeoff.returncode == 0:
                    st.success("✅ Tradeoff analysis completed!")
                    st.text(result_tradeoff.stdout[-500:])  # Show last 500 chars
                else:
                    st.error(f"❌ Tradeoff analysis failed: {result_tradeoff.stderr}")
                    st.text(result_tradeoff.stdout)
                
            except Exception as config_error:
                st.error(f"Failed to update configuration: {config_error}")
                
            # Show dynamic fairness verdict
            st.markdown("#### 🎯 Fairness Verdict")
            try:
                # Read current config to get protected attributes
                try:
                    with open("config.yaml", "r") as f:
                        current_config = yaml.safe_load(f)
                    protected_attrs = current_config.get("protected_attributes", {})
                    
                    # Define emoji mapping for common attributes
                    attr_emojis = {
                        "gender": "👥",
                        "race": "🌍", 
                        "age": "📅",
                        "relationship": "💑",
                        "education": "🎓",
                        "workclass": "💼",
                        "occupation": "👷",
                        "marital_status": "💒",
                        "native_country": "🌎"
                    }
                    
                    if not protected_attrs:
                        st.warning("No protected attributes found in configuration.")
                    else:
                        for attr_name, attr_config in protected_attrs.items():
                            # Handle both old format (dict) and new format (string)
                            if isinstance(attr_config, dict):
                                attr_label = attr_config.get("label", attr_name)
                            else:
                                # New simplified format where attr_config is just the column name
                                attr_label = attr_name.replace('_', ' ').title()
                            
                            attr_emoji = attr_emojis.get(attr_name, "📊")
                            
                            # Try to read the criteria file for this attribute
                            try:
                                criteria_file = f"reports/criteria_{attr_name}.csv"
                                criteria_df = pd.read_csv(criteria_file)
                                
                                st.markdown(f"### {attr_emoji} {attr_label} Fairness Assessment")
                                
                                # Check if DataFrame has the expected structure
                                if not criteria_df.empty and len(criteria_df) > 0:
                                    # Get the first row (should contain the summary)
                                    row = criteria_df.iloc[0]
                                    
                                    # Display individual criteria with status
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        dp_pass = row.get('pass_dp', False)
                                        dp_icon = "✅" if dp_pass else "❌"
                                        st.markdown(f"{dp_icon} **Demographic Parity:** {'PASS' if dp_pass else 'FAIL'}")
                                        if 'dp_gap' in row:
                                            st.write(f"Gap: {row['dp_gap']:.3f}")
                                    
                                    with col2:
                                        eo_pass = row.get('pass_eo', False)
                                        eo_icon = "✅" if eo_pass else "❌"
                                        st.markdown(f"{eo_icon} **Equal Opportunity:** {'PASS' if eo_pass else 'FAIL'}")
                                        if 'eo_gap' in row:
                                            st.write(f"Gap: {row['eo_gap']:.3f}")
                                    
                                    with col3:
                                        di_pass = row.get('pass_di', False)
                                        di_icon = "✅" if di_pass else "❌"
                                        st.markdown(f"{di_icon} **Disparate Impact:** {'PASS' if di_pass else 'FAIL'}")
                                        if 'di_ratio' in row:
                                            st.write(f"Ratio: {row['di_ratio']:.3f}")
                                    
                                    # Overall verdict
                                    passes = row.get('pass_dp', 0) + row.get('pass_eo', 0) + row.get('pass_di', 0)
                                    attr_display_name = attr_name.replace('_', ' ').title()
                                    
                                    if passes == 3:
                                        st.success(f"🎉 **{attr_display_name} PASSES all fairness criteria!**")
                                    elif passes >= 2:
                                        st.warning(f"⚠️ **{attr_display_name} passes {passes}/3 fairness criteria**")
                                    else:
                                        st.error(f"🚨 **{attr_display_name} FAILS fairness assessment ({passes}/3 passed)**")
                                    
                                    st.markdown("---")
                                    
                            except FileNotFoundError:
                                st.warning(f"📋 Criteria file not found for {attr_label}. Make sure the scripts completed successfully.")
                            except Exception as e:
                                st.error(f"Could not load {attr_label} criteria: {e}")
                                
                except Exception as e:
                    st.error(f"Could not display fairness verdict: {e}")
                    
            except Exception as e:
                st.error(f"Could not display fairness verdict: {e}")
                
        except subprocess.CalledProcessError as e:
            st.error(f"Script failed: {e}")

# Manual model retraining
if retrain_btn and df is not None:
    st.info("🔄 Training a new model...")
    
    try:
        # Import required modules
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        
        # Prepare data
        y = (df[target_col] == positive_label).astype(int) if df[target_col].dtype == object else df[target_col]
        X = df.drop(columns=[target_col])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline
        cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        
        preproc = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )
        
        model = Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=200))])
        
        with st.spinner("Training model..."):
            model.fit(X_train, y_train)
        
        # Evaluate model
        try:
            proba = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, proba)
            st.success(f"✅ Model trained! Validation ROC AUC: {auc:.4f}")
        except Exception as e:
            st.warning(f"Model trained but AUC not computed: {e}")
            st.success("✅ Model trained successfully!")
        
        # Save the new model
        os.makedirs("reports", exist_ok=True)
        joblib.dump(model, model_file)
        st.success(f"✅ Model saved to {model_file}")
        
        # Also save test set
        X_test.assign(**{target_col: y_test}).to_csv("reports/test_set.csv", index=False)
        st.info("📊 Test set saved to reports/test_set.csv")
        
    except Exception as train_error:
        st.error(f"Failed to train model: {train_error}")

elif retrain_btn and df is None:
    st.error("Please upload a dataset first before training a model.")

# --- Step 3: LLM Analysis ---
st.markdown("### 🤖 Step 3: LLM Analysis")
st.info("📊 **After generating reports, use AI to analyze fairness metrics and get recommendations**")
st.header("LLM Analysis of Fairness Metrics")

# Dynamically build metrics files based on current config
try:
    with open("config.yaml", "r") as f:
        current_config = yaml.safe_load(f)
    protected_attrs = current_config.get("protected_attributes", {})
    
    # Build dynamic list of criteria files
    metrics_files = []
    for attr_name in protected_attrs.keys():
        attr_display = attr_name.replace('_', ' ').title()
        criteria_file = f"reports/criteria_{attr_name}.csv"
        metrics_files.append((f"{attr_display} Fairness", criteria_file))
    
    # Always include intersectional analysis
    metrics_files.append(("Intersectional Fairness", "reports/intersectional.csv"))
    
except Exception as e:
    st.warning(f"Could not read config for LLM analysis: {e}")
    # Fallback to default files
    metrics_files = [
        ("Gender Fairness", "reports/criteria_gender.csv"),
        ("Race Fairness", "reports/criteria_race.csv"),
        ("Intersectional Fairness", "reports/intersectional.csv")
    ]

# Show what files will be analyzed
with st.expander("📋 Files to be analyzed by LLM"):
    for label, path in metrics_files:
        st.write(f"• {label}: `{path}`")

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
        
        # Enhanced prompt with current configuration context
        thresholds = current_config.get('fairness_thresholds', {})
        dp_threshold = thresholds.get('dp_gap_max', 0.1)
        eo_threshold = thresholds.get('eo_gap_max', 0.1)
        di_threshold = thresholds.get('di_min_ratio', 0.8)
        
        config_context = f"""
Current Configuration:
- Protected Attributes: {list(protected_attrs.keys()) if 'protected_attrs' in locals() else 'Unknown'}
- Target: {current_config.get('target', 'Unknown') if 'current_config' in locals() else 'Unknown'}
- Positive Label: {current_config.get('positive_label', 'Unknown') if 'current_config' in locals() else 'Unknown'}
- Thresholds: DP≤{dp_threshold}, EO≤{eo_threshold}, DI≥{di_threshold}
"""
        
        prompt = f"""Analyze the following fairness metrics and provide a comprehensive assessment.

{config_context}

Fairness Metrics Data:
{metrics_text}

Please provide:
1. Summary of key fairness issues found
2. Which protected attributes have the most concerning bias
3. Specific recommendations for improvement
4. Assessment of whether current thresholds are appropriate
5. Priority actions to take next
"""
        
        llm_output = agent.ask(prompt, system_prompt="You are a fairness and data science expert. Provide detailed, actionable analysis of algorithmic bias. Be specific about which groups are affected and suggest concrete mitigation strategies.")
        st.markdown("#### LLM Analysis Output")
        st.write(llm_output)
    except Exception as e:
        st.error(f"LLM analysis failed: {e}")

st.markdown("---")

# --- Step 4: Representation Bias Analysis ---
st.markdown("### 🌍 Step 4: Representation Bias Analysis")
st.info("🔍 **Assess whether your dataset reflects diverse populations and identify representation gaps**")
st.header("Dataset Representation Analysis")

# Check if we have data to analyze
representation_df = None
if uploaded is not None and df is not None:
    representation_df = df
elif df is not None:
    representation_df = df
else:
    # Try to load default data
    try:
        representation_df = pd.read_csv("data/adult.csv")
        st.info("📊 Using default adult.csv dataset for representation analysis")
    except Exception as e:
        st.warning(f"Could not load dataset for representation analysis: {e}")

if representation_df is not None:
    if st.button("🌍 Analyze Dataset Representation"):
        with st.spinner("Analyzing dataset representation and diversity..."):
            try:
                # Import and initialize the Representation Bias Agent
                import sys
                sys.path.append(".")
                from Representation_Bias_Agent import RepresentationBiasAgent
                
                # Initialize agent
                rep_agent = RepresentationBiasAgent()
                
                # Load dataset
                if uploaded is not None:
                    data_path = f"data/{uploaded.name}"
                else:
                    data_path = "data/adult.csv"
                
                # Get protected attributes from current config
                try:
                    with open("config.yaml", "r") as f:
                        current_config = yaml.safe_load(f)
                    protected_attrs = list(current_config.get("protected_attributes", {}).keys())
                    if not protected_attrs:
                        protected_attrs = ["gender", "race"]  # Default fallback
                except:
                    protected_attrs = ["gender", "race"]  # Default fallback
                
                success = rep_agent.load_dataset(data_path, protected_attrs)
                
                if success:
                    # Generate comprehensive report
                    report = rep_agent.generate_representation_report()
                    
                    # Display results
                    st.markdown("#### 📊 Representation Analysis Results")
                    
                    # Dataset Overview
                    with st.expander("📋 Dataset Overview", expanded=True):
                        dataset_info = report['dataset_info']
                        st.write(f"**Total Samples:** {dataset_info['total_samples']:,}")
                        st.write(f"**Analyzed Attributes:** {', '.join(dataset_info['attributes_analyzed'])}")
                        st.write(f"**Analysis Date:** {dataset_info['analysis_timestamp'][:19]}")
                    
                    # Overall Assessment
                    with st.expander("🎯 Overall Assessment", expanded=True):
                        assessment = report['overall_assessment']
                        
                        # Display bias level with appropriate color
                        bias_level = assessment['bias_level']
                        combined_score = assessment['combined_bias_score']
                        
                        if "Low" in bias_level:
                            st.success(f"🎉 **{bias_level}** (Score: {combined_score:.3f})")
                        elif "Moderate" in bias_level:
                            st.warning(f"⚠️ **{bias_level}** (Score: {combined_score:.3f})")
                        else:
                            st.error(f"🚨 **{bias_level}** (Score: {combined_score:.3f})")
                        
                        # Individual scores
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Diversity Score", f"{assessment['overall_diversity_score']:.3f}")
                        with col2:
                            st.metric("Representation Score", f"{assessment['overall_representation_score']:.3f}")
                        with col3:
                            st.metric("Inclusion Score", f"{assessment['overall_inclusion_score']:.3f}")
                    
                    # Key Concerns
                    if assessment['key_concerns']:
                        with st.expander("⚠️ Key Concerns", expanded=True):
                            for concern in assessment['key_concerns']:
                                st.write(f"• {concern}")
                    
                    # Demographic Distribution
                    with st.expander("📊 Demographic Distribution"):
                        demo_dist = report['demographic_distribution']
                        for attr, attr_data in demo_dist.items():
                            st.markdown(f"**{attr.title()}:**")
                            
                            # Show counts and proportions
                            counts_df = pd.DataFrame([
                                {"Group": group, "Count": count, "Proportion": f"{attr_data['proportions'][group]:.1%}"}
                                for group, count in attr_data['counts'].items()
                            ])
                            st.dataframe(counts_df, hide_index=True)
                            
                            # Show dominance ratio
                            if attr_data['dominance_ratio'] > 2.0:
                                st.warning(f"⚠️ High dominance ratio: {attr_data['dominance_ratio']:.1f}x")
                            
                            st.markdown("---")
                    
                    # Representation Gaps (if benchmarks available)
                    representation_gaps = report['representation_gaps']
                    if representation_gaps:
                        with st.expander("📈 Representation Gaps vs. Population Benchmarks"):
                            for attr, attr_gaps in representation_gaps.items():
                                st.markdown(f"**{attr.title()} Representation:**")
                                
                                gaps_data = []
                                for group, group_data in attr_gaps.items():
                                    gaps_data.append({
                                        "Group": group,
                                        "Observed": f"{group_data['observed_proportion']:.1%}",
                                        "Expected": f"{group_data['expected_proportion']:.1%}",
                                        "Ratio": f"{group_data['representation_ratio']:.2f}",
                                        "Status": group_data['status']
                                    })
                                
                                gaps_df = pd.DataFrame(gaps_data)
                                st.dataframe(gaps_df, hide_index=True)
                                
                                # Highlight severely under/over-represented groups
                                severe_issues = [
                                    row for row in gaps_data 
                                    if "Severely" in row['Status'] or "Over-represented" in row['Status']
                                ]
                                if severe_issues:
                                    st.warning("⚠️ Groups with severe representation issues:")
                                    for issue in severe_issues:
                                        st.write(f"  • {issue['Group']}: {issue['Status']}")
                                
                                st.markdown("---")
                    
                    # Inclusion Gaps Analysis
                    inclusion_gaps = report['inclusion_gaps']
                    if inclusion_gaps:
                        with st.expander("🔍 Inclusion Gaps Analysis"):
                            for attr, attr_gaps in inclusion_gaps.items():
                                st.markdown(f"**{attr.title()} Inclusion Analysis:**")
                                
                                # Basic inclusion metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Coverage Ratio", f"{attr_gaps['coverage_ratio']:.1%}")
                                with col2:
                                    st.metric("Inclusion Score", f"{attr_gaps['inclusion_score']:.3f}")
                                with col3:
                                    st.metric("Groups Found", f"{len(attr_gaps['observed_groups'])}")
                                
                                # Missing groups (most important)
                                if attr_gaps['missing_groups']:
                                    st.error("🚨 **Missing Demographic Groups:**")
                                    for missing_group in attr_gaps['missing_groups']:
                                        st.write(f"• {missing_group} - Not represented in dataset")
                                else:
                                    st.success("✅ All expected demographic groups are present")
                                
                                # Unexpected groups (groups in data but not in benchmark)
                                if attr_gaps['unexpected_groups']:
                                    st.info("ℹ️ **Additional Groups Found:**")
                                    for unexpected_group in attr_gaps['unexpected_groups']:
                                        st.write(f"• {unexpected_group} - Present in dataset but not in population benchmark")
                                
                                # Show all groups comparison
                                st.markdown(f"**📋 Detailed {attr.title()} Group Analysis:**")
                                groups_df = pd.DataFrame([
                                    {
                                        "Group": group,
                                        "In Dataset": "✅" if group in attr_gaps['observed_groups'] else "❌",
                                        "In Benchmark": "✅" if group in attr_gaps['expected_groups'] else "❌",
                                        "Status": "Present" if group in attr_gaps['observed_groups'] else "Missing"
                                    }
                                    for group in set(attr_gaps['observed_groups'] + attr_gaps['expected_groups'])
                                ])
                                st.dataframe(groups_df, hide_index=True)
                                
                                st.markdown("---")
                    
                    # Diversity Metrics
                    with st.expander("🎲 Advanced Diversity Metrics & Tracking"):
                        diversity_metrics = report['diversity_metrics']
                        for attr, metrics in diversity_metrics.items():
                            st.markdown(f"**{attr.title()} Comprehensive Diversity Analysis:**")
                            
                            # Diversity Level Classification
                            diversity_level = metrics.get('diversity_level', 'Unknown')
                            if 'Very High' in diversity_level:
                                st.success(f"🌟 **{diversity_level}**")
                            elif 'High' in diversity_level:
                                st.success(f"✅ **{diversity_level}**")
                            elif 'Moderate' in diversity_level:
                                st.warning(f"⚠️ **{diversity_level}**")
                            else:
                                st.error(f"🚨 **{diversity_level}**")
                            
                            # Core Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Overall Diversity Score", f"{metrics['diversity_score']:.3f}")
                                st.metric("Shannon Diversity (H')", f"{metrics['shannon_diversity']:.3f}")
                                st.metric("Simpson Diversity", f"{metrics['simpson_diversity']:.3f}")
                            with col2:
                                st.metric("Pielou's Evenness (J')", f"{metrics['evenness_pielou']:.3f}")
                                st.metric("Shannon Evenness", f"{metrics['evenness_shannon']:.3f}")
                                st.metric("Berger-Parker Dominance", f"{metrics['berger_parker_dominance']:.3f}")
                            with col3:
                                st.metric("Total Groups", metrics['total_groups'])
                                st.metric("Effective Groups", f"{metrics['effective_groups']:.1f}")
                                st.metric("Minority Groups (<10%)", metrics['minority_groups_count'])
                            
                            # Advanced Metrics
                            st.markdown(f"**📈 Advanced {attr.title()} Metrics:**")
                            adv_col1, adv_col2 = st.columns(2)
                            with adv_col1:
                                st.metric("Hill 0 (Richness)", f"{metrics['hill_0_richness']}")
                                st.metric("Hill 1 (Shannon exp)", f"{metrics['hill_1_shannon_exp']:.2f}")
                                st.metric("Hill 2 (Simpson inv)", f"{metrics['hill_2_simpson_inv']:.2f}")
                                st.metric("Gini-Simpson", f"{metrics['gini_simpson']:.3f}")
                            with adv_col2:
                                st.metric("Inverse Simpson", f"{metrics['inverse_simpson']:.2f}")
                                st.metric("Rényi-2 Diversity", f"{metrics['renyi_2_diversity']:.3f}")
                                st.metric("Theil Index", f"{metrics['theil_diversity_index']:.3f}")
                                st.metric("Dominant Group %", f"{metrics['dominant_group_proportion']:.1%}")
                            
                            # Diversity Tracking Over Time
                            try:
                                temporal_tracking = rep_agent.track_diversity_over_time()
                                if attr in temporal_tracking:
                                    track_data = temporal_tracking[attr]
                                    st.markdown(f"**📈 {attr.title()} Diversity Trends:**")
                                    
                                    trend_col1, trend_col2, trend_col3 = st.columns(3)
                                    with trend_col1:
                                        trend_icon = "📈" if track_data['shannon_trend'] == 'Increasing' else "📉" if track_data['shannon_trend'] == 'Decreasing' else "➡️"
                                        st.metric("Trend Direction", f"{trend_icon} {track_data['shannon_trend']}")
                                    with trend_col2:
                                        st.metric("Diversity Volatility", f"{track_data['diversity_volatility']:.3f}")
                                    with trend_col3:
                                        st.metric("Trend Strength", f"{track_data['trend_strength']:.3f}")
                                    
                                    # Show temporal progression
                                    temporal_df = pd.DataFrame(track_data['temporal_metrics'])
                                    st.line_chart(temporal_df.set_index('period')[['shannon_diversity', 'simpson_diversity']])
                                    
                            except Exception as e:
                                st.info("💡 Temporal tracking available in detailed reports")
                            
                            st.markdown("---")
                    
                    # Diversity Tracking Report
                    with st.expander("📊 Generate Detailed Diversity Tracking Report"):
                        if st.button(f"📈 Generate Comprehensive Diversity Tracking Report"):
                            try:
                                success = rep_agent.save_diversity_tracking_report("reports/diversity_tracking_detailed.json")
                                if success:
                                    st.success("✅ Detailed diversity tracking report saved!")
                                    
                                    # Show preview of recommendations
                                    basic_metrics = rep_agent.calculate_diversity_metrics()
                                    temporal_tracking = rep_agent.track_diversity_over_time()
                                    recommendations = rep_agent._generate_diversity_recommendations(basic_metrics, temporal_tracking)
                                    
                                    st.markdown("**🎯 Key Diversity Recommendations:**")
                                    for i, rec in enumerate(recommendations[:5], 1):
                                        level = rec.split(':')[0] if ':' in rec else 'INFO'
                                        message = rec.split(':', 1)[1].strip() if ':' in rec else rec
                                        
                                        if level == 'CRITICAL':
                                            st.error(f"{i}. {message}")
                                        elif level == 'WARNING':
                                            st.warning(f"{i}. {message}")
                                        elif level == 'IMPROVE' or level == 'ENHANCE':
                                            st.info(f"{i}. {message}")
                                        else:
                                            st.write(f"{i}. {message}")
                                    
                                    st.success("📁 Full report saved to `reports/diversity_tracking_detailed.json`")
                                else:
                                    st.error("❌ Failed to save diversity tracking report")
                            except Exception as e:
                                st.error(f"Error generating report: {e}")
                    
                    # Historical Context
                    with st.expander("📚 Historical Context"):
                        historical = report['historical_context']
                        st.markdown(f"**Dataset Era:** {historical['dataset_era']}")
                        
                        st.markdown("**Historical Notes:**")
                        for category, note in historical['historical_notes'].items():
                            st.write(f"• **{category.title()}:** {note}")
                        
                        st.markdown("**Modern Relevance:**")
                        st.markdown("*Strengths:*")
                        for strength in historical['modern_relevance']['strengths']:
                            st.write(f"• {strength}")
                        
                        st.markdown("*Limitations:*")
                        for limitation in historical['modern_relevance']['limitations']:
                            st.write(f"• {limitation}")
                    
                    # Priority Actions
                    with st.expander("🎯 Priority Actions", expanded=True):
                        priority_actions = assessment['priority_actions']
                        st.markdown("**Recommended actions to improve representation:**")
                        for i, action in enumerate(priority_actions[:8], 1):
                            st.write(f"{i}. {action}")
                    
                    # Generate and save visualization
                    try:
                        import os
                        os.makedirs("reports", exist_ok=True)
                        
                        fig = rep_agent.visualize_representation("reports/representation_analysis.png")
                        st.markdown("#### 📊 Representation Visualizations")
                        st.pyplot(fig)
                        st.success("📁 Visualization saved to `reports/representation_analysis.png`")
                        
                        # Save detailed report
                        import json
                        with open("reports/representation_report.json", "w") as f:
                            json.dump(report, f, indent=2, default=str)
                        st.success("📁 Detailed report saved to `reports/representation_report.json`")
                        
                    except Exception as viz_error:
                        st.warning(f"Could not generate visualization: {viz_error}")
                    
                    st.success("✅ Representation bias analysis completed!")
                    
                else:
                    st.error("Failed to load dataset for representation analysis")
                    
            except Exception as e:
                st.error(f"Representation analysis failed: {e}")
                import traceback
                st.error(f"Details: {traceback.format_exc()}")
else:
    st.warning("⚠️ Please upload a dataset or ensure data is available for representation analysis")

st.markdown("---")

# --- Real-time Manual Analysis ---
st.header("📊 Real-time Fairness Analysis")
st.info("This section provides immediate fairness analysis using the current model and data.")

if run_btn and df is not None:
    model = None
    
    # Try to load existing model
    try:
        model = joblib.load(model_file)
        st.success(f"✅ Loaded existing model from {model_file}")
    except Exception as e:
        st.warning(f"Could not load model at {model_file}: {e}")
        
        # Offer to train a new model
        st.info("🔄 Training a new model with current data...")
        
        try:
            # Train a new model inline
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import LogisticRegression
            
            # Prepare data
            y = (df[target_col] == positive_label).astype(int) if df[target_col].dtype == object else df[target_col]
            X = df.drop(columns=[target_col])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create pipeline
            cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
            num_cols = [c for c in X_train.columns if c not in cat_cols]
            
            preproc = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ]
            )
            
            model = Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=200))])
            model.fit(X_train, y_train)
            
            # Save the new model
            os.makedirs("reports", exist_ok=True)
            joblib.dump(model, model_file)
            
            st.success(f"✅ New model trained and saved to {model_file}")
            
        except Exception as train_error:
            st.error(f"Failed to train new model: {train_error}")
            st.stop()
    
    if model is None:
        st.error("No model available for analysis")
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

    prot_list = [attr.strip() for attr in prot_attrs.split(",") if attr.strip()]
    
    # Create protected attributes dictionary from the dataframe
    protected_attributes = {}
    for attr in prot_list:
        if attr in df.columns:
            # Convert to list to ensure compatibility with the agent
            protected_attributes[attr] = df[attr].tolist()
        else:
            st.error(f"Protected attribute '{attr}' not found in dataset columns: {list(df.columns)}")
            st.stop()
    
    agent = AlgorithmicFairnessAgent(favorable_label=positive_label, protected_attributes=protected_attributes)
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
