import argparse, yaml, os, sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness_agent import AlgorithmicFairnessAgent, FairnessThresholds

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    cfg = load_config(args.config)
    test_df = pd.read_csv("reports/test_set.csv")
    
    # Try to load existing model, handle compatibility issues
    try:
        model = joblib.load("reports/model.pkl")
        print("✅ Loaded existing model")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        print("🔄 Training a new model...")
        
        # Train a new model if loading fails
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        
        # Load full dataset for training
        data = pd.read_csv(cfg["data_path"])
        target = cfg["target"]
        y = data[target]
        X = data.drop(columns=[target])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.get("test_size", 0.2), 
            random_state=cfg.get("random_state", 42), stratify=y
        )
        
        cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        
        preproc = ColumnTransformer(
            transformers=[
                ("num","passthrough", num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )
        
        model = Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=200))])
        model.fit(X_train, y_train)
        
        # Save the new model and test set
        joblib.dump(model, "reports/model.pkl")
        test_df = X_test.assign(**{target: y_test})
        test_df.to_csv("reports/test_set.csv", index=False)
        print("✅ New model trained and saved")

    target = cfg["target"]
    pos_label = cfg["positive_label"]
    y_true = (test_df[target] == pos_label).astype(int) if test_df[target].dtype == object else test_df[target]
    X = test_df.drop(columns=[target])

    try:
        scores = model.predict_proba(X)[:,1]
    except Exception:
        try:
            from sklearn.utils import column_or_1d
            scores = column_or_1d(model.decision_function(X))
        except Exception:
            pred = model.predict(X)
            scores = (pred == pos_label).astype(int) if getattr(pred, "dtype", None) == object else pred

    prot_map = {k:(X[v] if v in X.columns else test_df[v]) for k,v in cfg["protected_attributes"].items()}
    agent = AlgorithmicFairnessAgent(
        favorable_label=1 if y_true.max()==1 else pos_label,
        protected_attributes=prot_map,
        privileged_groups=cfg.get("privileged_groups", {})
    )

    ft_cfg = cfg.get("fairness_thresholds", {})
    thresholds = FairnessThresholds(
        dp_gap_max=ft_cfg.get("dp_gap_max", 0.1),
        eo_gap_max=ft_cfg.get("eo_gap_max", 0.1),
        di_min_ratio=ft_cfg.get("di_min_ratio", 0.8),
    )

    os.makedirs("reports", exist_ok=True)
    for attr in prot_map.keys():
        sweep = agent.assess_tradeoffs(scores, y_true, attribute=attr, thresholds=thresholds)
        out_csv = f"reports/tradeoff_{attr}.csv"
        sweep.to_csv(out_csv, index=False)

        plt.figure()
        plt.plot(sweep["threshold"], sweep["accuracy"], label="accuracy")
        plt.plot(sweep["threshold"], sweep["di_ratio"], label="di_ratio (worst)")
        plt.axhline(thresholds.di_min_ratio, linestyle="--", label="di_min_ratio")
        plt.xlabel("threshold")
        plt.ylabel("metric value")
        plt.title(f"Trade-off sweep for {attr}")
        plt.legend()
        out_png = f"reports/tradeoff_{attr}.png"
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()

        print(f"Wrote {out_csv} and {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args())
