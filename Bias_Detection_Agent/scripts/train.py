import argparse, yaml, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    cfg = load_config(args.config)
    data = pd.read_csv(cfg["data_path"])
    target = cfg["target"]
    y = data[target]
    X = data.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("test_size", 0.2), random_state=cfg.get("random_state", 42), stratify=y
    )

    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("num","passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=200))])
    clf.fit(X_train, y_train)

    try:
        proba = clf.predict_proba(X_test)[:,1]
        if y.dtype == object:
            auc = roc_auc_score((y_test==cfg["positive_label"]).astype(int), proba)
        else:
            auc = roc_auc_score(y_test, proba)
        print(f"Validation ROC AUC: {auc:.4f}")
    except Exception as e:
        print(f"AUC not computed: {e}")

    os.makedirs("reports", exist_ok=True)
    joblib.dump(clf, "reports/model.pkl")
    X_test.assign(**{target: y_test}).to_csv("reports/test_set.csv", index=False)
    print("Saved model to reports/model.pkl and test set to reports/test_set.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args())
