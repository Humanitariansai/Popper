import argparse, yaml, os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fairness_agent import AlgorithmicFairnessAgent, FairnessThresholds

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    cfg = load_config(args.config)
    test_df = pd.read_csv("reports/test_set.csv")
    model = joblib.load("reports/model.pkl")

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
