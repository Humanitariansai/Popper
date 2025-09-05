def plot_fairness_bars(Xte, yte, yhat, yscore, sensitive_cols, outdir=None, prefix="fairness"):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    if not sensitive_cols:
        print("No sensitive columns for fairness plots.")
        return []
    img_paths = []
    for col in sensitive_cols:
        if col not in Xte.columns:
            continue
        cats = pd.Series(Xte[col]).astype("category")
        groups = list(cats.cat.categories)
        prates, tprs = [], []
        for g in groups:
            idx = (cats==g)
            prates.append(float(np.mean(yhat[idx]==1)) if idx.sum()>0 else np.nan)
            pos = np.sum(yte[idx]==1)
            tprs.append(float(np.sum((yte[idx]==1)&(yhat[idx]==1))/pos) if pos>0 else np.nan)
        # Positive Rate
        fig, ax = plt.subplots()
        ax.bar(range(len(groups)), [0 if np.isnan(v) else v for v in prates])
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_title(f"Positive Rate by {col}")
        plt.tight_layout()
        img1 = None
        if outdir:
            img1 = os.path.join(outdir, f"{prefix}_prate_{col}.png")
            plt.savefig(img1, bbox_inches='tight')
            plt.close()
            img_paths.append((f"Positive Rate by {col}", img1))
        else:
            plt.show()
        # TPR
        fig, ax = plt.subplots()
        ax.bar(range(len(groups)), [0 if np.isnan(v) else v for v in tprs])
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_title(f"TPR by {col}")
        plt.tight_layout()
        img2 = None
        if outdir:
            img2 = os.path.join(outdir, f"{prefix}_tpr_{col}.png")
            plt.savefig(img2, bbox_inches='tight')
            plt.close()
            img_paths.append((f"TPR by {col}", img2))
        else:
            plt.show()
    return img_paths

def plot_robustness_curve(model, X, y, task, outdir=None, fname="robustness_curve.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    if task == "regression":
        print("Robustness plot defined for classification here.")
        return None
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols)==0:
        print("No numeric features for noise perturbation.")
        return None
    base_pred = model.predict(X)
    rates=[]
    for s in noise_levels:
        Xp = X.copy()
        Xp[num_cols] = Xp[num_cols] + np.random.normal(0, s, size=Xp[num_cols].shape)
        yhat = model.predict(Xp)
        rates.append(float(np.mean(yhat != base_pred)))
    fig, ax = plt.subplots()
    ax.plot(noise_levels, rates, marker="o")
    ax.set_title("Prediction Flip Rate vs Noise Std")
    ax.set_xlabel("Noise std added to numeric features")
    ax.set_ylabel("Flip rate")
    plt.tight_layout()
    if outdir:
        import os
        img = os.path.join(outdir, fname)
        plt.savefig(img, bbox_inches='tight')
        plt.close()
        return ("Robustness Curve", img)
    else:
        plt.show()
        return None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

def plot_confusion_matrix_binary(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_pr(y_true, y_score, save_prefix=None):
    if len(np.unique(y_true)) != 2:
        print("ROC/PR only for binary classification.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    # ROC
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_roc.png", bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    # PR
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label=f"AP={ap:.3f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_pr.png", bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_calibration_curve(y_true, y_score, n_bins=10, save_path=None):
    bins = np.quantile(y_score, np.linspace(0,1,n_bins+1))
    mids, obs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1] + 1e-12
        m = (y_score>=lo) & (y_score<hi)
        if m.sum()==0: continue
        mids.append(y_score[m].mean())
        obs.append(y_true[m].mean())
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1],"--")
    ax.plot(mids, obs, marker="o")
    ax.set_title("Calibration (Reliability) Diagram")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, X, y, task, save_path=None):
    try:
        pi = permutation_importance(model, X, y,
                                    n_repeats=5,
                                    scoring="accuracy" if task!="regression" else "r2",
                                    random_state=7)
        means = pi.importances_mean
        order = np.argsort(np.abs(means))[::-1][:15]
        names = [X.columns[i] for i in order]
        vals = means[order]
        fig, ax = plt.subplots()
        ax.barh(range(len(names)), vals)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_title("Top Features (Permutation Importance)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print("Permutation importance failed:", e)
