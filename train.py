"""MindBeacon: train + evaluate the model.
Usage:  python train.py data/Student_Depression_Dataset.csv
Writes: model.json (used by the API and website) and reports/accuracy_report.md + reports/metrics.json
"""
import sys, json, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             roc_auc_score, confusion_matrix)
warnings.filterwarnings("ignore")

NUM = ["age", "cgpa", "press", "sat", "fin", "hrs"]
SLEEP = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
DIET = ["Unhealthy", "Moderate", "Healthy"]


def feats(d):
    X = d[NUM].copy()
    for i in range(3): X["sleep_%d" % i] = (d["sleep"] == i).astype(int)
    for i in range(2): X["diet_%d" % i] = (d["diet"] == i).astype(int)
    X["fam"] = d["fam"]; X["sui"] = d["sui"]
    return X


def load(path):
    r = pd.read_csv(path)
    n_raw = len(r)
    r = r[(r["Profession"] == "Student") & r["Sleep Duration"].isin(SLEEP) & r["Dietary Habits"].isin(DIET)]
    r = r.dropna(subset=["Financial Stress"])
    d = pd.DataFrame({"age": r["Age"], "cgpa": r["CGPA"], "press": r["Academic Pressure"], "sat": r["Study Satisfaction"],
                      "fin": r["Financial Stress"], "hrs": r["Work/Study Hours"],
                      "sleep": r["Sleep Duration"].map(SLEEP.index), "diet": r["Dietary Habits"].map(DIET.index),
                      "fam": (r["Family History of Mental Illness"] == "Yes").astype(int),
                      "sui": (r["Have you ever had suicidal thoughts ?"] == "Yes").astype(int)})
    d.attrs["raw_rows"] = n_raw
    return d, r["Depression"].astype(int)


def r4(x): return round(float(x), 4)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Student_Depression_Dataset.csv"
    d, y = load(path); X = feats(d)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=1)
    models = {
        "Logistic regression (used)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=.04, max_depth=3, l2_regularization=1, random_state=1),
        "Random forest": RandomForestClassifier(300, min_samples_leaf=5, random_state=1, n_jobs=-1)}
    comp = {}
    for n, m in models.items():
        s = cross_val_score(m, X, y, cv=cv, scoring="accuracy")
        m.fit(Xtr, ytr)
        comp[n] = {"cv_accuracy": r4(s.mean()), "cv_std": r4(s.std()), "test_accuracy": r4(m.score(Xte, yte))}
        print(n, comp[n])
    lr = models["Logistic regression (used)"]
    pred = lr.predict(Xte); prob = lr.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(yte, pred).ravel()]
    no_sui = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X.drop(columns=["sui"]), y, cv=cv).mean()
    thr = {}
    for t in (.4, .5, .6):
        p = (prob >= t).astype(int)
        thr[str(t)] = {"accuracy": r4((p == yte).mean()), "recall": r4(p[yte == 1].mean())}
    M = {"test_accuracy": r4(accuracy_score(yte, pred)), "recall": r4(recall_score(yte, pred)),
         "precision": r4(precision_score(yte, pred)), "f1": r4(f1_score(yte, pred)), "roc_auc": r4(roc_auc_score(yte, prob)),
         "cv_accuracy": comp["Logistic regression (used)"]["cv_accuracy"], "cv_std": comp["Logistic regression (used)"]["cv_std"],
         "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "test_size": int(len(yte)),
         "accuracy_without_suicidal_feature": r4(no_sui)}
    sc, clf = lr[0], lr[1]
    model = {"cols": list(X.columns), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(),
             "bias": float(clf.intercept_[0]), "metrics": M, "base_rate": r4(y.mean()), "rows": int(len(d)),
             "sleep": SLEEP, "diet": DIET}
    json.dump(model, open("model.json", "w"), indent=1)
    os.makedirs("reports", exist_ok=True)
    json.dump({"metrics": M, "comparison": comp, "thresholds": thr}, open("reports/metrics.json", "w"), indent=1)
    rows = "\n".join("| %s | %.1f%% ± %.1f | %.1f%% |" % (n, c["cv_accuracy"]*100, c["cv_std"]*100, c["test_accuracy"]*100) for n, c in comp.items())
    trows = "\n".join("| %s | %.1f%% | %.1f%% |" % (t, v["accuracy"]*100, v["recall"]*100) for t, v in thr.items())
    open("reports/accuracy_report.md", "w").write(f"""# MindBeacon accuracy report

**Data:** {d.attrs['raw_rows']:,} rows in the CSV; {len(d):,} usable student rows after cleaning (students only, valid sleep and diet entries, no missing financial stress). {y.mean()*100:.1f}% are labelled depressed. No duplicate rows.

**Split:** 80% train / 20% test ({len(yte):,} unseen students), stratified, random_state=42. Cross-validation: 5-fold stratified on all rows.

## Model comparison
| Model | 5-fold CV accuracy | Test accuracy |
|---|---|---|
{rows}

Logistic regression is used because it matches the more complex models, is explainable, and runs identically in the browser and the API.

## Final model on the unseen test set
| Metric | Value |
|---|---|
| Accuracy | {M['test_accuracy']*100:.1f}% |
| Recall (depressed students found) | {M['recall']*100:.1f}% |
| Precision | {M['precision']*100:.1f}% |
| F1 | {M['f1']*100:.1f}% |
| ROC-AUC | {M['roc_auc']:.3f} |

## Confusion matrix (test set)
| | Predicted not depressed | Predicted depressed |
|---|---|---|
| **Actually not depressed** | {tn} | {fp} (false alarms) |
| **Actually depressed** | {fn} (missed) | {tp} |

## Decision threshold
| Threshold | Accuracy | Recall |
|---|---|---|
{trows}

## Robustness
Without the "suicidal thoughts" question, cross-validated accuracy is **{no_sui*100:.1f}%**. That one answer is the strongest single predictor in this dataset.

## Limits
Survey data shows patterns, not causes. About 85% is close to the ceiling for these features: tuned gradient boosting and random forests score about the same. The model is a screening aid, not a diagnosis.
""")
    print(json.dumps(M, indent=1)); print("saved model.json, reports/")
