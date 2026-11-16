"""Trains the same three models as notebook.py and saves everything the web app needs
to docs/data.json (test predictions, model parameters, and the raw columns for the charts).
Run from the repo root:  python scripts/export_results.py"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

df = pd.read_csv("data/diabetes_130.csv")
cols = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]
X = df[cols].copy()
X["diabetes_med"] = (df["diabetes_med"] == "Yes").astype(int)
y = (df["readmitted"] != "Not readmitted").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_train)
lr = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42).fit(X_train, y_train)

def tree_json(t):
    s = t.tree_
    leaf_p = s.value[:, 0, 1] / s.value[:, 0, :].sum(axis=1)
    return {"l": s.children_left.tolist(), "r": s.children_right.tolist(),
            "f": s.feature.tolist(), "t": [round(float(v), 6) for v in s.threshold],
            "p": [round(float(v), 5) for v in leaf_p]}

probs = {"Logistic regression": lr.predict_proba(scaler.transform(X_test))[:, 1],
         "Decision tree": tree.predict_proba(X_test)[:, 1],
         "Random forest": rf.predict_proba(X_test)[:, 1]}

out = {
    "features": list(X.columns),
    "y_test": y_test.tolist(),
    "probs": {k: [round(float(v), 5) for v in p] for k, p in probs.items()},
    "auroc_python": {k: round(float(roc_auc_score(y_test, p)), 4) for k, p in probs.items()},
    "lr": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
           "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])},
    "tree": tree_json(tree),
    "forest": [tree_json(t) for t in rf.estimators_],
    "data": {c: X[c].tolist() for c in X.columns},
    "readmitted": y.tolist(),
    "readmitted_levels": df["readmitted"].value_counts().to_dict(),
    "cats": {c: df[c].tolist() for c in ["age", "gender", "admission_type"]},
    "importance": {"Decision tree": [round(float(v), 5) for v in tree.feature_importances_],
                   "Random forest": [round(float(v), 5) for v in rf.feature_importances_]},
    "tree_text": export_text(tree, feature_names=list(X.columns), decimals=1),
    "n_train": len(X_train), "n_test": len(X_test),
}
with open("docs/data.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))
print({k: round(float(roc_auc_score(y_test, p)), 3) for k, p in probs.items()})
