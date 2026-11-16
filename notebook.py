# CAI 5735 Final Project - analysis notebook
# Group members: Ryan Merlino, April Walter, Nikolas Platon
# GitHub repository: https://github.com/ryanmerlino21-debug/cai5735-final-project

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

    # Imports live in one cell so every later cell can use them.
    return (
        DecisionTreeClassifier, LogisticRegression, RandomForestClassifier,
        StandardScaler, confusion_matrix, make_pipeline, mo, np, pd, plt,
        roc_auc_score, roc_curve, train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Predicting hospital readmission for patients with diabetes

    **Group members:** Ryan Merlino, April Walter, Nikolas Platon

    ## Research question

    > **How accurately can a patient's length of stay, number of lab tests, procedures,
    > medications and diagnoses, together with whether they were prescribed a diabetes
    > medication, predict whether they will be readmitted to the hospital?**

    The response is binary: **readmitted** (came back, either within or after 30 days)
    or **not readmitted**. I chose this because readmissions are hard on patients and
    hospitals are penalized for avoidable ones. A model that flags high risk patients
    before discharge could help staff plan follow up care.

    **Data source:** *Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning
    Repository (Strack et al., 2014), https://doi.org/10.24432/C5230J. The course supplied
    a simplified 3,000-row sample.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Load the dataset
    We read the CSV with pandas and check its size and missing values before doing anything else.
    """)
    return


@app.cell
def _(pd):
    # Path works both locally and when the notebook is opened from the repo root.
    df = pd.read_csv("data/diabetes_130.csv")
    print("shape:", df.shape, "| missing values:", int(df.isna().sum().sum()))
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Select predictor and response variables
    The original `readmitted` column has three levels (*Not readmitted*, *After 30 days*,
    *Within 30 days*). Our question is about whether the patient came back at all, so we
    collapse it into a 0/1 variable. The classes are reasonably balanced
    (about 40% readmitted), so plain accuracy is meaningful here.

    **Predictors** are the five numeric counts plus `diabetes_med` (Yes = 1). We left out
    age, gender and admission type to keep the question focused on what happens
    *during the stay*.
    """)
    return


@app.cell
def _(df):
    numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                    "num_medications", "number_diagnoses"]
    X = df[numeric_cols].copy()
    X["diabetes_med"] = (df["diabetes_med"] == "Yes").astype(int)
    y = (df["readmitted"] != "Not readmitted").astype(int)  # 1 = readmitted
    print(y.value_counts(normalize=True).rename({0: "not readmitted", 1: "readmitted"}))
    return X, y


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Train / test split
    We hold out 25% of rows as a test set that no model sees during training.
    `stratify=y` keeps the readmission rate the same in both sets, and a fixed
    `random_state` makes results reproducible for everyone in the group.
    """)
    return


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(len(X_train), "train rows,", len(X_test), "test rows")
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Train three classification models
    * **Logistic regression** - simple, interpretable baseline. Features are standardised so
      coefficients are comparable.
    * **Decision tree** - captures non-linear rules; `max_depth=4` and `min_samples_leaf=30`
      stop it from memorising the training data.
    * **Random forest** - an ensemble of many trees that usually generalises better than one tree.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier, LogisticRegression, RandomForestClassifier,
    StandardScaler, X_train, make_pipeline, y_train,
):
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42),
        "Random forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42
        ),
    }
    for _m in models.values():
        _m.fit(X_train, y_train)
    list(models)
    return (models,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Evaluate each model on the test set
    For each model we compute, at the default 0.5 threshold:
    * **Confusion matrix** - counts of true/false positives and negatives.
    * **Accuracy** = (TP+TN) / all
    * **Sensitivity** (recall) = TP / (TP+FN) - the share of truly readmitted patients we catch.
    * **Specificity** = TN / (TN+FP) - the share of non-readmitted patients we correctly leave alone.
    * **ROC curve and AUROC** - performance across *all* thresholds.
    """)
    return


@app.cell
def _(X_test, confusion_matrix, models, pd, roc_auc_score, roc_curve, y_test):
    def metrics_at(y_true, prob, thr=0.5):
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        return dict(TN=tn, FP=fp, FN=fn, TP=tp,
                    accuracy=(tp + tn) / len(y_true),
                    sensitivity=tp / (tp + fn),
                    specificity=tn / (tn + fp))

    probs = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}
    rocs = {n: roc_curve(y_test, p) for n, p in probs.items()}
    results = pd.DataFrame(
        {n: {**metrics_at(y_test, p), "AUROC": roc_auc_score(y_test, p)} for n, p in probs.items()}
    ).T
    results.round(3)
    return metrics_at, probs, results, rocs


@app.cell
def _(models, np, plt, probs, y_test, metrics_at):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (name, p) in zip(axes, probs.items()):
        m = metrics_at(y_test, p)
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Not readm.", "Readmitted"])
        ax.set_yticks([0, 1], ["Not readm.", "Readmitted"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
    fig.tight_layout()
    fig
    return


@app.cell
def _(plt, results, rocs):
    fig2, ax2 = plt.subplots(figsize=(5.5, 5))
    for _n, (fpr, tpr, _) in rocs.items():
        ax2.plot(fpr, tpr, label=f"{_n} (AUROC {results.loc[_n, 'AUROC']:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", label="Chance")
    ax2.set_xlabel("False positive rate (1 - specificity)")
    ax2.set_ylabel("True positive rate (sensitivity)")
    ax2.set_title("ROC curves on the test set"); ax2.legend()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Classify a new patient
    We build a made-up patient and ask each model for a readmission probability. The
    patient stayed 7 days, had 55 lab tests, 2 procedures, 20 medications, 8 diagnoses
    and was prescribed a diabetes medication.
    """)
    return


@app.cell
def _(X, models, pd):
    new_patient = pd.DataFrame([{
        "time_in_hospital": 7, "num_lab_procedures": 55, "num_procedures": 2,
        "num_medications": 20, "number_diagnoses": 8, "diabetes_med": 1,
    }])[X.columns]
    for _name, _m in models.items():
        _p = _m.predict_proba(new_patient)[0, 1]
        print(f"{_name:20s} P(readmitted) = {_p:.2f} -> "
              f"{'READMITTED' if _p >= 0.5 else 'not readmitted'}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Compare models and choose the best
    The table in section 5 is the comparison. AUROC is our main yardstick because it does
    not depend on a threshold.

    All three models land at an AUROC of about **0.59**, only slightly better than a coin
    flip (0.50). The **random forest** has the highest AUROC (0.590) and the highest
    accuracy (0.613), so we pick it, but the margin over logistic regression (0.587) and the
    decision tree (0.586) is tiny and well within test-set noise. We would also defend
    logistic regression as the practical choice because it is simpler and explainable.
    At the default 0.5 threshold, all models have very low sensitivity (0.16 to 0.30):
    they mostly predict "not readmitted".
    """)
    return


@app.cell
def _(results):
    best = results["AUROC"].idxmax()
    print(results[["accuracy", "sensitivity", "specificity", "AUROC"]].round(3).to_string())
    print("\nHighest AUROC:", best)
    return (best,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Choosing a threshold
    A readmission that we miss (false negative) means a patient who needed follow-up care
    does not get it, while a false alarm (false positive) only costs a follow-up call.
    So we should favour **sensitivity** over specificity. The table below sweeps the
    threshold for the chosen model; Youden's J (sensitivity + specificity - 1) marks the
    point that balances both.

    **Our choice: threshold = 0.40.** For the random forest this gives sensitivity of about
    0.66 and specificity of about 0.48 (the highest Youden's J, 0.139). At 0.50 the model
    would catch only 21% of readmissions. Going lower than 0.35 flags nearly everyone, which
    makes the model useless for prioritising follow-up.
    """)
    return


@app.cell
def _(best, metrics_at, np, pd, probs, y_test):
    sweep = pd.DataFrame(
        [{"threshold": t, **{k: v for k, v in metrics_at(y_test, probs[best], t).items()
                             if k in ("accuracy", "sensitivity", "specificity")}}
         for t in np.arange(0.2, 0.65, 0.05)]
    ).round(3)
    sweep["youden_J"] = (sweep.sensitivity + sweep.specificity - 1).round(3)
    print("Model:", best)
    print(sweep.to_string(index=False))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Conclusion
    **Answer to the research question:** these variables do not predict readmission very
    accurately. The best model, the random forest, reached an AUROC of about 0.59 and 61%
    accuracy at a threshold of 0.5, while always guessing not readmitted gives 60%. The
    counts recorded during the stay only have a weak link to readmission. A better model
    would probably need more information, like prior admissions, discharge destination, and
    lab results such as HbA1c. For now the models are only a rough screening tool and should
    not be used alone to make decisions about a patient. The app (`app.py`) has the
    interactive version.
    """)
    return


if __name__ == "__main__":
    app.run()
