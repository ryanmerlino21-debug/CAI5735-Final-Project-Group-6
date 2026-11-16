# CAI 5735 Final Project - interactive app
# Group members: Ryan Merlino, April Walter, Nikolas Platon
# GitHub repository: https://github.com/ryanmerlino21-debug/cai5735-final-project

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium", app_title="Predicting Diabetes Readmission")


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

    df = pd.read_csv(str(mo.notebook_location() / "public" / "diabetes_130.csv"))
    X = df[["time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_diagnoses"]].copy()
    X["diabetes_med"] = (df["diabetes_med"] == "Yes").astype(int)
    y = (df["readmitted"] != "Not readmitted").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42),
        "Random forest": RandomForestClassifier(n_estimators=200, max_depth=6,
                                                min_samples_leaf=20, random_state=42),
    }
    for _m in models.values():
        _m.fit(X_train, y_train)
    probs = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}
    aucs = {n: roc_auc_score(y_test, p) for n, p in probs.items()}
    return (X, X_test, aucs, confusion_matrix, df, mo, models, np, pd, plt,
            probs, roc_curve, y, y_test)


@app.cell
def _(mo):
    mo.md(r"""
    # Can we predict who comes back to the hospital?
    *CAI 5735 Final Project - Ryan Merlino, April Walter, Nikolas Platon*

    ## Research question
    > **How accurately can a diabetes patient's length of stay, number of lab tests,
    > procedures, medications and diagnoses, plus whether they were prescribed a diabetes
    > medication, predict whether they will be readmitted to the hospital?**

    **Why this matters.** I chose this question because readmissions are hard on patients
    and also costly for hospitals. Hospitals in the US are even penalized when too many
    patients come back. If a model could flag high risk patients before they go home, staff
    could focus follow up calls and medication reviews on those patients first. Which means
    that better care could go to the people who need it most. In this project we test how far
    a few simple numbers recorded during the stay can take us.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The dataset
    **Source:** *Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning
    Repository. Strack et al. (2014), *Impact of HbA1c measurement on hospital readmission
    rates*, BioMed Research International. https://doi.org/10.24432/C5230J.
    Our copy is a simplified **3,000-stay** sample (from 130 hospitals) supplied by the course.

    | Column | Role | Meaning |
    |---|---|---|
    | `readmitted` | **Response** | Came back to the hospital? (we use: yes / no) |
    | `time_in_hospital` | Predictor | Days in hospital |
    | `num_lab_procedures` | Predictor | Lab tests during the stay |
    | `num_procedures` | Predictor | Non-lab procedures |
    | `num_medications` | Predictor | Distinct medications given |
    | `number_diagnoses` | Predictor | Diagnoses recorded |
    | `diabetes_med` | Predictor | Prescribed a diabetes medication (yes / no) |
    | age, gender, admission_type | Not used | Left out to focus on in-stay measurements |
    """)
    return


@app.cell
def _(df, mo):
    _counts = df["readmitted"].value_counts()
    mo.md(f"""
    The original outcome has three groups: not readmitted ({_counts['Not readmitted']}),
    after 30 days ({_counts['After 30 days']}), and within 30 days ({_counts['Within 30 days']}).
    Our question is about whether a patient comes back at all, so we combined the last two
    groups into one group called readmitted ({_counts['After 30 days'] + _counts['Within 30 days']} stays, about 40%).
    There are no missing values in this file.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Explore the data
    Pick a variable to see how readmitted and not readmitted stays compare. The two groups
    look very similar, which already hints at what we found.
    """)
    return


@app.cell
def _(mo):
    var = mo.ui.dropdown(
        ["time_in_hospital", "num_lab_procedures", "num_procedures",
         "num_medications", "number_diagnoses"],
        value="num_medications", label="Variable")
    var
    return (var,)


@app.cell
def _(df, np, plt, var, y):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    _lo, _hi = df[var.value].min(), df[var.value].max()
    _bins = np.linspace(_lo, _hi + 1, min(25, int(_hi - _lo) + 2))
    ax.hist(df.loc[y == 0, var.value], bins=_bins, alpha=.6, density=True,
            label="Not readmitted", color="#4C78A8")
    ax.hist(df.loc[y == 1, var.value], bins=_bins, alpha=.6, density=True,
            label="Readmitted", color="#E45756")
    ax.set_xlabel(var.value); ax.set_ylabel("Share of stays"); ax.legend()
    _m0, _m1 = df.loc[y == 0, var.value].mean(), df.loc[y == 1, var.value].mean()
    ax.set_title(f"Mean: not readmitted {_m0:.2f} vs readmitted {_m1:.2f}")
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Analysis process
    First, I split the data into 75% training and 25% testing, so every result below comes
    from 750 stays the models never saw. Then we trained three models: logistic regression
    as a simple baseline, a decision tree with a depth of 4, and a random forest. Next, we
    evaluated each one with a confusion matrix, accuracy, sensitivity, specificity, and the
    ROC curve with AUROC. Sensitivity is the share of real readmissions the model catches,
    and specificity is the share of patients who did not come back that the model correctly
    leaves alone. Finally, we compared the models and chose a threshold.

    You can use the controls below to explore each model. The threshold is the predicted
    probability where a patient gets labeled as readmitted.
    """)
    return


@app.cell
def _(mo, models):
    model_pick = mo.ui.dropdown(list(models), value="Random forest", label="Model")
    thr = mo.ui.slider(0.10, 0.70, step=0.01, value=0.40, label="Threshold", show_value=True)
    mo.hstack([model_pick, thr], justify="start", gap=2)
    return model_pick, thr


@app.cell
def _(aucs, confusion_matrix, mo, model_pick, np, plt, probs, roc_curve, thr, y_test):
    _p = probs[model_pick.value]
    _pred = (_p >= thr.value).astype(int)
    _tn, _fp, _fn, _tp = confusion_matrix(y_test, _pred).ravel()
    _acc = (_tp + _tn) / len(y_test)
    _sens = _tp / (_tp + _fn); _spec = _tn / (_tn + _fp)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 4))
    _cm = np.array([[_tn, _fp], [_fn, _tp]])
    _a1.imshow(_cm, cmap="Blues")
    for (_i, _j), _v in np.ndenumerate(_cm):
        _a1.text(_j, _i, _v, ha="center", va="center", fontsize=16,
                 color="white" if _v > _cm.max() / 2 else "black")
    _a1.set_xticks([0, 1], ["Not readm.", "Readmitted"])
    _a1.set_yticks([0, 1], ["Not readm.", "Readmitted"])
    _a1.set_xlabel("Predicted"); _a1.set_ylabel("Actual"); _a1.set_title("Confusion matrix (test set)")

    for _n, _pp in probs.items():
        _f, _t, _ = roc_curve(y_test, _pp)
        _a2.plot(_f, _t, lw=2.5 if _n == model_pick.value else 1,
                 alpha=1 if _n == model_pick.value else .4, label=f"{_n} ({aucs[_n]:.3f})")
    _a2.plot([0, 1], [0, 1], "k--", lw=1)
    _a2.scatter([1 - _spec], [_sens], s=90, color="red", zorder=5, label="Current threshold")
    _a2.set_xlabel("False positive rate"); _a2.set_ylabel("Sensitivity")
    _a2.set_title("ROC curves (AUROC)"); _a2.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()

    mo.vstack([
        mo.hstack([mo.stat(f"{_acc:.1%}", label="Accuracy"),
                   mo.stat(f"{_sens:.1%}", label="Sensitivity"),
                   mo.stat(f"{_spec:.1%}", label="Specificity"),
                   mo.stat(f"{aucs[model_pick.value]:.3f}", label="AUROC")]),
        _fig,
    ])
    return


@app.cell
def _(aucs, mo):
    mo.md(f"""
    ### Model comparison (test set, threshold 0.5)
    | Model | Accuracy | Sensitivity | Specificity | AUROC |
    |---|---|---|---|---|
    | Logistic regression | 0.583 | 0.157 | 0.867 | {aucs['Logistic regression']:.3f} |
    | Decision tree | 0.604 | 0.300 | 0.807 | {aucs['Decision tree']:.3f} |
    | Random forest | 0.613 | 0.213 | 0.880 | {aucs['Random forest']:.3f} |

    **Which model is best?** I chose the random forest because it has the highest AUROC and
    accuracy. The three models are very close though, with AUROC around 0.59. Logistic
    regression is also a good choice because it is simpler and easier to explain. At a
    threshold of 0.5 every model has very low sensitivity because they mostly guess not
    readmitted. About 60% of stays are not readmitted, so 60% accuracy is the number to beat.

    **Threshold.** Missing a patient who comes back is worse than making an extra follow up
    call, so I want higher sensitivity. For the random forest, 0.40 gave the best balance
    (sensitivity 0.66 and specificity 0.48). Try the slider to see the trade off. Below about
    0.35 almost every patient is flagged, and above 0.5 almost nobody is.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Try it: classify a new patient
    Enter the details of a patient's stay and see each model's predicted chance of readmission.
    """)
    return


@app.cell
def _(mo):
    s_days = mo.ui.slider(1, 14, value=7, label="Days in hospital", show_value=True)
    s_lab = mo.ui.slider(1, 130, value=55, label="Lab tests", show_value=True)
    s_proc = mo.ui.slider(0, 6, value=2, label="Procedures", show_value=True)
    s_med = mo.ui.slider(1, 80, value=20, label="Medications", show_value=True)
    s_diag = mo.ui.slider(1, 16, value=8, label="Diagnoses", show_value=True)
    s_dm = mo.ui.checkbox(value=True, label="Prescribed a diabetes medication")
    mo.vstack([mo.hstack([s_days, s_lab, s_proc], justify="start"),
               mo.hstack([s_med, s_diag, s_dm], justify="start")])
    return s_days, s_diag, s_dm, s_lab, s_med, s_proc


@app.cell
def _(X, mo, models, pd, s_days, s_diag, s_dm, s_lab, s_med, s_proc, thr):
    _pt = pd.DataFrame([{
        "time_in_hospital": s_days.value, "num_lab_procedures": s_lab.value,
        "num_procedures": s_proc.value, "num_medications": s_med.value,
        "number_diagnoses": s_diag.value, "diabetes_med": int(s_dm.value)}])[X.columns]
    _rows = ""
    for _n, _m in models.items():
        _pr = _m.predict_proba(_pt)[0, 1]
        _rows += (f"| {_n} | {_pr:.0%} | "
                  f"{'**Readmitted**' if _pr >= thr.value else 'Not readmitted'} |\n")
    mo.md(f"""
    | Model | P(readmitted) | Prediction at threshold {thr.value:.2f} |
    |---|---|---|
    {_rows}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Results and conclusion
    **Answer.** These variables do not predict readmission very accurately. Our best model,
    the random forest, reached an AUROC of about 0.59. A score of 0.5 is a coin flip and 1.0
    is perfect, so this is only slightly better than guessing. Its accuracy of 61% at a
    threshold of 0.5 barely beats always guessing not readmitted, which is 60%. When I lowered
    the threshold to 0.40, the model caught about two thirds of readmissions, but it also
    flagged about half of the patients who did not come back.

    **What this means.** The numbers recorded during a stay, like days in the hospital, tests,
    and medications, are only weakly related to coming back. Things like prior admissions,
    where the patient goes after discharge, and lab results such as HbA1c would probably help
    more, but they are not in this simplified file. Because of that, the model could only be a
    rough screening tool, and it should not be used alone to make decisions about a patient.

    **Limitations.** The sample is only 3,000 stays and we used one random split, so the
    AUROC could change by about 0.02 with a different split. We also did not tune the models,
    and the data is from 1999 to 2008.
    """)
    return


if __name__ == "__main__":
    app.run()
