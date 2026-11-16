# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///

# CAI 5735 Final Project - analysis notebook
# Group 6 members: Ryan Merlino, April Walter, Nikolas Platon
# GitHub repository: https://github.com/ryanmerlino21-debug/CAI5735-Final-Project-Group-6
# Web app: https://ryanmerlino21-debug.github.io/CAI5735-Final-Project-Group-6/
# Dataset: Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning Repository
#          (Strack et al., 2014), https://doi.org/10.24432/C5230J
#
# Every code cell below has a "Why" comment (or a markdown cell right above it)
# that explains why the code was written.

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
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

    # Why: marimo shares variables between cells, so we import every library once here.
    # pandas/numpy handle the data, matplotlib draws the plots, and scikit-learn
    # provides the split, the three models and the evaluation metrics.
    return (
        DecisionTreeClassifier, LogisticRegression, RandomForestClassifier,
        StandardScaler, confusion_matrix, make_pipeline, mo, np, pd, plot_tree,
        plt, roc_auc_score, roc_curve, train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Predicting hospital readmission for patients with diabetes

    **CAI 5735 Final Project, Group 6:** Ryan Merlino, April Walter, Nikolas Platon

    **GitHub repository:** https://github.com/ryanmerlino21-debug/CAI5735-Final-Project-Group-6
    · **Web app:** https://ryanmerlino21-debug.github.io/CAI5735-Final-Project-Group-6/

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

    **Steps in this notebook:** 1 load the data, 2 pick variables, 3 split, 4 train three
    models, 5 evaluate each one, 6 classify a new patient, 7 compare the models,
    8 choose a threshold, 9 conclusion.
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
def _(mo, pd):
    # Why: this is the load step. mo.notebook_location() finds the data folder next to the
    # notebook no matter which directory marimo was started from. We show the shape and the
    # missing value count as a quick check that the file loaded correctly, then show the
    # first rows so everyone can see what one hospital stay looks like.
    df = pd.read_csv(mo.notebook_location() / "data" / "diabetes_130.csv")
    mo.vstack([
        mo.md(f"**{df.shape[0]:,} rows, {df.shape[1]} columns, "
              f"{int(df.isna().sum().sum())} missing values.**"),
        df.head(),
    ])
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
def _(df, mo):
    # Why: this cell picks the response (y) and predictors (X). y turns the three level
    # readmitted column into 0/1, and X keeps the five numeric counts plus a 0/1 version
    # of diabetes_med so every model gets numbers only.
    numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                    "num_medications", "number_diagnoses"]
    X = df[numeric_cols].copy()
    X["diabetes_med"] = (df["diabetes_med"] == "Yes").astype(int)
    y = (df["readmitted"] != "Not readmitted").astype(int)  # 1 = readmitted
    mo.md(f"**Response:** `readmitted` (1 = came back). **{y.mean():.1%}** of stays are readmitted "
          f"and **{1 - y.mean():.1%}** are not.  \n**Predictors:** {', '.join(f'`{c}`' for c in X.columns)}")
    return X, y


@app.cell
def _(mo):
    mo.md(r"""
    ### A quick look at the predictors
    Before modeling we compare each predictor between the two groups. If a predictor were a
    strong signal, the two histograms would look clearly different.
    """)
    return


@app.cell
def _(X, plt, y):
    # Why: an exploratory check. For each numeric predictor we overlay a histogram for
    # readmitted and not readmitted stays (as a share of each group so the groups are
    # comparable even though they have different sizes). The x axes include units.
    _units = {"time_in_hospital": "Days in hospital (days)",
              "num_lab_procedures": "Lab tests (count per stay)",
              "num_procedures": "Procedures (count per stay)",
              "num_medications": "Medications (count per stay)",
              "number_diagnoses": "Diagnoses (count per stay)"}
    _fig, _axes = plt.subplots(1, 5, figsize=(17, 3.2))
    for _ax, (_c, _lab) in zip(_axes, _units.items()):
        _bins = 12
        _ax.hist(X.loc[y == 0, _c], bins=_bins, alpha=.6, density=True, color="#00a99d", label="Not readmitted")
        _ax.hist(X.loc[y == 1, _c], bins=_bins, alpha=.6, density=True, color="#ff5d73", label="Readmitted")
        _ax.set_xlabel(_lab, fontsize=8)
        _ax.set_ylabel("Share of group (density)", fontsize=8)
    _axes[0].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


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
def _(X, mo, train_test_split, y):
    # Why: the split. Models must be judged on stays they were not trained on, so we keep
    # 25% back for testing. stratify=y keeps the readmission rate equal in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    mo.md(f"**{len(X_train):,} training stays** ({y_train.mean():.1%} readmitted) and "
          f"**{len(X_test):,} test stays** ({y_test.mean():.1%} readmitted).")
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Train three classification models
    * **Logistic regression** - simple, interpretable baseline. Features are standardized so
      coefficients are comparable.
    * **Decision tree** - captures non-linear rules; `max_depth=4` and `min_samples_leaf=30`
      stop it from memorizing the training data.
    * **Random forest** - an ensemble of many trees that usually generalizes better than one tree.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier, LogisticRegression, RandomForestClassifier,
    StandardScaler, X_train, make_pipeline, mo, y_train,
):
    # Why: these are the three classification models we compare. Logistic regression gets a
    # StandardScaler in a pipeline because it works best with variables on the same scale.
    # The tree depth and leaf size limits are there to prevent overfitting. We fit every
    # model on the training set only.
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42),
        "Random forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42
        ),
    }
    for _m in models.values():
        _m.fit(X_train, y_train)
    mo.md("Trained: " + ", ".join(f"**{n}**" for n in models))
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
    # Why: this cell scores every model on the test set. metrics_at() turns predicted
    # probabilities into yes/no predictions at a chosen threshold and returns the confusion
    # matrix counts, accuracy, sensitivity and specificity. We reuse it later for the
    # threshold sweep. roc_curve and roc_auc_score give the ROC points and the AUROC, which
    # do not depend on a threshold. The table is the model comparison.
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
def _(metrics_at, models, np, plt, probs, y_test):
    # Why: a confusion matrix for each model, drawn side by side so the mistakes are easy to
    # compare. Rows are what really happened and columns are what the model predicted.
    # Each box shows the number of test stays (count).
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, p) in zip(axes, probs.items()):
        m = metrics_at(y_test, p)
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Not readmitted", "Readmitted"])
        ax.set_yticks([0, 1], ["Not readmitted", "Readmitted"])
        ax.set_xlabel("Predicted (number of test stays)")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nsensitivity {m['sensitivity']:.2f}, specificity {m['specificity']:.2f}", fontsize=10)
    fig.tight_layout()
    fig
    return


@app.cell
def _(plt, results, rocs):
    # Why: the ROC curve for every model on one plot. The dashed diagonal is a coin flip, so
    # curves closer to the top left are better. The legend shows each model's AUROC.
    fig2, ax2 = plt.subplots(figsize=(5.8, 5))
    for _n, (fpr, tpr, _) in rocs.items():
        ax2.plot(fpr, tpr, label=f"{_n} (AUROC {results.loc[_n, 'AUROC']:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", label="Coin flip (AUROC 0.500)")
    ax2.set_xlabel("False positive rate (share of non-readmitted flagged, 0 to 1)")
    ax2.set_ylabel("True positive rate = sensitivity (0 to 1)")
    ax2.set_title("ROC curves on the test set"); ax2.legend(fontsize=8)
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
def _(X, mo, models, pd):
    # Why: this demonstrates using each trained model on a brand new patient. We build one row
    # with the same columns as X, ask each model for a probability of readmission, and apply
    # the default 0.5 cutoff to get a yes/no prediction.
    new_patient = pd.DataFrame([{
        "time_in_hospital": 7, "num_lab_procedures": 55, "num_procedures": 2,
        "num_medications": 20, "number_diagnoses": 8, "diabetes_med": 1,
    }])[X.columns]
    _rows = []
    for _name, _m in models.items():
        _p = _m.predict_proba(new_patient)[0, 1]
        _rows.append(f"| {_name} | {_p:.0%} | {'**Readmitted**' if _p >= 0.5 else 'Not readmitted'} |")
    mo.md("| Model | Predicted chance of readmission | Prediction at 0.5 |\n|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Try your own patient
    Move the sliders to describe a different patient. The table updates with each model's
    prediction at the threshold you pick.
    """)
    return


@app.cell
def _(mo):
    # Why: marimo can make the notebook interactive. These widgets let anyone in the group
    # describe a patient and see the models respond without editing code.
    s_days = mo.ui.slider(1, 14, value=7, label="Days in hospital (days)", show_value=True)
    s_lab = mo.ui.slider(1, 132, value=55, label="Lab tests (count)", show_value=True)
    s_proc = mo.ui.slider(0, 6, value=2, label="Procedures (count)", show_value=True)
    s_med = mo.ui.slider(1, 81, value=20, label="Medications (count)", show_value=True)
    s_diag = mo.ui.slider(1, 16, value=8, label="Diagnoses (count)", show_value=True)
    s_dm = mo.ui.checkbox(value=True, label="Prescribed a diabetes medication")
    s_thr = mo.ui.slider(0.1, 0.7, step=0.01, value=0.4, label="Threshold (probability)", show_value=True)
    mo.vstack([mo.hstack([s_days, s_lab, s_proc], justify="start"),
               mo.hstack([s_med, s_diag, s_dm], justify="start"), s_thr])
    return s_days, s_diag, s_dm, s_lab, s_med, s_proc, s_thr


@app.cell
def _(X, mo, models, pd, s_days, s_diag, s_dm, s_lab, s_med, s_proc, s_thr):
    # Why: this reads the widgets, builds one row for the patient and shows every model's
    # predicted probability and yes/no prediction at the chosen threshold.
    _pt = pd.DataFrame([{
        "time_in_hospital": s_days.value, "num_lab_procedures": s_lab.value,
        "num_procedures": s_proc.value, "num_medications": s_med.value,
        "number_diagnoses": s_diag.value, "diabetes_med": int(s_dm.value),
    }])[X.columns]
    _rows = []
    for _name, _m in models.items():
        _p = _m.predict_proba(_pt)[0, 1]
        _rows.append(f"| {_name} | {_p:.0%} | {'**Readmitted**' if _p >= s_thr.value else 'Not readmitted'} |")
    mo.md(f"| Model | Predicted chance of readmission | Prediction at {s_thr.value:.2f} |\n|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Compare models and choose the best
    AUROC is our main yardstick because it does not depend on a threshold. The table in
    section 5 has the numbers, and the chart below puts them side by side.

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
def _(np, plt, results):
    # Why: this picks the best model by AUROC (it does not depend on a threshold) and charts
    # all four measures for the three models so the comparison is easy to check by eye.
    best = results["AUROC"].idxmax()
    _cols = ["accuracy", "sensitivity", "specificity", "AUROC"]
    _x = np.arange(len(_cols)); _w = 0.26
    fig3, ax3 = plt.subplots(figsize=(8, 3.8))
    for _i, (_n, _row) in enumerate(results[_cols].iterrows()):
        ax3.bar(_x + (_i - 1) * _w, _row.values, _w, label=_n)
    ax3.set_xticks(_x, ["Accuracy", "Sensitivity", "Specificity", "AUROC"])
    ax3.set_ylabel("Score on the test set (0 to 1)")
    ax3.set_xlabel("Measure (threshold 0.5 for the first three)")
    ax3.set_ylim(0, 1); ax3.legend(fontsize=8)
    ax3.set_title(f"Model comparison (highest AUROC: {best})")
    fig3
    return (best,)


@app.cell
def _(models, np, plt):
    # Why: this shows which variables the models relied on, so we can explain the results.
    # Logistic regression uses the absolute size of its standardized coefficients and the
    # tree models use their built in feature importances. Each is scaled to add to 100%.
    _names = ["Days in hospital", "Lab tests", "Procedures", "Medications", "Diagnoses", "Diabetes med"]
    _lr = np.abs(models["Logistic regression"][-1].coef_[0])
    _imp = {"Logistic regression": _lr / _lr.sum() * 100,
            "Decision tree": models["Decision tree"].feature_importances_ * 100,
            "Random forest": models["Random forest"].feature_importances_ * 100}
    fig4, ax4 = plt.subplots(figsize=(9, 3.8))
    _x = np.arange(len(_names)); _w = 0.26
    for _i, (_n, _v) in enumerate(_imp.items()):
        ax4.bar(_x + (_i - 1) * _w, _v, _w, label=_n)
    ax4.set_xticks(_x, _names)
    ax4.set_xlabel("Predictor")
    ax4.set_ylabel("Share of the model's importance (%)")
    ax4.set_title("What each model relied on"); ax4.legend(fontsize=8)
    fig4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Could the differences just be luck?
    We redraw the 750 test stays with replacement 300 times and recompute each AUROC. If the
    models' ranges overlap, the test set cannot tell them apart.
    """)
    return


@app.cell
def _(np, plt, probs, roc_auc_score, y_test):
    # Why: a bootstrap check. A single test set gives a single AUROC per model, but a
    # different set of stays could give slightly different numbers. Resampling shows how
    # much the AUROC moves by chance, which tells us whether "best" is really meaningful.
    _rng = np.random.default_rng(42)
    _y = y_test.to_numpy()
    _aucs = {n: [] for n in probs}
    for _ in range(300):
        _idx = _rng.integers(0, len(_y), len(_y))
        for _n, _p in probs.items():
            _aucs[_n].append(roc_auc_score(_y[_idx], _p[_idx]))
    fig5, ax5 = plt.subplots(figsize=(7, 3))
    for _i, (_n, _v) in enumerate(_aucs.items()):
        _lo, _hi = np.percentile(_v, [2.5, 97.5])
        ax5.plot([_lo, _hi], [_i, _i], lw=8, solid_capstyle="round", label=f"{_n}: {_lo:.3f} to {_hi:.3f}")
    ax5.set_yticks(range(len(_aucs)), list(_aucs))
    ax5.set_xlabel("AUROC (middle 95% of 300 resamples, 0.5 = coin flip)")
    ax5.set_ylabel("Model")
    ax5.axvline(0.5, color="k", ls="--", lw=1); ax5.legend(fontsize=7)
    fig5
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Choosing a threshold
    A readmission that we miss (false negative) means a patient who needed follow-up care
    does not get it, while a false alarm (false positive) only costs a follow-up call.
    So we should favor **sensitivity** over specificity. The plot and table below sweep the
    threshold for the chosen model; Youden's J (sensitivity + specificity - 1) marks the
    point that balances both.

    **Our choice: threshold = 0.40.** For the random forest this gives sensitivity of about
    0.66 and specificity of about 0.48 (the highest Youden's J, 0.139). At 0.50 the model
    would catch only 21% of readmissions. Going lower than 0.35 flags nearly everyone, which
    makes the model useless for prioritizing follow-up.
    """)
    return


@app.cell
def _(best, metrics_at, np, pd, plt, probs, y_test):
    # Why: this supports the threshold decision. For the chosen model we recompute accuracy,
    # sensitivity and specificity at many thresholds, plus Youden's J (sensitivity +
    # specificity - 1) to see where the balance between the two is best. The plot shows the
    # trade off and the table lists the same numbers.
    _ts = np.arange(0.1, 0.701, 0.01)
    sweep = pd.DataFrame(
        [{"threshold": t, **{k: v for k, v in metrics_at(y_test, probs[best], t).items()
                             if k in ("accuracy", "sensitivity", "specificity")}} for t in _ts]
    )
    sweep["youden_J"] = sweep.sensitivity + sweep.specificity - 1
    fig6, ax6 = plt.subplots(figsize=(8, 3.8))
    ax6.plot(sweep.threshold, sweep.sensitivity, label="Sensitivity", color="#ff5d73")
    ax6.plot(sweep.threshold, sweep.specificity, label="Specificity", color="#00a99d")
    ax6.plot(sweep.threshold, sweep.youden_J, label="Youden's J", color="#7a4dff", ls="--")
    ax6.axvline(0.4, color="k", ls=":", label="Our choice (0.40)")
    ax6.set_xlabel("Threshold (predicted probability, 0 to 1)")
    ax6.set_ylabel("Score on the test set (0 to 1)")
    ax6.set_title(f"Threshold sweep for the {best}"); ax6.legend(fontsize=8)
    fig6
    return (sweep,)


@app.cell
def _(best, mo, sweep):
    # Why: the same sweep as a table at round thresholds, so the numbers quoted in the
    # write up (0.40 vs 0.50) can be checked directly.
    mo.vstack([mo.md(f"**{best}** at common thresholds"),
               sweep[sweep.threshold.round(2).isin([0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6])].round(3).reset_index(drop=True)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Try a different threshold
    Drag the slider to see how the confusion matrix for the random forest changes.
    """)
    return


@app.cell
def _(mo):
    # Why: an interactive control so the group can see the trade off for themselves.
    thr = mo.ui.slider(0.1, 0.7, step=0.01, value=0.4, label="Threshold (probability)", show_value=True)
    thr
    return (thr,)


@app.cell
def _(metrics_at, mo, probs, thr, y_test):
    # Why: this recomputes the confusion matrix and the three required measures for the
    # random forest at whatever threshold the slider is set to.
    _m = metrics_at(y_test, probs["Random forest"], thr.value)
    mo.md(f"""
    At a threshold of **{thr.value:.2f}**:

    | | Predicted not readmitted | Predicted readmitted |
    |---|---|---|
    | **Actually not readmitted** | {_m['TN']} | {_m['FP']} |
    | **Actually readmitted** | {_m['FN']} | {_m['TP']} |

    Accuracy **{_m['accuracy']:.3f}** · Sensitivity **{_m['sensitivity']:.3f}** · Specificity **{_m['specificity']:.3f}**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Conclusion
    **Answer to the research question:** these variables do not predict readmission very
    accurately. The best model, the random forest, reached an AUROC of about 0.59 and 61%
    accuracy at a threshold of 0.5, while always guessing not readmitted gives 60%. The
    counts recorded during the stay only have a weak link to readmission. The number of
    diagnoses mattered most to the models, but even that does not separate the two groups
    well. A better model would probably need more information, like prior admissions,
    discharge destination, and lab results such as HbA1c. For now the models are only a
    rough screening tool and should not be used alone to make decisions about a patient.

    **Limitations:** a 3,000 row sample, one random split (the bootstrap shows the AUROC
    could move by about 0.02 or more), no tuning of the models, and data from 1999 to 2008.

    The web app on GitHub Pages (link at the top of this notebook and in the README) has the
    interactive version with many more charts.
    """)
    return


if __name__ == "__main__":
    app.run()
