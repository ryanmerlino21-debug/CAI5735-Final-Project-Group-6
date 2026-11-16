# CAI5735 Final Project - Group 6

## Predicting Hospital Readmission in Patients with Diabetes



**Group members:** Ryan Merlino, April Walter, Nikolas Platon

**Research question:** How accurately can a diabetes patient's length of stay, number of lab tests, procedures, medications and diagnoses, plus whether they were prescribed a diabetes medication, predict whether they will be readmitted?

**Interactive app:** https://ryanmerlino21-debug.github.io/CAI5735-Final-Project-Group-6/

**Dataset:** Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository, Strack et al., 2014, https://doi.org/10.24432/C5230J). A simplified 3,000-row copy is in `data/diabetes_130.csv`.

## Files

| File | What it is |
|---|---|
| `notebook.py` | Marimo analysis notebook (split, 3 models, confusion matrices, ROC/AUROC, new patient demo, threshold) |
| `docs/index.html` | The web app (plain HTML, CSS and JavaScript) served by GitHub Pages |
| `docs/data.json` | Model results and parameters the app reads |
| `scripts/export_results.py` | Trains the same models as the notebook and writes `docs/data.json` |
| `data/diabetes_130.csv` | The dataset |

## Running the notebook

```bash
git clone https://github.com/ryanmerlino21-debug/CAI5735-Final-Project-Group-6.git
cd CAI5735-Final-Project-Group-6
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
marimo edit notebook.py
```

## Running the app locally

```bash
python3 -m http.server --directory docs
```

Then open http://localhost:8000. If you change the models, rerun `python scripts/export_results.py` to refresh `docs/data.json`.

## Key result

All three models (logistic regression, decision tree, random forest) reach an AUROC of about 0.59, so these in-stay measurements predict readmission only weakly. We pick the random forest with a 0.40 threshold. See the app for details.
