# Predicting Hospital Readmission in Patients with Diabetes

CAI 5735 Final Project

**Group members:** Ryan Merlino, April Walter, Nikolas Platon

**Research question:** How accurately can a diabetes patient's length of stay, number of lab tests, procedures, medications and diagnoses, plus whether they were prescribed a diabetes medication, predict whether they will be readmitted?

**Interactive app:** https://ryanmerlino21-debug.github.io/cai5735-final-project/

**Dataset:** Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository, Strack et al., 2014, https://doi.org/10.24432/C5230J). A simplified 3,000-row copy is in `data/diabetes_130.csv`.

## Files

| File | What it is |
|---|---|
| `notebook.py` | Marimo analysis notebook (split, 3 models, confusion matrices, ROC/AUROC, new-patient demo, threshold) |
| `app.py` | Marimo app source (the interactive page) |
| `docs/` | Exported app served by GitHub Pages |
| `data/`, `public/` | The CSV (`public/` copy is used by the browser app) |

## Running the notebook

```bash
git clone https://github.com/ryanmerlino21-debug/cai5735-final-project.git
cd cai5735-final-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
marimo edit notebook.py      # edit/run the analysis
marimo run app.py            # run the app locally
```

## Updating the published app

After editing `app.py`, rebuild and push:

```bash
marimo export html-wasm app.py -o docs --mode run
git add -A && git commit -m "Update app" && git push
```

## Key result

All three models (logistic regression, decision tree, random forest) reach an AUROC of about 0.59, so these in-stay measurements predict readmission only weakly. We pick the random forest with a 0.40 threshold. See the app for details.
