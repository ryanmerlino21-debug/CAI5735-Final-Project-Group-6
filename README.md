# CAI5735 Final Project - Group 6

## Predicting Hospital Readmission in Patients with Diabetes



**Group members:** Ryan Merlino, April Walter, Nikolas Platon

**Research question:** How accurately can a diabetes patient's length of stay, number of lab tests, procedures, medications and diagnoses, plus whether they were prescribed a diabetes medication, predict whether they will be readmitted?

**Interactive app:** https://ryanmerlino21-debug.github.io/CAI5735-Final-Project-Group-6/

**Marimo notebook (rendered, with outputs):** https://ryanmerlino21-debug.github.io/CAI5735-Final-Project-Group-6/notebook.html

**Notebook source code:** https://github.com/ryanmerlino21-debug/CAI5735-Final-Project-Group-6/blob/main/notebook.py (see "Running the notebook" below to open it in Marimo)

**Dataset:** Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository, Strack et al., 2014, https://doi.org/10.24432/C5230J). A simplified 3,000-row copy is in `data/diabetes_130.csv`.

## Files

| File | What it is |
|---|---|
| `notebook.py` | Marimo analysis notebook (split, 3 models, confusion matrices, ROC/AUROC, new patient demo, threshold) |
| `docs/index.html`, `docs/app.css`, `docs/app.js` | The web app (plain HTML, CSS and JavaScript, no build step) served by GitHub Pages |
| `docs/notebook.html` | The notebook exported with its outputs, so it can be read in a browser |
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

To refresh the rendered copy after editing: `marimo export html notebook.py -o docs/notebook.html`.

## Running the app locally

```bash
python3 -m http.server --directory docs
```

Then open http://localhost:8000. If you change the models, rerun `python scripts/export_results.py` to refresh `docs/data.json`.

## How this project meets the guidelines

| Requirement | Where |
|---|---|
| Research question about a binary outcome and 2+ variables | Top of `notebook.py` and the "Research question" tab of the app |
| Marimo notebook with an explanation for every cell | `notebook.py` (a markdown cell or "Why" comment on each cell) |
| Load data, pick variables, train/test split | Notebook sections 1 to 3 |
| 3 classification models (logistic regression, decision tree, random forest) | Notebook section 4 |
| Confusion matrix, accuracy, sensitivity, specificity, ROC and AUROC for each model | Notebook section 5 |
| Classify a new patient with each model | Notebook section 6 and the "Try a patient" tab |
| Compare models and justify the best one | Notebook section 7 and the "Analysis: Models" tab |
| Justify the threshold | Notebook section 8 and the "Analysis: Threshold" tab |
| App on GitHub Pages with question, dataset, process and results | The interactive app link above |
| Dataset source stated | UCI Machine Learning Repository, Strack et al. (2014), https://doi.org/10.24432/C5230J |
| All group members named | README, notebook and app |

## Key result

All three models (logistic regression, decision tree, random forest) reach an AUROC of about 0.59, so these in-stay measurements predict readmission only weakly. We pick the random forest with a 0.40 threshold. See the app for details.
