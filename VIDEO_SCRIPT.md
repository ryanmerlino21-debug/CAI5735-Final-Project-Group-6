# Video Script

CAI5735 Final Project, Group 6 · about 5 minutes total

Each person records their own part (screen and voice), about 1 to 2 minutes each. Then the three clips are put together in order: Ryan, April, Nikolas.

App: https://ryanmerlino21-debug.github.io/CAI5735-Final-Project-Group-6/

---

## Part 1: Ryan

**Introduction and dataset** (about 1:15)

*Start on the Research question tab (it is the first page). Scroll slowly down to the box titled "Research question".*

Hi everyone. I'm Ryan Merlino. This is our CAI5735 final project from Group 6. My group members are April Walter and Nikolas Platon, and each of us will walk through one part.

Our research question is: how accurately can a patient's length of stay, number of lab tests, procedures, medications and diagnoses, plus whether they were prescribed a diabetes medication, predict whether they will be readmitted to the hospital?

*Scroll down to "What problem are we trying to solve?".*

We chose this question because readmissions are hard on patients and costly for hospitals. If we could flag high risk patients before they go home, staff could call those patients first.

*Click "Dataset" in the left menu. Point at the "Data at a glance" boxes at the top (3,000 rows).*

We used the Diabetes 130 US Hospitals dataset from the UCI Machine Learning Repository. Our copy has 3,000 hospital stays.

*Scroll to "Columns we used" and move your mouse down the table.*

This table lists the columns we used. What we want to predict is whether the patient was readmitted.

*Scroll to "Setting up the outcome". On the "Outcome groups" chart, click "Original 3", then "Our 2", then "100 patients".*

About 40 percent of the stays were readmitted.

*Scroll to "Browse the data" and click the "Shuffle" button once.*

You can also browse real rows in the data preview.

Now April will show how we analyzed the data.

---

## Part 2: April

**Models, threshold and the interactive explorer** (about 2:10)

*Click "Analysis: Models" in the left menu. Point at the three model boxes under "The three models".*

Hi, I'm April Walter. I'll walk through our analysis. We split the data into 75 percent for training and 25 percent for testing. Then we trained three models: logistic regression, a decision tree, and a random forest.

*Scroll to the "Confusion matrix" chart. Click "Counts", then "All 3 models".*

For each model the app shows a confusion matrix.

*Scroll to the "Performance curve" chart on the right. Leave it on "ROC" and "All three". Point at the legend with the AUROC numbers.*

It also shows the ROC curve with the AUROC. All three AUROC values are about 0.59. A coin flip would be 0.50.

*Scroll to "Comparison table" and point at the accuracy, sensitivity, specificity and AUROC columns.*

The random forest scored highest, so we chose it, but the three models are very close.

*Click "Analysis: Threshold" in the left menu. At the top, make sure the Model box says "Random forest". Click the "Default 0.50" button and point at the "In plain terms" box.*

The Threshold tab shows how the cutoff changes the results. At 0.50 the random forest only catches about 21 percent of readmissions.

*Click the "Our choice 0.40" button and point at the "In plain terms" box again.*

At 0.40 it catches about 66 percent, but it also flags more patients who did not come back. We chose 0.40 because missing a patient who comes back is worse than making an extra follow up call.

*Click "Interactive explorer" in the left menu. In the box "1. Filter the patients", click the two tallest bars on the right side under "Diagnoses (count per stay)". Point at the numbers across the top.*

Our last tab is the Interactive explorer, where every chart is linked. Here I keep only patients with many diagnoses, and every chart updates to just those patients.

*In "3. ROC curve", click a point in the middle of the curve. Point at the confusion matrix.*

If I click the ROC curve, it sets the threshold, and the confusion matrix and the numbers change too. Now Nikolas will pick a patient and share our results.

---

## Part 3: Nikolas

**A single patient and results** (about 1:30)

*Click "Interactive explorer" in the left menu. Click "Reset everything". Scroll to "5. Every test patient" and click any dot. Scroll to "6. The selected patient" and point at the three prediction boxes.*

Hi, I'm Nikolas Platon. In the explorer I can click any dot to pick a real test patient. Here is each model's predicted chance of readmission for this patient, and whether it was right.

*Click the "Open in Try a patient" button. Point at the sliders and the three prediction boxes.*

This sends the patient to the Try a patient tab, where I can change the days, lab tests, procedures, medications and diagnoses myself.

*Scroll to the "What if?" chart. Open the "Vary" dropdown and pick "Diagnoses".*

The what-if chart changes one variable at a time. The lines are fairly flat, which shows each variable is weak on its own.

*Click "Results" in the left menu. Point at the highlighted "The answer" box and the table below it.*

So what is the answer to our research question? These variables do not predict readmission very accurately. Our best model, the random forest, had an AUROC of about 0.59. Its accuracy at 0.50 was about 61 percent, and always guessing "not readmitted" gets 60 percent.

*Scroll to the "Scoreboard" chart. Leave the metric on "Sensitivity" and point at the two bars for each model.*

At the lower threshold every model catches more readmissions.

*Scroll to "What this means", then "Limitations".*

The number of diagnoses mattered most, but even that did not separate the groups well. Things like prior admissions and lab results such as HbA1c would probably help more, but they are not in this dataset. Our limits are a small sample, one random split, and data from 1999 to 2008.

Our conclusion is that these numbers are only a weak signal. The model could be a rough screening tool, but it should not be used alone to make decisions about a patient.

Thank you for watching.
