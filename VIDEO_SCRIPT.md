# Video Script

CAI5735 Final Project, Group 6 · about 5 minutes total

Each person records their own part (screen and voice), about 1 minute 40 seconds each. Then the three clips are put together in order: Ryan, April, Nikolas.

App: https://ryanmerlino21-debug.github.io/CAI5735-Final-Project-Group-6/

---

## Part 1: Ryan

**Introduction and dataset** (about 1:40)

*Show: the Research question tab.*

Hi everyone. I'm Ryan Merlino. This is our CAI5735 final project from Group 6. My group members are April Walter and Nikolas Platon, and each of us will walk through one part.

Our research question is: how accurately can a patient's length of stay, number of lab tests, procedures, medications and diagnoses, plus whether they were prescribed a diabetes medication, predict whether they will be readmitted to the hospital?

We chose this question because readmissions are hard on patients and costly for hospitals. If we could flag high risk patients before they go home, staff could call those patients first.

*Show: the Dataset tab.*

We used the Diabetes 130 US Hospitals dataset from the UCI Machine Learning Repository. Our copy has 3,000 hospital stays.

This table lists the columns we used. What we want to predict is whether the patient was readmitted. About 40 percent of the stays were. You can also browse real rows in the data preview.

Now April will show how we analyzed the data.

---

## Part 2: April

**Analysis: exploring, models and threshold** (about 1:50)

*Show: the Explore tab. Pick "Number of diagnoses" and switch the chart type.*

Hi, I'm April Walter. I'll walk through our analysis.

On the Explore tab you can look at every variable. Each chart has buttons to change how it is drawn. For example, here are the diagnoses for readmitted and not readmitted patients. The two groups look almost the same. That is an early sign this will be hard to predict.

*Show: the Analysis: Models tab. Point at the confusion matrix and the ROC curve.*

Next is the Models tab. We split the data into 75 percent for training and 25 percent for testing. Then we trained three models: logistic regression, a decision tree, and a random forest.

For each model the app shows a confusion matrix, accuracy, sensitivity, specificity, and the ROC curve with the AUROC. All three AUROC values are about 0.59. A coin flip would be 0.50. The random forest scored highest, so we chose it, but the three models are very close.

*Show: the Analysis: Threshold tab. Drag the slider from 0.50 to 0.40.*

The Threshold tab shows how the cutoff changes the results. At 0.50 the random forest only catches about 21 percent of readmissions. At 0.40 it catches about 66 percent, but it also flags more patients who did not come back.

We chose 0.40 because missing a patient who comes back is worse than making an extra follow up call.

Now Nikolas will classify a new patient and share our results.

---

## Part 3: Nikolas

**Classifying a patient and results** (about 1:40)

*Show: the Try a patient tab. Click a preset such as "Long, complex stay".*

Hi, I'm Nikolas Platon. On the Try a patient tab you can classify a new patient. You set the days in the hospital, lab tests, procedures, medications, diagnoses, and whether they were prescribed a diabetes medication.

Here I'll pick a long, complex stay. Each model gives a predicted chance of readmission and tells us if it is above our threshold. The what-if chart changes one variable at a time. The lines are fairly flat, which shows each variable is weak on its own.

*Show: the Results tab.*

So what is the answer to our research question? These variables do not predict readmission very accurately. Our best model, the random forest, had an AUROC of about 0.59. Its accuracy at 0.50 was about 61 percent, and always guessing "not readmitted" gets 60 percent.

The number of diagnoses mattered most, but even that did not separate the groups well. Things like prior admissions and lab results such as HbA1c would probably help more, but they are not in this dataset. Our limits are a small sample, one random split, and data from 1999 to 2008.

Our conclusion is that these numbers are only a weak signal. The model could be a rough screening tool, but it should not be used alone to make decisions about a patient.

Thank you for watching.
