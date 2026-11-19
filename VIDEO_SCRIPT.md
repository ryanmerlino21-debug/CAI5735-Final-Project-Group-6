# Video Script (about 5 minutes)

**CAI5735 Final Project - Group 6**
**Speakers:** Ryan Merlino, April Walter, Nikolas Platon (everyone must speak)

**App to screen share:** https://ryanmerlino21-debug.github.io/CAI5735-Final-Project-Group-6/

The assignment needs three parts: an **introduction** (group members and research question), an **app walkthrough** (someone in the group walking through the app), and the **results** (the answer to the research question). Speaking pace is about 140 words per minute, so this script is around 700 words. The speaker names are only a suggestion, so feel free to swap parts.

---

## Part 1: Introduction (0:00 to 0:50)

*On screen: the Research question tab of the app.*

**RYAN:** Hi everyone. We are Group 6 for CAI5735. I am Ryan Merlino.

**APRIL:** I am April Walter.

**NIKOLAS:** And I am Nikolas Platon.

**RYAN:** For our final project we used the Diabetes 130 US Hospitals dataset from the UCI Machine Learning Repository. It has 3,000 hospital stays from patients with diabetes. Our research question is: how accurately can a patient's length of stay, number of lab tests, procedures, medications and diagnoses, plus whether they were prescribed a diabetes medication, predict whether they will be readmitted to the hospital?

**APRIL:** We chose this because readmissions are hard on patients and costly for hospitals. If we could flag high risk patients before they go home, staff could focus follow up calls on them first. So we wanted to see how far a few simple numbers from the stay could take us.

---

## Part 2: App walkthrough (0:50 to 3:30)

### Dataset and Explore tabs (0:50 to 1:40), Nikolas

*On screen: click the Dataset tab, then the Explore tab.*

**NIKOLAS:** I will walk you through the app. On the Dataset tab you can see where the data came from and the columns we used. The response is readmitted or not. We combined the "within 30 days" and "after 30 days" groups, so about 40 percent of stays are readmitted. There is also a preview where you can browse real rows and shuffle them.

*On screen: Explore tab. Pick "Number of diagnoses", then switch the chart type.*

**NIKOLAS:** On the Explore tab, every chart has buttons to change how it is drawn. For example, here is the distribution of diagnoses for readmitted and not readmitted patients. I can switch to a box plot or a cumulative curve. The important thing is that the two groups look almost the same. Under every chart there is a note that explains how to read it and what it says.

### Models and evaluation (1:40 to 2:40), April

*On screen: Analysis: Models tab.*

**APRIL:** Next is our analysis. We split the data into 75 percent for training and 25 percent for testing, then trained three models: logistic regression, a decision tree, and a random forest.

*On screen: point at the confusion matrix and the ROC curve.*

**APRIL:** For each model the app shows a confusion matrix, accuracy, sensitivity, specificity, and the ROC curve with the AUROC. All three AUROC values are about 0.59, which is only a little better than a coin flip at 0.50. The random forest scored highest, so we chose it. But the bootstrap check at the bottom shows the three models are really too close to separate.

*On screen: Analysis: Threshold tab. Drag the slider from 0.50 to 0.40.*

**APRIL:** On the Threshold tab you can drag the threshold and watch the numbers change. At 0.50 the random forest only catches about 21 percent of readmissions. At 0.40 it catches about 66 percent, but it also flags more patients who did not come back.

### Try a patient (2:40 to 3:30), Ryan

*On screen: Try a patient tab.*

**RYAN:** The app also lets you classify a new patient. I can set the days in the hospital, lab tests, procedures, medications and diagnoses, then see each model's predicted chance of readmission. Here I will pick a long, complex stay. You can see all three models give a probability, and whether it is above our threshold. The "what if" chart shows what happens when I change just one variable. The lines are fairly flat, which shows how weak each variable is on its own.

---

## Part 3: Results (3:30 to 5:00)

*On screen: the Results tab.*

**NIKOLAS:** So what is the answer to our research question? These variables do not predict readmission very accurately. Our best model, the random forest, had an AUROC of about 0.59. Its accuracy at a threshold of 0.5 was about 61 percent, and always guessing "not readmitted" gets 60 percent.

**APRIL:** We picked a threshold of 0.40. Missing a patient who comes back is worse than making an extra follow up call, so we wanted higher sensitivity. At 0.40 the random forest catches about two thirds of readmissions, but it also flags about half of the patients who did not come back.

**RYAN:** The number of diagnoses mattered most to the models, but even that did not separate the groups well. Things like prior admissions, where the patient goes after discharge, and lab results such as HbA1c would probably help more, but they are not in this dataset. Our limits are the small sample of 3,000 stays, one random split, and data from 1999 to 2008.

**NIKOLAS:** So our conclusion is that these in-stay numbers are only a weak signal. The model could be a rough screening tool, but it should not be used alone to make decisions about a patient.

**ALL:** Thank you for watching. Our code, notebook and app links are on GitHub and in our discussion post.

---

## Recording checklist

- [ ] Everyone speaks at least once (this script has all three of us in every part).
- [ ] Open the app first and use a full screen browser window. Zoom in a little so text is easy to read.
- [ ] Practice once with a timer. The video should be **about 5 minutes** (4:30 to 5:30 is fine).
- [ ] Record the screen with sound (Zoom, QuickTime or Loom all work).
- [ ] Upload to **YouTube** and set visibility to **Unlisted** (or Public) so anyone with the link can watch **without signing in**. Test the link in a private window.
- [ ] Put the **video link in the discussion post** along with the app link, all three names, and a short project description.

## Numbers to double check on screen

| Item | Value |
|---|---|
| Stays in the dataset | 3,000 (about 40% readmitted) |
| Train / test split | 2,250 / 750 |
| AUROC | Logistic regression 0.587, decision tree 0.586, random forest 0.590 |
| Random forest at threshold 0.50 | accuracy 0.613, sensitivity 0.213, specificity 0.880 |
| Random forest at threshold 0.40 | accuracy 0.552, sensitivity 0.657, specificity 0.482 |
| Baseline (always guess not readmitted) | about 60% |
