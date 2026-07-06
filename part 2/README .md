

## Part 2 — Predictive Modeling (`Supervised machine learning.ipynb`)

This part builds a **regression model** (predicts a continuous number) and
a **classification model** (predicts a Yes/No outcome) from `cleaned_data.csv`.

### Target and feature definitions

- **`y_reg` (regression label): `Cholesterol`** — a continuous numeric column.
- **`y_clf` (classification label): `HeartDisease`**, encoded as `1 = Yes`,
  `0 = No` — a natural binary column already present in the data, so we did
  not need to binarize a continuous column at its median.
- **`X` (features):** every other column, **except**:
  - `PatientID` — just an identifier, carries no predictive information
  - `Cholesterol` and `HeartDisease` — these are the two targets, so
    including either as a feature would be leaking the answer into the
    inputs
  - `DiagnosisYear` — over 84% missing (see Part 1/2), and dropped since a
    column that sparse can't reliably contribute, and what little it does
    say is already captured by the `Diabetes` column

### Encoding categorical columns

Our categorical columns are `Sex`, `Smoker`, `BloodType`, `FamilyHistory`,
and `Diabetes`. **None of them have a natural order** — there's no sense
in which one blood type or one Yes/No value is numerically "more" than
another. Because of that, **all five were one-hot encoded** with
`pd.get_dummies(..., drop_first=True)`, resulting in 18 final feature
columns.

**Why not label encoding here?** Label encoding assigns integers like
`BloodType: A+ = 0, A- = 1, AB+ = 2, ...`. A model would then treat those
numbers as having real mathematical meaning — that AB+ is "twice as far"
from A+ as A- is, for example — which is a relationship we'd be inventing,
not one that exists in the data. One-hot encoding avoids this by giving
each category its own independent 0/1 column, so no false ordering or
false distance gets introduced. `drop_first=True` drops one dummy column
per variable to avoid multicollinearity (if you know every other dummy
column's value, the dropped one is already implied).

*(This dataset doesn't contain a genuinely ordinal categorical column, like
a Low/Medium/High rating — if it did, that column would instead get label
encoded with integers that preserve the real order, e.g. Low=0, Medium=1,
High=2.)*

### Leak-free train/test split and scaling

We used one `train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)`
call so the **same** 520 training rows / 130 test rows are used for both
models. `StandardScaler` was **fit only on `X_train`**, then used to
transform both `X_train` and `X_test`.

**Why not fit the scaler on the full dataset?** `StandardScaler` learns
each column's mean and standard deviation and uses those two numbers to
rescale every value. If it were fit on the combined train+test data, the
test set's own statistics would leak into the numbers the model is
trained on — even though the model never sees the test *labels*, it would
still be trained using scaling statistics partly calculated *from* the
test data. That's data leakage, and it makes evaluation results overly
optimistic compared to how the model would perform on truly new,
never-seen data.

### Regression: Linear Regression

| Metric | Value |
|---|---|
| MSE | 1144.87 |
| R² | **−0.064** |

**Top 3 features by absolute coefficient size:**

| Feature | Coefficient |
|---|---|
| BloodType_AB- | +3.39 |
| SystolicBP | −3.26 |
| BloodType_O+ | +2.28 |

**Interpreting the coefficients:** because all features were scaled first,
a coefficient tells us how much the predicted Cholesterol changes for a
**1 standard-deviation increase** in that feature. A large **positive**
coefficient (like `BloodType_AB-` at +3.39) means patients with that blood
type are predicted to have Cholesterol about 3.39 mg/dL higher than the
reference blood type, holding everything else constant. A large
**negative** coefficient (like `SystolicBP` at −3.26) means that as
Systolic BP rises by one standard deviation, predicted Cholesterol drops
by about 3.26 mg/dL, holding everything else constant.

**About that R² of −0.064:** a negative R² means this model performs
*worse* than simply predicting the average Cholesterol for every patient.
This lines up with what Part 2's correlation heat map already showed —
none of these features had any meaningful linear relationship with
Cholesterol (all correlations were close to 0). This dataset simply
doesn't contain a strong linear signal for predicting Cholesterol from
these particular features, and the model result correctly reflects that
rather than being a bug.

### Ridge Regression comparison

| Model | MSE | R² |
|---|---|---|
| Linear Regression (OLS) | 1144.87 | −0.0637 |
| Ridge (alpha=1.0) | 1144.59 | −0.0634 |

**Why might Ridge differ from OLS, and what does `alpha` control?** Ridge
adds a penalty term that discourages large coefficients — the `alpha`
parameter controls how strong that penalty is (higher `alpha` = more
shrinkage toward zero). This tends to produce smaller, more conservative
coefficients than plain OLS, which can help when features are correlated
with each other or when OLS is overfitting noise. Here, the two models
perform almost identically, which makes sense: with no real underlying
signal to overfit to in the first place (see the R² discussion above),
there isn't much for Ridge's penalty to meaningfully correct — so it barely
moves the needle either way.

### Classification: class balance check

| Class (0=No, 1=Yes) | Count | Percentage |
|---|---|---|
| 0 (No) | 279 | 53.65% |
| 1 (Yes) | 241 | 46.35% |

The smaller class (`Yes`, 46.35%) is **above the 35% threshold**, so per
the stated rule, no imbalance correction (SMOTE or `class_weight='balanced'`)
was necessary — the classes are close enough to balanced already. We
trained a standard `LogisticRegression` with default class weighting
rather than applying an unneeded correction.

### Classification: Logistic Regression (baseline, C=1.0)

**Confusion matrix:**

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 60 (TN) | 22 (FP) |
| **Actual Yes** | 16 (FN) | 32 (TP) |

**Classification report:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No | 0.79 | 0.73 | 0.76 |
| Yes | 0.59 | 0.67 | 0.63 |
| **Accuracy** | | | **0.71** |

**AUC: 0.7785** (see `plots/7_roc_curve.png`)

**(a) Precision and Recall formulas:**
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

**(b) Which matters more here?** For heart disease detection, **recall is
more important**. A false negative (FN) means telling a patient who
actually has heart disease that they're fine — a missed diagnosis that
could delay real treatment. A false positive (FP) means flagging a
healthy patient for further review, which costs time and causes worry but
isn't dangerous by itself. Since missing a true case is far more costly
than a false alarm here, we'd rather tolerate more false positives in
exchange for catching more true positives.

**(c) What does AUC = 0.7785 mean?** AUC measures how well the model
separates the two classes across every possible threshold — 1.0 would be
perfect separation, 0.5 would be no better than a coin flip. An AUC of
0.78 means that if you picked one random "Yes" patient and one random "No"
patient, the model would correctly assign the "Yes" patient a higher risk
score about 78% of the time. That's a reasonably good, though not perfect,
level of separation.

### Decision-threshold sensitivity

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.5181 | 0.8958 | 0.6565 |
| 0.40 | 0.5397 | 0.7083 | 0.6126 |
| 0.50 | 0.5926 | 0.6667 | 0.6275 |
| 0.60 | 0.6341 | 0.5417 | 0.5843 |
| 0.70 | 0.7200 | 0.3750 | 0.4932 |

**(a)** Formulas repeated here for convenience:
`Precision = TP / (TP + FP)`, `Recall = TP / (TP + FN)`.

**(b) Threshold that maximizes F1: 0.30** (F1 = 0.6565).

**(c) Precision or recall — which matters more?** As established above,
**recall** — missing a real heart disease case is more costly than a false
alarm.

**(d) Would we raise or lower the threshold, and what's the cost?** We
would **lower** the threshold (toward 0.30) to prioritize recall, since a
lower threshold makes the model flag "Yes" more readily, catching more
true positive cases. The cost is a drop in precision — at threshold 0.30,
recall jumps to 0.896 but precision falls to 0.518, meaning close to half
of the patients the model flags as high-risk will turn out not to have
heart disease. That trade-off is generally acceptable in a screening
context, where a flagged patient gets a follow-up test rather than an
irreversible action, but it does mean more patients would need that
follow-up.

### Regularization experiment (C=1.0 vs C=0.01)

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.5926 | 0.6667 | 0.7785 |
| C=0.01 (strong regularization) | 0.6170 | 0.6042 | 0.7782 |

**What does `C` control?** `C` is the inverse of regularization strength
in `LogisticRegression` — a **smaller** `C` means **stronger**
regularization, penalizing large coefficients more and pushing the model
toward simpler, more conservative decision boundaries; a **larger** `C`
lets the model fit the training data more closely. Here, dropping `C` from
1.0 to 0.01 (a 100x stronger penalty) barely changed anything — AUC moved
by only 0.0003, precision rose slightly, and recall fell slightly. This
tells us the baseline model wasn't overfitting in a way that heavy
regularization could fix; performance on this dataset is essentially
capped by how much real signal the features contain, not by model
complexity.

### Bootstrap confidence interval for the AUC difference

We drew 500 bootstrap samples (with replacement) from the test set,
computed the AUC difference (C=1.0 minus C=0.01) on each sample, and
summarized the results:

| Statistic | Value |
|---|---|
| Mean AUC difference | 0.00042 |
| 95% CI lower bound (2.5th percentile) | −0.00984 |
| 95% CI upper bound (97.5th percentile) | 0.00962 |

**The 95% confidence interval includes zero** (it spans from slightly
negative to slightly positive). This means we **cannot** confidently say
the C=1.0 model's AUC is truly, reliably higher than the C=0.01 model's —
the tiny difference we observed (0.0003) is well within the range you'd
expect from ordinary sampling noise alone. In practice, this confirms what
the regularization experiment already suggested: on this dataset, these
two regularization strengths perform statistically indistinguishably from
one another.

---

## Files in this project

- `heart_disease_dataset.csv` — the original, raw dataset
- `clean_heart_disease_data.py` — Tasks 1–4 (load, nulls, duplicates, dtypes)
- `eda_heart_disease_data.py` — Tasks 5–8c (stats, outliers, visuals, correlation, grouping)
- `modeling_heart_disease.py` — Part 3 (regression, classification, thresholds, bootstrap)
- `heart_disease_dataset_cleaned.csv` — output of Part 1
- `cleaned_data.csv` — output of Part 2, used as the input to Part 3
- `plots/` — all chart images (6 from Part 2, plus the ROC curve from Part 3)
- `README.md` — this file

## How to run it

```bash
python clean_heart_disease_data.py
python eda_heart_disease_data.py
python modeling_heart_disease.py
```
