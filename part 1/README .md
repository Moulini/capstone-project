# Heart Disease Dataset — Cleaning Report

This project takes a raw, messy dataset (`heart_disease_dataset.csv`) and
cleans it up using pandas. This README explains **what we did**, **why we did
it**, and **what the results were**, step by step. It's written for someone
who is new to pandas/data cleaning.

---

## The Dataset

`heart_disease_dataset.csv` contains 670 patient records with 16 columns,
including age, blood pressure, cholesterol, glucose, exercise habits, and
whether the patient has heart disease. It arrived "raw," meaning it has not
been cleaned yet — it has missing values, duplicate rows, and at least one
column stored as the wrong data type.

---

## Task 1: Loading and First Look

We loaded the file with `pd.read_csv('heart_disease_dataset.csv')`.

- **Shape:** 670 rows, 16 columns
- We printed `.head()` (first 5 rows), `.dtypes` (data type of every column),
  and `.shape` (rows, columns) to get a first impression of what we're
  working with.

---

## Task 2: Missing Values (Nulls)

### How we measured it
We used:
```python
df.isnull().sum()                        # count of missing values per column
(df.isnull().sum() / df.shape[0]) * 100   # percentage missing per column
```

### Columns above the 20% threshold (NOT filled in)

| Column        | % Missing |
|---------------|-----------|
| GlucoseMgDl   | 24.33%    |
| DiagnosisYear | 84.93%    |

We chose **not** to fill these in. When a column is missing this much data,
guessing the missing values (even with something like the median) would
mean we're mostly making up data rather than describing real patients. It's
more honest to leave these as missing and note the limitation.

**Special note on `DiagnosisYear`:** this column isn't missing at random —
it's only filled in for patients who actually have diabetes. Patients
without diabetes never got a diagnosis year in the first place, so a high
null percentage here is expected and doesn't indicate a data quality
problem the same way it would for other columns.

### Columns below the 20% threshold (filled in)

| Column             | % Missing | Filled with (median) |
|---------------------|-----------|-----------------------|
| Age                | 0.90%     | median of Age         |
| BMI                | 6.72%     | 27.0                  |
| Cholesterol        | 10.15%    | 201.0                 |
| ExerciseHrsPerWeek | 17.46%    | 1.6                   |

We used:
```python
df[col] = df[col].fillna(df[col].median())
```

### Why the median instead of the mean?

The **mean** (average) is sensitive to outliers — a small number of
unusually high or low values can pull it in one direction. In medical data
especially, a handful of patients with extreme readings (e.g. very high
cholesterol) can drag the mean upward, so filling missing values with the
mean would push those values toward an unrealistic "typical" patient.

The **median** is the middle value once everything is sorted, so it isn't
affected by how extreme the outliers are — only by how many values are
above or below it. That makes it a more robust and realistic stand-in for
a "typical" value, which is exactly what we want when guessing a missing
number.

---

## Task 3: Duplicate Rows

### How we measured it
```python
df.duplicated().sum()
```
**Result:** 20 duplicate rows were found (rows that were an exact copy of
another row).

### Removing them
```python
df = df.drop_duplicates()
```
- Rows before: 670
- Rows after: 650
- **Rows removed: 20**

### Did this change the null percentages?

Yes, slightly. Removing rows changes the total row count (`df.shape[0]`),
which is the denominator in the null percentage formula, so most
percentages shifted a small amount even though the actual number of
missing values per column didn't change. For example:

| Column        | Null % Before | Null % After |
|---------------|---------------|--------------|
| Age           | 0.90%         | 0.77%        |
| GlucoseMgDl   | 24.33%        | 24.00%       |
| DiagnosisYear | 84.93%        | 85.08%       |

Columns with zero missing values stayed at 0% either way.

---

## Task 4: Fixing Data Types

### Memory usage before cleanup
```python
df.memory_usage(deep=True).sum()
```
**Result: 317,966 bytes (≈ 310.5 KB)**

### Problem 1: `Age` was stored as text (object), not numbers

A handful of rows had text values like `"unknown"`, `"N/A"`, and
`"not recorded"` typed into the Age column instead of a number. Because a
column can only have ONE data type in pandas, having even a few text
entries forces pandas to treat the **entire column** as text (`object`
dtype) instead of numbers.

**Fix:**
```python
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
```
- `pd.to_numeric()` converts values to numbers.
- `errors='coerce'` tells pandas: if a value can't be converted (like the
  word "unknown"), turn it into `NaN` (missing) instead of crashing.
- We then filled the new missing values with the median Age, the same way
  we did in Task 2.

**Result:** `Age` changed from `object` to `float64`.

### Problem 2: Repetitive text columns wasting memory

Columns like `Sex`, `Smoker`, `BloodType`, `FamilyHistory`, `Diabetes`, and
`HeartDisease` only contain a small handful of repeated values (e.g.
`"Yes"`/`"No"`). Storing them as plain text means pandas re-stores the full
text for every single row, which wastes memory.

**Fix:**
```python
for col in ['Sex', 'Smoker', 'BloodType', 'FamilyHistory', 'Diabetes', 'HeartDisease']:
    df[col] = df[col].astype('category')
```
The `category` dtype stores each unique value once and just points each row
to it, which is far more memory-efficient for columns with few repeated
values.

### Memory usage after cleanup

**Result: 93,230 bytes (≈ 91.0 KB)**

**Memory saved: 224,736 bytes — a 70.7% reduction**

### Task 5: Descriptive Statistics and Skewness

`df.describe()` was run on all 9 numeric columns (Age, BMI, SystolicBP,
DiastolicBP, Cholesterol, GlucoseMgDl, RestingHR, ExerciseHrsPerWeek,
DiagnosisYear).

**Skewness of each column (sorted by strength):**

| Column | Skew |
|---|---|
| ExerciseHrsPerWeek | **+1.860** |
| GlucoseMgDl | +0.258 |
| DiagnosisYear | +0.159 |
| RestingHR | +0.140 |
| DiastolicBP | −0.136 |
| Age | −0.112 |
| BMI | +0.106 |
| SystolicBP | +0.045 |
| Cholesterol | +0.004 |

**Most skewed column: `ExerciseHrsPerWeek`** (skew ≈ 1.86).

**What does this skew mean?**
`ExerciseHrsPerWeek` has **positive skew**: most patients exercise a small
number of hours per week, but a small number of very active patients
exercise a lot more, stretching the distribution's tail out to the right.
Almost every other column here has skew very close to 0, meaning they're
roughly symmetric (bell-shaped) — no meaningfully long tail in either
direction.

**Consequence for mean imputation:** in a positively skewed column, the
mean gets dragged upward by that handful of high-exercise patients, so it
no longer represents what a "typical" patient looks like. If you filled
missing values with the mean here, you'd be inserting a value that's
higher than most real patients actually report. The median isn't affected
by how far out the tail stretches, only by how many points are on each
side of it, so it's the safer choice for a skewed column — which is
exactly the logic used in Task 8a below.

### Task 6: Outlier Detection with IQR

We used the standard rule: outliers are values more than 1.5×IQR below Q1
or above Q3.

| Column | Q1 | Q3 | IQR | Valid Range | Outliers |
|---|---|---|---|---|---|
| ExerciseHrsPerWeek | 0.60 | 3.30 | 2.70 | [−3.45, 7.35] | **23 rows** |
| Cholesterol | 178.00 | 227.00 | 49.00 | [104.50, 300.50] | **2 rows** |

**We did not drop these outliers.** For `ExerciseHrsPerWeek`, the "outlier"
values (people exercising a lot per week) are realistic, not data-entry
errors — dropping them would throw away real high-activity patients and
bias the dataset toward sedentary ones. For `Cholesterol`, only 2 rows are
flagged, and clinically-extreme cholesterol readings are plausible in a
real patient population.

 we will **retain** both sets of outliers as
they stand, rather than dropping or capping them. If a model later turns
out to be overly sensitive to these extreme values (e.g. a linear model
with unstable coefficients), we'd revisit and consider capping
(winsorizing) `ExerciseHrsPerWeek` at the upper bound rather than removing
rows outright, since removing rows would lose otherwise-valid patient
records.

### Task 7: Visualizations

All charts are saved in the `plots/` folder.

1. **`1_line_plot_cholesterol.png`** — Cholesterol values plotted against
   row index. Since row order has no real meaning here (each row is just a
   patient record, not a time series), this mainly shows that Cholesterol
   bounces around a stable middle range with no obvious trend or drift
   across the dataset — a good sanity check that there's no ordering bug.

2. **`2_bar_chart_cholesterol_by_bloodtype.png`** — Average Cholesterol by
   Blood Type. Averages cluster tightly between roughly 198–206 mg/dL for
   most blood types, with `AB-` noticeably higher at ~236 mg/dL (based on a
   small sample size for that group, so treat it cautiously).

3. **`3_histogram_most_skewed.png`** — Histogram of `ExerciseHrsPerWeek`
   (the most skewed column). The shape is a classic **right-skewed**
   distribution: a tall bar near zero hours, a rapid drop-off, and a long
   thin tail stretching out toward 15 hours/week. This visually confirms
   the positive skew number from Task 5.

4. **`4_scatter_systolic_vs_diastolic.png`** — Systolic vs. Diastolic
   blood pressure. **Correlation: r ≈ 0.038 (essentially no relationship
   in this data).** Clinically, these two numbers usually move together
   in real patients. Here, because the dataset was generated with each
   column drawn independently, that expected relationship doesn't show up
   — a good reminder to always check a scatter plot rather than assuming
   a relationship exists just because it "should," and evidence that this
   is a synthetic rather than a real clinical dataset.

5. **`5_boxplot_cholesterol_by_heartdisease.png`** — Cholesterol split by
   Heart Disease status. Patients with heart disease have a slightly
   higher median Cholesterol than those without, and the two boxes overlap
   substantially — the difference is real but modest, not a dramatic
   separation.

### Task 8 Correlation Heat Map

**`correlation_heatmap.png`** — Pearson correlation across all 9 numeric
columns.

**Strongest pair: `DiastolicBP` & `DiagnosisYear`, r ≈ 0.260.**

**Is this likely causal?** Almost certainly not, for two reasons: (1) it's
the *only* moderately-sized correlation in the entire matrix — everything
else is close to 0 — which is the pattern you'd expect from random noise
occasionally producing one larger-looking number rather than a real
relationship; and (2) `DiagnosisYear` only has 97 non-missing values (it's
84.93% null), and correlations computed from small samples are naturally
less stable and more likely to look artificially strong by chance. A
plausible alternative explanation is simply **sampling noise from a small
subgroup** — there's no clinical mechanism linking a patient's diastolic
blood pressure to the calendar year they were diagnosed with diabetes, so
this is a good example of a correlation to treat with skepticism rather
than a discovery.

---

### Task 9a: Imputation Strategy Comparison

The two most-skewed columns from Task 5 are `ExerciseHrsPerWeek` and
`GlucoseMgDl`.

| Column | Mean (before fill) | Median (before fill) | Skew | Nulls before fill |
|---|---|---|---|---|
| ExerciseHrsPerWeek | 2.34 | **1.60** | +1.86 | 117 |
| GlucoseMgDl | 99.95 | **99.00** | +0.26 | 156 |

**Chosen strategy: median, for both columns.** Both columns are positively
skewed, meaning their means are pulled upward by a smaller number of
high-value patients (heavy exercisers / higher glucose readings). That
makes the mean an unrepresentative stand-in for a typical patient, so we
filled missing values with the median instead, using `fillna()`.

After filling, `df[['ExerciseHrsPerWeek', 'GlucoseMgDl']].isnull().sum()`
confirms **0 nulls remain** in both columns.

*(Note: `GlucoseMgDl` is one of the two most-skewed columns and was
therefore filled here, even though it's above the 20% missing threshold
from Task 2 — Task 9a's skew-based imputation takes priority over the
general "don't fill >20%" rule for these two specific columns.
`DiagnosisYear`, which is also above 20% missing but wasn't among the
top-2 skewed columns, was left unfilled and is reported as-is.)*

### Task 9b: Spearman Rank Correlation

Spearman correlation ranks each column's values first, then measures the
relationship between the ranks — this catches relationships that are
*consistent* (values tend to move together) even if they're not a straight
line, which Pearson can miss.

**Top 3 pairs where Spearman and Pearson disagree the most:**

| Column A | Column B | \|Spearman − Pearson\| | Pearson r | Spearman r |
|---|---|---|---|---|
| DiagnosisYear | RestingHR | 0.057 | 0.099 | 0.042 |
| DiagnosisYear | ExerciseHrsPerWeek | 0.038 | 0.031 | 0.068 |
| GlucoseMgDl | ExerciseHrsPerWeek | 0.037 | −0.004 | 0.033 |

For all three pairs, both the Pearson and Spearman values are very close
to zero, and the gap between them is tiny (≤ 0.06). This means:

(a) None of these three pairs show a real monotonic-but-non-linear
relationship — the differences here are small enough to be noise rather
than a meaningful signal either measure is "catching" that the other
misses. If anything, `DiagnosisYear` & `ExerciseHrsPerWeek` has
|Spearman| slightly greater than |Pearson| (0.068 vs 0.031), which would
technically suggest a mild non-linear tendency, but both values are too
close to zero to treat as a real pattern.

(b) **For feature-selection guidance , we'll rely on Pearson as
the primary measure**, since none of these variables show the kind of
strong rank-based-but-not-linear pattern that would justify preferring
Spearman, and Pearson is simpler to interpret alongside a linear model. We
would switch to leaning on Spearman only for a variable pair where the two
measures diverge by a large, consistent margin — which isn't the case
anywhere in this dataset.

### Task 9c: Grouped Aggregation

Grouped `Cholesterol` (numeric) by `HeartDisease` (categorical):

| HeartDisease | mean | std | count |
|---|---|---|---|
| No | 197.11 | 34.52 | 361 |
| Yes | 205.76 | 34.90 | 289 |

(a) **Highest mean:** `HeartDisease = Yes` group (205.76 mg/dL).
**Highest std (spread):** also the `Yes` group (34.90), though it's only
marginally higher than the `No` group's 34.52.

(b) **Is high within-group std a concern?** Yes, somewhat. A standard
deviation of ~35 mg/dL within *both* groups is fairly large relative to
the ~9 mg/dL gap between their means — this tells us Cholesterol varies a
lot among patients who share the same heart disease status, so knowing
someone's Cholesterol level alone wouldn't reliably predict their heart
disease status; there's a lot of overlap between the two groups.

(c) **Ratio of highest group mean to lowest group mean:** 205.76 / 197.11
= **1.044**. A ratio this close to 1.0 suggests only a weak separation
between the groups — Cholesterol by itself likely carries limited
predictive signal for heart disease in this dataset, and would need to be
combined with other features (age, smoking, blood pressure, etc.) to be
useful in a model.

---

### Task 10 Final Output

`cleaned_data.csv` — the fully cleaned dataset (650 rows × 16 columns),
saved with `df.to_csv('cleaned_data.csv', index=False)`. Every numeric
column has 0 nulls except `DiagnosisYear`, which is intentionally left as
84.9% missing per the reasoning in Task 2 / 9a above.


## Files in this project

- `heart_disease_dataset.csv` — the original, raw dataset
- `cleaned_data.csv` — final output of Part 1
- `plots/` — all 6 chart images from Task 7
- `README.md` — this file


---




