

## Part 3 Advanced — Ensembles, Tuning, and Full ML Pipeline (`advanced_modeling_heart_disease.py`)

This section builds on the classification problem above (predicting
`HeartDisease`) with tree-based models, ensembles, systematic tuning, and
a saved, reloadable pipeline.

### Task 1 & 2: Decision Tree — unconstrained vs controlled

| Model | Training Accuracy | Test Accuracy | Train−Test Gap |
|---|---|---|---|
| Unconstrained (max_depth=None) | 1.0000 | 0.6538 | **0.3462** |
| Controlled (max_depth=5, min_samples_split=20) | 0.7385 | 0.6077 | **0.1308** |

**Does the unconstrained tree overfit?** Clearly yes — it reaches **100%**
training accuracy, meaning it perfectly memorized every training patient,
while test accuracy is only 65.4%. That 34.6-point gap is the signature of
overfitting: the tree learned quirks specific to the training rows rather
than a generalizable pattern.

**Why are decision trees "high-variance" models?** At each split, a
decision tree greedily picks whatever rule best separates the data *right
now*, without ever reconsidering earlier splits or asking whether that
rule will generalize. Left unconstrained, it keeps splitting until every
leaf is pure (or has just one sample), which means it can carve out a
rule custom-fit to a single training row. Because those ultra-specific
rules depend heavily on exactly which rows happened to be in the training
set, a tree trained on a slightly different sample of the same population
could end up looking very different — that sensitivity to the training
sample is what "high variance" means.

**Effect of `max_depth` and `min_samples_split`:** `max_depth=5` stops the
tree from growing past 5 levels, which limits how specific/complex its
rules can get — trading away some accuracy on training data (its accuracy
drops from 1.00 to 0.7385) in exchange for rules that generalize better.
`min_samples_split=20` prevents the tree from splitting a group of
patients further once fewer than 20 remain in that group, which stops it
from creating splits based on tiny, noise-prone subsets. Together, they
shrink the train−test gap from 0.346 down to 0.131 — a clear reduction in
overfitting, even though the controlled tree's raw test accuracy dipped
slightly (0.6538 → 0.6077) on this particular split. That's an expected
bias/variance trade-off: we accepted a bit more bias (simpler rules, less
perfect fit) in exchange for meaningfully less variance (a much smaller
gap between how it performs on seen vs. unseen data).

### Task 3: Gini vs Entropy

**Gini impurity formula:**
```
Gini = 1 − Σ(pᵢ²)
```
**Entropy formula:**
```
Entropy = −Σ(pᵢ × log2(pᵢ))
```
(where pᵢ is the proportion of samples in a node belonging to class *i*)

**What does Gini = 0 mean?** A node with Gini impurity of 0 is perfectly
"pure" — every single sample in that node belongs to the same class. Since
one class has proportion pᵢ = 1 and the other has pᵢ = 0, the formula
becomes `1 − (1² + 0²) = 0`.

**Results:** both criteria produced **identical test accuracy: 0.5923**
at `max_depth=5` on this dataset — Gini and Entropy are different formulas
for measuring impurity, but in practice they very often pick the same
splits (or splits that lead to the same downstream accuracy), which is
what we see here.

### Task 4: Random Forest

| Metric | Value |
|---|---|
| Training Accuracy | 0.9962 |
| Test Accuracy | 0.6615 |
| Test ROC-AUC | 0.6964 |

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| Age | 0.1580 |
| SystolicBP | 0.1386 |
| BMI | 0.1340 |
| RestingHR | 0.1056 |
| DiastolicBP | 0.0967 |

**How does Random Forest compute feature importance, and how is that
different from a regression coefficient?** For each feature, the forest
looks at every split (across every tree) that used that feature, and
measures how much that split reduced Gini impurity — i.e. how much purer
the resulting groups became. These reductions are averaged across all
trees to produce one importance score per feature. This is fundamentally
different from a linear regression coefficient: a coefficient tells you
the size and *direction* (positive/negative) of a feature's estimated
linear effect on the outcome, assuming a straight-line relationship.
Feature importance instead just tells you how *useful* a feature was for
splitting the data well, with no direction and no assumption that the
relationship is a straight line — a feature can have high importance
through a complex, non-linear pattern that a linear coefficient could
never capture.

**Bagging, in one paragraph:** Random Forest builds many individual
decision trees, but forces them to be different from each other in two
ways: (1) each tree is trained on a *bootstrap sample* — a random sample
of the training rows drawn **with replacement**, so some patients appear
multiple times in a given tree's training data and others don't appear at
all; and (2) at each split, the tree is only allowed to consider a random
subset of √(number of features) features, rather than all of them. This
forces the trees to find different, sometimes weaker, patterns rather
than all converging on the exact same greedy splits. Averaging predictions
across many such "diverse" trees smooths out each individual tree's
idiosyncratic overfitting — mistakes that come from one tree memorizing
noise tend to cancel out across the ensemble, which is why the forest as
a whole has much lower variance than any single unconstrained tree.

### Task 4a: Gradient Boosting

| Metric | Value |
|---|---|
| Training Accuracy | 0.9096 |
| Test Accuracy | 0.6846 |
| Test ROC-AUC | 0.7299 |

Gradient Boosting is included in the cross-validated comparison in Task 5
below.

### Task 4b: Feature Ablation Study

The 5 lowest-importance features from the Random Forest were all
**BloodType dummy columns**: `BloodType_A-`, `BloodType_B+`,
`BloodType_AB+`, `BloodType_B-`, `BloodType_AB-`.

| Model | Test AUC |
|---|---|
| Full model (18 features) | 0.6964 |
| Reduced model (13 features, 5 removed) | **0.7002** |

**Were these features genuinely uninformative?** Yes — removing them
didn't hurt performance at all; AUC actually rose very slightly (0.6964 →
0.7002). This confirms these 5 blood-type indicators were contributing
essentially nothing but noise to the model, consistent with there being
no real clinical reason blood type would predict heart disease in this
data.

**What does this imply for production deployment?** Since dropping these
5 features caused no meaningful AUC loss (if anything, a tiny
improvement), deploying the simpler, 13-feature model would be the
better choice in practice: fewer features to collect and validate at
inference time, a smaller/faster model, and less ongoing maintenance
burden (one less set of columns that could break if the data pipeline
changes upstream) — all with no accuracy cost. In general, this trade-off
is only acceptable when the AUC drop from removing features is at or
below a small, pre-agreed tolerance; here there was no drop at all, so
it's an easy call.

### Task 5: Cross-Validated Comparison

| Model | Mean 5-fold CV AUC | Std CV AUC |
|---|---|---|
| Logistic Regression | 0.7546 | 0.0376 |
| Decision Tree (max_depth=5) | 0.6218 | 0.0326 |
| Random Forest | 0.7402 | 0.0235 |
| Gradient Boosting | 0.7272 | 0.0084 |

**Why is cross-validation more reliable than one train/test split?** A
single split gives you exactly one estimate of performance, and that
estimate depends partly on which specific patients happened to land in
the test set — an "easy" or "hard" test set by chance can make a model
look better or worse than it really is. Cross-validation instead trains
and tests the model 5 separate times, using a different 20% slice as the
test set each time, so every patient gets to be in the test set exactly
once across the 5 rounds. Averaging across all 5 gives an estimate that
isn't overly dependent on any single lucky or unlucky split, and the
standard deviation across folds also tells us how *consistent* that
performance is — a model with a small std (like Gradient Boosting's
0.0084) is behaving predictably across different data samples, while a
larger std (like Logistic Regression's 0.0376) means performance varies
more from fold to fold.

### Task 6: Hyperparameter Tuning with GridSearchCV

**Best parameters found:**
```
max_depth: 10
min_samples_leaf: 5
n_estimators: 200
```
**Best cross-validated AUC: 0.7665**
**Test-set AUC of the tuned pipeline: 0.7022**

**How many configurations were evaluated?** The grid has 3 choices for
`n_estimators` × 3 choices for `max_depth` × 2 choices for
`min_samples_leaf` = **18 total configurations**. With 5-fold
cross-validation, that's **18 × 5 = 90 individual model fits** in total.

**Grid Search vs Randomized Search trade-off:** Grid Search is
*exhaustive* — it tries every single combination in the grid, guaranteeing
it finds the best combination *within that grid*, but the number of fits
grows multiplicatively as you add more parameters or more values per
parameter, which can get slow fast. Randomized Search instead samples a
fixed number of random combinations from the specified ranges — it won't
guarantee finding the single best combination, but it can explore a much
wider range of values in the same amount of computation time, which tends
to work better when you have many hyperparameters or wide ranges to
search and can't afford to try every combination exhaustively.

### Task 7: Manual Learning Curve

| Training Fraction | Rows Used | Training AUC | Test AUC |
|---|---|---|---|
| 20% | 104 | 0.9759 | 0.6623 |
| 40% | 208 | 0.9782 | 0.6931 |
| 60% | 312 | 0.9848 | 0.7096 |
| 80% | 416 | 0.9801 | 0.7066 |
| 100% | 520 | 0.9812 | 0.7022 |

**(i) Does training AUC decrease as the training set grows?** No — it
stays consistently very high (between 0.976 and 0.985) at every size.
For a genuinely high-variance model we'd normally expect training
performance to *drop* somewhat as more, more-varied data makes it harder
to fit every single row perfectly. Here it doesn't drop, which tells us
the tuned Random Forest (`max_depth=10`, `min_samples_leaf=5`) still has
enough capacity to fit almost all of the training data closely, even as
that training data grows to its full size.

**(ii) Does test AUC increase with more data?** It rises from 0.662 (at
20%) up to a peak of 0.710 (at 60%), then flattens out and even dips
slightly by 100% (0.702). It is **not** still climbing at 100% — it
plateaued around the 60–80% mark.

**(iii) Conclusion — data-limited or capacity-limited?** The combination
of (a) training AUC staying persistently high regardless of data size and
(b) test AUC plateauing rather than continuing to rise points to a
**capacity/overfitting** issue rather than a pure data-quantity problem.
If we were purely data-limited, we'd expect test AUC to still be
climbing at 100% training data with training AUC gradually easing
downward as the model is forced to generalize more. Instead, the
persistent train/test gap (~0.98 vs ~0.70 throughout) suggests the model
is fitting patterns in the training data that don't transfer to new
patients, regardless of how much training data it gets — collecting more
of the *same kind* of data would likely help only marginally; more
benefit would probably come from stronger regularization, simpler
features, or fundamentally more informative input variables.

### Task 8: Model Serialization

The tuned pipeline was saved with:
```python
joblib.dump(best_pipeline, 'best_model.pkl')
```

**Reload-and-predict test** (proves the saved file works without
retraining):
```python
import joblib
import pandas as pd

loaded_model = joblib.load('best_model.pkl')

new_patients = pd.DataFrame([...])  # two hand-crafted patient rows
predictions = loaded_model.predict(new_patients)
probabilities = loaded_model.predict_proba(new_patients)[:, 1]
```
**Result:**
- **Patient A** (68 years old, high blood pressure, smoker, family
  history, diabetic): predicted **Yes**, probability = 0.762
- **Patient B** (29 years old, healthy vitals, non-smoker, no family
  history): predicted **No**, probability = 0.121

---

## Summary Comparison Table (Parts 2 & 3)

| Model | Mean 5-fold CV AUC | Std CV AUC | Test-set AUC |
|---|---|---|---|
| Logistic Regression | 0.7546 | 0.0376 | 0.7785 |
| Decision Tree (max_depth=5) | 0.6218 | 0.0326 | 0.6077* |
| Random Forest | 0.7402 | 0.0235 | 0.6964 |
| Gradient Boosting | 0.7272 | 0.0084 | 0.7299 |
| **Tuned Random Forest (GridSearchCV)** | **0.7665** | — | 0.7022 |


### Recommendation

**We recommend the Logistic Regression model.** Although the tuned
Random Forest has the highest mean cross-validated AUC (0.7665), Logistic
Regression has the highest **test-set AUC (0.7785)** of any model, one of
the lowest CV standard deviations relative to its performance level, and
is far simpler to explain to a clinical audience — coefficients map
directly and transparently onto each feature's estimated effect, which
matters in a healthcare context where stakeholders often need to
understand *why* a model flagged a patient as high-risk. The tree-based
ensembles didn't provide a large enough accuracy advantage here to justify
their added complexity and reduced interpretability, especially since the
learning curve in Task 7 suggests the Random Forest's gap comes from
overfitting rather than genuinely superior pattern-finding on this
dataset.
