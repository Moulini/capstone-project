# Red Wine Quality — Exploratory Data Analysis & Preprocessing Pipeline (Part 1)

This repository contains the foundational data preprocessing, data cleaning, and exploratory data analysis (EDA) pipeline for the UCI Red Wine Quality dataset. This phase ensures the dataset is structurally sound, mathematically audited, and optimally prepared for predictive modeling in Parts 2 and 3.

## 1. Environment & Initialization
The pipeline imports raw data directly from the UCI Machine Learning Repository. Because the source file utilizes semicolon delimiters (`(;)`), the initial ingest architecture explicitly standardizes the parsing framework using the `pandas` engine:

```python
import pandas as pd
import numpy as np

dataset_csv = "[https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)"
df = pd.read_csv(dataset_csv, sep=';')



Shape: You will see Shape: (1599, 12), meaning the dataset contains 1,599 individual wine samples and 12 different features (columns).

Data Types: 11 of the columns are numeric physiochemical properties (like fixed acidity, volatile acidity, residual sugar, and alcohol) stored as float64.

Target Variable: The 12th column is quality, stored as an int64. This is your target variable, which rates the wine on a scale from 0 to 10.



## 2. Data Imputation Strategy: Why Median Strategy Was Chosen Over Mean

For columns containing fewer than 20% missing values, numeric features were imputed using the **column median** rather than the mean. This decision was driven by the following architectural considerations:

* **Robustness to Outliers:** The mean (average) is highly sensitive to extreme values or skewed distributions. If a feature contains a few exceptionally large or small values, the mean shifts significantly, which would introduce bias into the missing records. The median, being the middle value, is a robust statistic that remains unaffected by outliers.
* **Preservation of Data Distribution:** In real-world datasets, numerical features (such as income, age, or transaction amounts) are rarely perfectly normally distributed. Using the median ensures that the imputed values reflect the central tendency of the *actual* majority of the data, rather than an artificially inflated or deflated average.
* **Safety Default:** Without manually verifying the distribution shape of every single column beforehand, the median serves as a mathematically safer "default" imputation strategy across diverse numeric features.
##3.dulpicate detection and removal :
Rows Removed Calculation: The difference between the original row count and the cleaned row count is computed as follows:

Rows Removed=df.shape[0]−df_cleaned.shape[0]
Null Percentage Impact: Removing duplicates can alter a column's null percentage. If a duplicate row contains missing values, dropping it reduces both the total row count (the denominator) and the missing value count (the numerator). The mathematical relationship dictates that:

If the duplicate rows have a higher proportion of nulls than the rest of the dataset, the null percentage decreases.

If the duplicate rows have a lower proportion of nulls than the rest of the dataset, the null percentage increases.

If they match precisely or contain no nulls, the percentage stays relatively stable or changes depending on the overall ratio.
##4.data type correction:
Flaw 1 (Incorrect Numeric Dtype): We will insert a corrupted string ('bad_data') into the total_sulfur_dioxide column, forcing pandas to read it as an object type instead of a numeric type.

Flaw 2 (Repetitive String Category): We will create an artificial text column called wine_type filled with repetitive values ('Red') to demonstrate converting it to the memory-efficient category datatype.
pd.to_numeric(..., errors='coerce'): If text anomalies creep into structural metrics, traditional .astype(float) operations will error out. By using errors='coerce', pandas transforms string corruptions directly into native NaN (Not a Number) floating-point structures, seamlessly changing the column format back from object to float64..astype('category'): Strings stored as object types are heavy because pandas holds pointers to every separate, raw character sequence in memory. Converting to a category builds an integer-indexed mapping array underneath. Instead of rewriting the word "Red" 1,599 times, pandas saves the word once and populates the series array with small integer flags ($0, 0, 0, \dots$).
memory_usage(deep=True): Using the deep=True flag tells pandas to dig beyond the surface system dataframe framework and calculate the actual byte sizes allocated to string components inside object columns.
##5.Descriptive statistics and skewness:
## Statistical Analysis: Skewness and Imputation Consequences

Through exploratory data analysis, the feature **`[Insert_Highest_Skew_Column_Name_Here]`** was identified as having the highest absolute skewness in the dataset. 

### Understanding the Skewness of this Column

* **If the Skewness is Positive ($> 0$):** The distribution is **right-skewed** (tail extends toward higher values). This means the majority of the data points are clustered at lower values, while a few exceptionally large outliers pull the right tail out. 
* **If the Skewness is Negative ($< 0$):** The distribution is **left-skewed** (tail extends toward lower values). This means most data points are clustered at higher values, with a few exceptionally small outliers dragging the left tail out.



### Consequences for Imputing Missing Values with the Mean

Using the arithmetic mean to fill missing values in this specific column introduces significant statistical bias due to its skewness:

1. **The Mean is Displaced by Outliers:** * In a *positively skewed* distribution, the extreme high values pull the mean upward, making it greater than both the mode and the median ($\text{Mean} > \text{Median}$).
   * In a *negatively skewed* distribution, the extreme low values pull the mean downward ($\text{Mean} < \text{Median}$).
2. **Artificial Variance and Distortion:** If we impute missing records using the mean, we are injecting a value that does not represent the "typical" data point. For a right-skewed feature, the mean overestimates the typical value; for a left-skewed feature, it underestimates it. This distorts the natural variance, creates an artificial spike at the mean, and heavily misrepresents the true behavior of the underlying population to downstream machine learning models.

##6. Outlier Documentation & Handling Strategy (Part 2)

During the exploratory data analysis (EDA) phase, substantial right-skewness and significant outlier populations were discovered via the Interquartile Range ($IQR$) method for the following features:
* **Residual Sugar:** $155$ anomalies detected outside the statistical upper bound ($> 3.6500$).
* **Chlorides:** $112$ anomalies detected outside the statistical upper bound ($> 0.1200$).

### Strategy for Part 2: Capping (Winsorization) and Robust Transformation

Rather than dropping or ignoring these outliers, they will be preserved and managed via a combination of **Capping (Winsorization)** and **Logarithmic Transformation** during the feature engineering stage.

#### Why We Are NOT Dropping Them:
1. **Domain Relevance:** High residual sugar levels or elevated chloride levels are not entry/measurement errors; they represent valid, distinct chemical variations in specific styles of wine (e.g., sweet wines or wines produced from vineyards close to saline soils). Removing them would scrub away critical chemical edge-cases.
2. **Data Preservation:** Dropping outliers across multiple columns simultaneously would lead to a dramatic reduction in sample size, stripping away over $10\%$ of our training patterns.

#### Why We Are Handling Them via Capping/Transformation:
* **Linear Model Sensitivity:** Algorithms that rely on distance metrics or variance minimization (e.g., Linear Regression, Logistic Regression, Support Vector Machines) are highly sensitive to extreme inputs. A chloride concentration of $0.6110$ is more than $5$ times the upper bound and would disproportionately warp the decision boundaries.
* **Capping Implementation:** In Part 2, extreme values beyond the $1.5 \times IQR$ upper thresholds will be capped at the 95th or 99th percentile value. This neutralizes the leverage of extreme values while maintaining the rank and order distribution of the underlying data points.
* **Alternative Tree-Based Approach:** If Ensemble Trees (e.g., Random Forests or XGBoost) are selected as our primary modeling pipeline, the outliers will be **retained completely unchanged**, as tree-based partition nodes are natively invariant to the scale and magnitudes of extreme leaf values.
## 7.Exploratory Data Analysis & Visual Interpretations

### 1. Histogram Distribution (Chlorides)
* **Shape Profile:** The distribution exhibits a heavy **extreme right-skew (positive skewness of 5.68)**. 
* **Observations:** The vast majority of observations are tightly clustered around a modal peak between $0.07$ and $0.09$ g/dm³. However, a prolonged, sparse tail extends far to the right, reaching up to $0.61$ g/dm³. This emphasizes that while red wine generally maintains uniform, low salt concentrations, severe structural outliers are present in this feature subset.

### 2. Scatter Plot Correlation (Fixed Acidity vs. Density)
* **Direction:** **Strongly Positive.** As fixed acidity values escalate, wine density displays a corresponding linear increases.
* **Strength & Interpretation:** Approximate moderate-to-strong relationship. This direct correlation makes physical sense: fixed organic acids (like tartaric acid) weigh more than water. Higher concentrations of these dissolved solids naturally dense up the liquid solution mass per unit volume.

### 3. Box Plot Distribution (Sulphates across Quality)
* **Median Differences:** There is a visible upward shift in median values as quality tier climbs. 'Low' quality wines exhibit a lower median sulphate baseline ($\approx 0.55$), whereas 'High' quality wines display a noticeably higher median concentration ($\approx 0.74$).
* **Spread and Outliers:** The 'Medium' quality cohort occupies the largest absolute spread and contains a dense collection of extreme upper outliers. The interquartile range (IQR) box for 'High' quality wine sits higher, showing that premium ratings are heavily associated with an elevated, controlled window of preservation via sulphates.
##8.## Correlation vs. Causality: Fixed Acidity and pH

The exploratory correlation analysis revealed that **fixed acidity** and **pH** share the strongest absolute correlation in the dataset, with a coefficient of **$-0.68$**.
### Does this indicate a direct causal relationship?
Yes, in this case, the relationship is **directly causal** due to fundamental laws of chemistry. By definition, pH is the negative logarithm of hydrogen ion ($\text{H}^+$) concentration in a liquid solution:

$$\text{pH} = -\log_{10}[\text{H}^+]$$

Fixed acids in wine (primarily tartaric, malic, and citric acids) release hydrogen ions when dissolved in water. Therefore, an increase in the concentration of **fixed acidity** directly causes a proportional increase in free $\text{H}^+$ ions, which mathematically forces the **pH** value to drop. 

### Plausible Confounding or Mediating Factors (Alternative Explanations)
While the chemical link is direct, treating it as a simple two-variable vacuum ignores the complex buffer systems present in vinification. A third variable can heavily alter or mitigate this relationship:

1. **Potassium Ions ($\text{K}^+$) / Soil Composition:** * **How it interferes:** Potassium acts as a major confounding element. If a vineyard's soil has high potassium levels, the grapes absorb more potassium ions. 
   * **The Alternative Effect:** These $\text{K}^+$ ions bind with tartaric acid to form potassium bitartrate. This reaction neutralizes free hydrogen ions without removing the physical acid molecules from the liquid. Consequently, a wine can simultaneously have **high fixed acidity** and a **high pH** (low real-world acidity) because the potassium ions mask the expected chemical drop.
2. **Fermentation Conditions (Malolactic Fermentation - MLF):**
   * During winemaking, bacteria can convert harsh malic acid into softer lactic acid. While both contribute to the overall "fixed acidity" metric, they have entirely different ionization strengths. This biological shift can alter the pH balance without noticeably changing the total physical mass of fixed acids calculated by basic laboratory assays.
   ##9.Part 2 Project Documentation Additions (README)A. Imputation Strategy JustificationFor both chlorides and residual sugar (the two most heavily right-skewed columns), Median Imputation was selected over Mean Imputation.The Mathematical Rule: When a feature is highly positively skewed, its mean is pulled unnaturally upward by extreme outliers. Imputing missing records with the mean would introduce a systematic upward bias.The Decision: The median represents the exact 50th percentile of the actual data density, remaining completely unaffected by extreme outliers. It provides a more robust estimate of central tendency for missing values in skewed physical attributes.B. Spearman vs. Pearson Matrix AnalysisBased on the absolute difference calculations, the top three column pairs exhibiting the widest divergence between metrics are analyzed below:1. Volatile Acidity & Citric AcidRelationship Profile: Monotonic but Non-Linear ($|\text{Spearman}| > |\text{Pearson}|$). The rank correlation is notably stronger than the linear metric, indicating that as volatile acidity decreases, citric acid rises along a curvilinear trajectory rather than a rigid straight line.Feature Selection Strategy: Spearman will guide feature selection. Relying solely on Pearson would undervalue this critical chemical interplay, risking the premature exclusion of highly informative patterns from linear models.2. Free Sulfur Dioxide & Total Sulfur DioxideRelationship Profile: Monotonic but Non-Linear ($|\text{Spearman}| > |\text{Pearson}|$). Because free sulfur dioxide is a chemical subset of total sulfur dioxide, they move together consistently. However, this relationship scales non-proportionally due to varying molecular binding rates across different wine batches.Feature Selection Strategy: Spearman is the preferred choice here. It accurately captures the structural lock-step variance between the two features without requiring complex mathematical power transforms up front.3. Density & AlcoholRelationship Profile: Approximately Linear ($|\text{Pearson}| \ge |\text{Spearman}|$). The two metrics are closely aligned, showing a strong negative relationship. This indicates that as alcohol content increases, density drops at a remarkably steady, proportional rate.Feature Selection Strategy: Pearson is entirely reliable for this pair, as the underlying relationship fits a standard straight-line assumption.C. Grouped Aggregation Insights (quality_category vs. alcohol)Group Statistics Summary:Highest Mean: The High quality group boasts the highest average alcohol concentration ($\approx 11.52\%$).Highest Standard Deviation: The Low quality group displays the largest internal variance ($\approx 0.96\%$).Modeling Considerations:Is high within-group variance a concern? Yes. The elevated standard deviation within the Low quality category demonstrates that alcohol content alone cannot act as a definitive single predictor. Because some low-quality wines still possess high alcohol metrics, a model will need additional features (like acidity or volatile compounds) to draw clean decision boundaries.Predictive Signal Strength: The ratio of the highest group mean to the lowest group mean is roughly $1.12$ ($11.52 / 10.21$). While an $12\%$ relative variance shift across categories might seem small on paper, in commercial vinification, an average difference of over $1.3\%$ absolute alcohol volume between table wines and premium wines is a massive statistical delimiter. This confirms that quality_category holds a highly viable predictive signal for classification algorithms.