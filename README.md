# Credit Card Default Prediction

An initial comparison of ten classifier configurations using 30,000 credit-card customer records.

**Python · pandas · scikit-learn · XGBoost**

## Project goal

Identify customers likely to default on their next payment using repayment history, billing information and customer characteristics. We prioritized recall to catch more actual defaults and examined precision and F1 to understand the trade-off in false alarms.

## My contribution — Rahul Malla

- Built model pipelines and configured ten classifier variants, including logistic regression, decision trees, random forest, SVMs, discriminant analysis and XGBoost.
- Ran stratified five-fold cross-validation and compared results, prioritizing recall alongside precision and F1 from the initial test evaluation.
- Created model-evaluation visualizations and worked with teammates to interpret the findings.

**Team:** Ethan Vanbeurden, Rahul Malla and Eric Testen. Developed for DTSC 2302, Project 2, Part A.

## Approach

The dataset contains 30,000 records, with approximately 22% defaults. Preparation removes the customer ID, encodes categorical values and adds utilization, payment-ratio and delinquency features.

The analysis uses a stratified 80/20 split: 24,000 training records and 6,000 test records. Scaling stays inside model pipelines. Five-fold cross-validation runs on the training partition.

[Explore the notebook](notebooks/credit_card_default.ipynb) · [Data source and access](data/README.md)

## Results

The depth-3 decision tree had the highest mean cross-validation recall among the ten configurations tested.

| Depth-3 decision tree | Result |
|---|---:|
| Five-fold mean recall | 68.4% |
| Fold standard deviation | 2.4 percentage points |
| Initial test recall | 64.7% |
| Initial test precision | 44.6% |
| Initial test F1 | 52.8% |

On the initial test split, the tree identified about 65% of actual defaults. About 45% of the customers it flagged actually defaulted, showing the trade-off between catching defaults and raising false alarms.

![Recall comparison across ten configurations](figures/model-comparison.png)

### Why did the tree lead on recall?

The comparison used limited tuning. The tree used balanced class weights, while the original XGBoost run ignored its `class_weight` argument. The random forest was also restricted to depth 3, and decision thresholds were not tuned. These differences limit what the ranking says about the model families.

The next modeling step would be to correct XGBoost's weighting, tune model settings and compare thresholds using training-only validation. Those experiments have not been run.

[Original comparison results](results/original_cv_results.csv) · [Verified tree metrics](results/verified_decision_tree.json)

The tree's results were reproduced during portfolio preparation. The comparison chart retains the original ten-model results; the full comparison was not rerun.

### What the tree shows

Maximum repayment delay is the first split in the fitted tree. Its shallow structure makes the decision rules easy to inspect.

![Fitted depth-3 decision tree](figures/decision-tree.png)

## Run the analysis

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab notebooks/credit_card_default.ipynb
```

Run cells in order. The notebook accepts a local CSV or uses the course data URL. Profiling is optional and disabled by default; SVM training and cross-validation can take time. The original dependency versions were not recorded, so versions are unpinned and results may vary.

## Scope and next steps

- The original analysis inspected test results across models before cross-validation. A stronger evaluation would use nested cross-validation or a fresh final holdout.
- Error costs, demographic fairness, probability calibration and performance over time still need evaluation before any lending use.
- Feature choices simplify the data: repayment codes are grouped, utilization is clipped, and summed monthly statements can include carried balances.

This repository presents the academic analysis. Model code and saved scores are preserved, apart from the previously removed XGBoost argument that had no effect. AI assistance in the original project was limited to occasional debugging suggestions.
