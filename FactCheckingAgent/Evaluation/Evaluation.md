# Fact-Checking Agent Evaluation

_Generated: 2025-08-27 20:08:43_

**Source file:** `Chapter 2 Fact Checking Results.csv`
**Detected final verdict column:** `Final Verdict`

## Dataset Overview

- Total statements (N): **94**

**Original label distribution**

| verdict | count |
| --- | --- |
| Correct | 76 |
| Flagged for Review | 14 |
| Incorrect | 4 |


**Mapped label distribution** (using `positive`, `negative`, `abstain`, `unmapped`)

| mapped | count |
| --- | --- |
| positive | 80 |
| abstain | 14 |


> **Assumption:** All statements in the CSV are factually correct (ground truth = positive for every row).
> Under this assumption, there are no ground-truth negatives; therefore FP and TN are necessarily 0.

## Confusion Matrix (Strict Policy)

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| Ground Truth: Positive | 80 | 14 |
| Ground Truth: Negative | 0 | 0 |


## Metrics (Strict Policy)

| Metric | Value |
| --- | --- |
| Accuracy | 0.8511 |
| Precision | 1.0 |
| Recall | 0.8511 |
| F1-score | 0.9195 |
| Coverage (non-abstain rate) | 0.8511 |
| Abstain rate | 0.1489 |
| Negative rate | 0.0 |
| Unmapped rate | 0.0 |


### Notes on Strict Policy

- **Positive = 'Correct/Supported/True'**, **Negative = everything else (including 'Flagged for Review').**
- With all ground-truth positives, **Precision is 1.0 by construction** (no false positives are possible).
- **Recall** = fraction of statements your agent labeled as positive.

## Metrics Excluding Abstains (Optional View)

- Here we **exclude 'abstain' rows** from evaluation to focus only on cases where the model made a definitive call.

| Metric | Value |
| --- | --- |
| Covered samples (N) | 80.0 |
| Non-abstain accuracy | 1.0 |
| Precision (non-abstain) | 1.0 |
| Recall (non-abstain) | 1.0 |
| F1-score (non-abstain) | 1.0 |
| TP (non-abstain) | 80.0 |
| FN (non-abstain) | 0.0 |


## Label Mapping Rules

We normalized verdicts using the following keyword-based mapping:

- **positive** if label contains any of: `correct`, `supported`, `true`, `yes`, `verified` or numeric `1`/`true`
- **negative** if label contains any of: `incorrect`, `refuted`, `false`, `no`, `contradict*`, `wrong` or numeric `0`/`false`
- **abstain** if label contains any of: `flagged`, `review`, `not enough`, `insufficient`, `unknown`, `unclear`, `cannot determine`, `not sure`, `maybe`, `needs review`
- **unmapped** otherwise

## Interpretation & Caveats

- Because ground truth has **no negatives**, FP and TN are **0**, so **Precision = 1.0** whenever at least one positive prediction exists.
- The key performance driver here is **Recall** (and thus F1): it measures how often your agent confidently labels truly-correct statements as **Correct/Supported**.
- **Abstain rate** reveals how frequently the system avoids a decision; a high abstain rate lowers overall recall under the strict policy.
- If you later obtain mixed ground truth (some incorrect statements), recompute the metrics with true labels to get a realistic precision.
