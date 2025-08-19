
# Assertion Subtype Classification – Evaluation Report

## Task Overview

The task involves **multi-class classification** of scientific statements into **one of six predefined assertion subtypes**:

1. **Definitional/Taxonomic**
2. **Causal/Mechanistic**
3. **Quantitative/Statistical**
4. **Observational/Correlational**
5. **Methodological**
6. **Experimental/Interventional**

The model was run **five times** on the same input dataset for each chapter. For each statement:

- `assertion_subtype_pred` represents the **majority-voted** predicted subtype label (i.e., the most frequent prediction across five runs).
- `assertion_subtype_pred2` represents the **secondary** predicted subtype (i.e., another label that appeared in at least one run but was not the majority vote).

---

## 📄 Dataset

The input data is organized in an Excel file (`Assertions.xlsx`) with **two sheets**:
- **Chapter 1**
- **Chapter 2**

Each sheet includes the following columns:

| Column Name            | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `statement_text`       | The original scientific assertion (sentence)                                |
| `assertion_subtype_pred`  | The final majority-vote classification from five model runs                |
| `assertion_subtype_pred2` | The second-most frequent classification (if any) across the five model runs |

---

## Evaluation Methodology

To assess classification consistency across runs, we evaluated how often the `assertion_subtype_pred` and `assertion_subtype_pred2` **match**.

- If the two columns **match** → ✅ **Correct prediction** (model showed stability)
- If the two columns **differ** → ❌ **Incorrect prediction** (model showed inconsistency)

### Metrics Reported

1. **Power**  
   → Proportion of rows where `pred1` and `pred2` matched (i.e., consistent classification)

2. **Confidence Level (95% CI)**  
   → Statistical confidence interval of the power score using the **Wilson score interval**

3. **Confusion Matrix**  
   → Detailed view of how subtype labels were predicted vs alternates across all assertions

---

## Evaluation Results

### Summary Table

| Sheet      | Total Rows | Correct | **Power (Accuracy)** | 95 % Confidence Interval |
|------------|------------|---------|----------------------|--------------------------|
| Chapter 1  | 162        | 156     | **96.30%**           | [92.16%, 98.29%]         |
| Chapter 2  | 89         | 75      | **84.27%**           | [75.31%, 90.39%]         |
| **Overall**| **251**    | **231** | **92.03%**           | **[88.01%, 94.78%]**     |

> **Interpretation**:
> - Chapter 1 showed **high stability** with over 96% agreement between primary and secondary predictions.
> - Chapter 2 had more variation, with a lower but still solid 84% agreement.
> - The model overall demonstrated strong consistency across 92% of all assertions.

---

### Confusion Matrices

To better understand where disagreements occurred between `assertion_subtype_pred` and `assertion_subtype_pred2`, confusion matrices were computed per chapter and overall.

- **Diagonal cells** = agreement (correct predictions)
- **Off-diagonal cells** = disagreement (confusion between subtypes)

---

## Key Takeaways

- The model predictions are highly consistent across multiple runs, especially for Chapter 1.
- Slight variability exists in Chapter 2, suggesting possible ambiguities in label boundaries or statement interpretation.
- The most common sources of disagreement can be identified and refined by analyzing the confusion matrix.


## Confusion Matrices

### Confusion Matrix – Chapter 1

| assertion_subtype_pred      |   Causal/Mechanistic |   Comparative |   Definitional/Taxonomic |   Observational/Correlational |   Quantitative/Statistical |
|:----------------------------|---------------------:|--------------:|-------------------------:|------------------------------:|---------------------------:|
| Causal/Mechanistic          |                   72 |             0 |                        4 |                             0 |                          0 |
| Comparative                 |                    0 |             7 |                        0 |                             0 |                          0 |
| Definitional/Taxonomic      |                    1 |             0 |                       44 |                             0 |                          0 |
| Observational/Correlational |                    0 |             0 |                        0 |                            19 |                          0 |
| Quantitative/Statistical    |                    0 |             0 |                        1 |                             0 |                         14 |


### Confusion Matrix – Chapter 2

| assertion_subtype_pred      |   Causal/Mechanistic |   Comparative |   Definitional/Taxonomic |   Experimental/Interventional |   Observational/Correlational |   Quantitative/Statistical |
|:----------------------------|---------------------:|--------------:|-------------------------:|------------------------------:|------------------------------:|---------------------------:|
| Causal/Mechanistic          |                   21 |             0 |                        0 |                             0 |                             5 |                          0 |
| Comparative                 |                    0 |             2 |                        1 |                             0 |                             0 |                          1 |
| Definitional/Taxonomic      |                    0 |             0 |                       29 |                             0 |                             1 |                          0 |
| Experimental/Interventional |                    0 |             0 |                        0 |                             2 |                             2 |                          0 |
| Observational/Correlational |                    0 |             0 |                        1 |                             0 |                            13 |                          0 |
| Quantitative/Statistical    |                    1 |             0 |                        1 |                             0 |                             1 |                          8 |


### Confusion Matrix – Overall

| assertion_subtype_pred      |   Causal/Mechanistic |   Comparative |   Definitional/Taxonomic |   Experimental/Interventional |   Observational/Correlational |   Quantitative/Statistical |
|:----------------------------|---------------------:|--------------:|-------------------------:|------------------------------:|------------------------------:|---------------------------:|
| Causal/Mechanistic          |                   93 |             0 |                        4 |                             0 |                             5 |                          0 |
| Comparative                 |                    0 |             9 |                        1 |                             0 |                             0 |                          1 |
| Definitional/Taxonomic      |                    1 |             0 |                       73 |                             0 |                             1 |                          0 |
| Experimental/Interventional |                    0 |             0 |                        0 |                             2 |                             2 |                          0 |
| Observational/Correlational |                    0 |             0 |                        1 |                             0 |                            32 |                          0 |
| Quantitative/Statistical    |                    1 |             0 |                        2 |                             0 |                             1 |                         22 |