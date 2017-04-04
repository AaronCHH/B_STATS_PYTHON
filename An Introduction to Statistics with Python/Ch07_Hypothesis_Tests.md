
# 7 Hypothesis Tests
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [7 Hypothesis Tests](#7-hypothesis-tests)
  * [7.1 Typical Analysis Procedure](#71-typical-analysis-procedure)
    * [7.1.1 Data Screening and Outliers](#711-data-screening-and-outliers)
    * [7.1.2 Normality Check](#712-normality-check)
    * [7.1.3 Transformation](#713-transformation)
  * [7.2 Hypothesis Concept, Errors, p-Value, and Sample Size](#72-hypothesis-concept-errors-p-value-and-sample-size)
    * [7.2.1 An Example](#721-an-example)
    * [7.2.2 Generalization and Applications](#722-generalization-and-applications)
    * [7.2.3 The Interpretation of the p-Value](#723-the-interpretation-of-the-p-value)
    * [7.2.4 Types of Error](#724-types-of-error)
    * [7.2.5 Sample Size](#725-sample-size)
  * [7.3 Sensitivity and Specificity](#73-sensitivity-and-specificity)
    * [7.3.1 Related Calculations](#731-related-calculations)
  * [7.4 Receiver-Operating-Characteristic (ROC) Curve](#74-receiver-operating-characteristic-roc-curve)

<!-- tocstop -->


## 7.1 Typical Analysis Procedure

### 7.1.1 Data Screening and Outliers

### 7.1.2 Normality Check

* a) Probability-Plots


```python
stats.probplot(data, plot=plt)
```

* b) Tests for Normality

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/07_CheckNormality_CalcSamplesize/checkNormality

### 7.1.3 Transformation

## 7.2 Hypothesis Concept, Errors, p-Value, and Sample Size

### 7.2.1 An Example


```python
import numpy as np
from scipy import stats

scores = np.array([ 109.4, 76.2, 128.7, 93.7, 85.6,
                   117.7, 117.2, 87.3, 100.3, 55.1])

tval = (110-np.mean(scores))/stats.sem(scores) # 1.84
td = stats.t(len(scores)-1) # "frozen" t-distribution
p = 2*td.sf(tval) # 0.0995
```

### 7.2.2 Generalization and Applications

* a) Generalization

* b) Additional Examples

### 7.2.3 The Interpretation of the p-Value

### 7.2.4 Types of Error

* a) Type I Errors

* b) Type II Errors and Test Power

* c) Pitfalls in the Interpretation of p-Values

### 7.2.5 Sample Size

* a) Examples

* b) Python Solution


```python
from statsmodels.stats import power
nobs = power.tt_ind_solve_power(effect_size = 0.5, alpha =0.05, power=0.8 )
print(nobs)
```

    63.76561177540974



```python
effect_size = power.tt_ind_solve_power(alpha =0.05, power=0.8, nobs1=25 )
print(effect_size)
```

    0.8087077886680407


* c) Programs: Sample Size

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/07_CheckNormality_CalcSamplesize/sampleSize.

## 7.3 Sensitivity and Specificity

### 7.3.1 Related Calculations

## 7.4 Receiver-Operating-Characteristic (ROC) Curve


```python

```
