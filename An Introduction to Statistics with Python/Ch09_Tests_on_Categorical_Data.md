
# 9 Tests on Categorical Data
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [9 Tests on Categorical Data](#9-tests-on-categorical-data)
  * [9.1 One Proportion](#91-one-proportion)
    * [9.1.1 Confidence Intervals](#911-confidence-intervals)
    * [9.1.2 Explanation](#912-explanation)
    * [9.1.3 Example](#913-example)
  * [9.2 Frequency Tables](#92-frequency-tables)
    * [9.2.1 One-Way Chi-Square Test](#921-one-way-chi-square-test)
    * [9.2.2 Chi-Square Contingency Test](#922-chi-square-contingency-test)
    * [9.2.3 Fisher's Exact Test](#923-fishers-exact-test)
    * [9.2.4 McNemar's Test](#924-mcnemars-test)
    * [9.2.5 Cochran's Q Test](#925-cochrans-q-test)
  * [9.3 Exercises](#93-exercises)

<!-- tocstop -->


## 9.1 One Proportion

### 9.1.1 Confidence Intervals

### 9.1.2 Explanation

### 9.1.3 Example

## 9.2 Frequency Tables

### 9.2.1 One-Way Chi-Square Test


```python
V, p = stats.chisquare(data)
print(p)
```

### 9.2.2 Chi-Square Contingency Test

* a) Assumptions

* b) Degrees of Freedom

* c) Example 1


```python
data = np.array([[43,9],
                 [44,4]])
V, p, dof, expected = stats.chi2_contingency(data)
print(p)
```

* d) Example 2

* e) Comments

### 9.2.3 Fisher's Exact Test

* a) Example: ``A Lady Tasting Tea


```python
oddsratio, p = stats.fisher_exact(obs, alternative='greater')
```

### 9.2.4 McNemar's Test

* a) Example


```python
from statsmodels.sandbox.stats.runs import mcnemar
obs = [[a,b], [c, d]]
chi2, p = mcnemar(obs)
```

### 9.2.5 Cochran's Q Test

* a) Example


```python
from statsmodels.sandbox.stats.runs import cochrans_q
obs = [[a,b], [c, d]]
q_stat, p = cochrans_q(obs)
```

## 9.3 Exercises


```python

```
