
# 13 Tests on Discrete Data
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [13 Tests on Discrete Data](#13-tests-on-discrete-data)
  * [13.1 Comparing Groups of Ranked Data](#131-comparing-groups-of-ranked-data)
  * [13.2 Logistic Regression](#132-logistic-regression)
    * [13.2.1 Example: The Challenger Disaster](#1321-example-the-challenger-disaster)
  * [13.3 Generalized Linear Models](#133-generalized-linear-models)
    * [13.3.1 Exponential Family of Distributions](#1331-exponential-family-of-distributions)
    * [13.3.2 Linear Predictor and Link Function](#1332-linear-predictor-and-link-function)
  * [13.4 Ordinal Logistic Regression](#134-ordinal-logistic-regression)
    * [13.4.1 Problem Definition](#1341-problem-definition)
    * [13.4.2 Optimization](#1342-optimization)
    * [13.4.3 Code](#1343-code)
    * [13.4.4 Performance](#1344-performance)

<!-- tocstop -->


## 13.1 Comparing Groups of Ranked Data

* book(https://github.com/thomas-haslwanter/dobson).

## 13.2 Logistic Regression

### 13.2.1 Example: The Challenger Disaster


```python
# %load ch13/L13_1_logitShort.py
# Import standard packages
import numpy as np
import os
import pandas as pd

# additional packages
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial

# Get the data
inFile = 'challenger_data.csv'
challenger_data = np.genfromtxt(inFile, skip_header=1,
                    usecols=[1, 2], missing_values='NA',
                    delimiter=',')
# Eliminate NaNs
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

# Create a dataframe, with suitable columns for the fit
df = pd.DataFrame()
df['temp'] = np.unique(challenger_data[:,0])
df['failed'] = 0
df['ok'] = 0
df['total'] = 0
df.index = df.temp.values

# Count the number of starts and failures
for ii in range(challenger_data.shape[0]):
    curTemp = challenger_data[ii,0]
    curVal  = challenger_data[ii,1]
    df.loc[curTemp,'total'] += 1
    if curVal == 1:
        df.loc[curTemp, 'failed'] += 1
    else:
        df.loc[curTemp, 'ok'] += 1

# fit the model

# --- >>> START stats <<< ---
model = glm('ok + failed ~ temp', data=df, family=Binomial()).fit()
# --- >>> STOP stats <<< ---

print(model.summary())

```

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/13_LogisticRegression/LogisticRegression.

## 13.3 Generalized Linear Models

### 13.3.1 Exponential Family of Distributions

### 13.3.2 Linear Predictor and Link Function

## 13.4 Ordinal Logistic Regression

### 13.4.1 Problem Definition

* This section has been taken with permission from Fabian Pedregosa’s blog on ordinal logistic regression, http://fa.bianp.net/blog/2013/logistic-ordinal-regression/.

### 13.4.2 Optimization

### 13.4.3 Code

### 13.4.4 Performance

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/13_LogisticRegression/OrdinalLogisticRegression.


```python

```
