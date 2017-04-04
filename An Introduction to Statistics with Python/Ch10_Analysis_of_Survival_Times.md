
# 10 Analysis of Survival Times
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [10 Analysis of Survival Times](#10-analysis-of-survival-times)
  * [10.1 Survival Distributions](#101-survival-distributions)
  * [10.2 Survival Probabilities](#102-survival-probabilities)
    * [10.2.1 Censorship](#1021-censorship)
    * [10.2.2 Kaplan–Meier Survival Curve](#1022-kaplanmeier-survival-curve)
  * [10.3 Comparing Survival Curves in Two Groups](#103-comparing-survival-curves-in-two-groups)

<!-- tocstop -->


## 10.1 Survival Distributions


```python
# %load ch10/L10_1_weibullDemo.py
''' Example of fitting the Weibull modulus. '''

# author: Thomas Haslwanter, date: Jun-2015

# Import standard packages
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

# Generate some sample data, with a Weibull modulus of 1.5
WeibullDist = stats.weibull_min(1.5)
data = WeibullDist.rvs(500)

# Now fit the parameter
fitPars = stats.weibull_min.fit(data)

# Note: fitPars contains (WeibullModulus, Location, Scale)
print('The fitted Weibull modulus is {0:5.2f}, compared to the exact value of 1.5 .'.format(fitPars[0]))
```

    The fitted Weibull modulus is  1.43, compared to the exact value of 1.5 .


## 10.2 Survival Probabilities

### 10.2.1 Censorship

### 10.2.2 Kaplan–Meier Survival Curve

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/10_SurvivalAnalysis/lifelinesDemo.


```python
# %load ch10/L10_2_lifelinesSurvival.py
''' Graphical representation of survival curves, and comparison of two
curves with logrank test.
"miR-137" is a short non-coding RNA molecule that functions to regulate
the expression levels of other genes.
'''
# author: Thomas Haslwanter, date: Jun-2015

# Import standard packages
import matplotlib.pyplot as plt

# additional packages
import sys
sys.path.append(r'..\Quantlets\Utilities')
import ISP_mystyle

from lifelines.datasets import load_waltons
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Set my favorite font
ISP_mystyle.setFonts(18)

# Load and show the data
df = load_waltons() # returns a Pandas DataFrame

print(df.head())
'''
    T  E    group
0   6  1  miR-137
1  13  1  miR-137
2  13  1  miR-137
3  13  1  miR-137
4  19  1  miR-137
'''

T = df['T']
E = df['E']

groups = df['group']
ix = (groups == 'miR-137')

kmf = KaplanMeierFitter()

kmf.fit(T[~ix], E[~ix], label='control')
ax = kmf.plot()

kmf.fit(T[ix], E[ix], label='miR-137')
kmf.plot(ax=ax)

plt.ylabel('Survival Probability')
outFile = 'lifelines_survival.png'
ISP_mystyle.showData(outFile)

# Compare the two curves
results = logrank_test(T[ix], T[~ix], event_observed_A=E[ix], event_observed_B=E[~ix])
results.print_summary()

```

## 10.3 Comparing Survival Curves in Two Groups


```python

```
