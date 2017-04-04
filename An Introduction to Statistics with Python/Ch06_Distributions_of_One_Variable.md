
# 6 Distributions of One Variable
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [6 Distributions of One Variable](#6-distributions-of-one-variable)
  * [6.1 Characterizing a Distribution](#61-characterizing-a-distribution)
    * [6.1.1 Distribution Center](#611-distribution-center)
    * [6.1.2 Quantifying Variability](#612-quantifying-variability)
    * [6.1.3 Parameters Describing the Form of a Distribution](#613-parameters-describing-the-form-of-a-distribution)
    * [6.1.4 Important Presentations of Probability Densities](#614-important-presentations-of-probability-densities)
  * [6.2 Discrete Distributions](#62-discrete-distributions)
    * [6.2.1 Bernoulli Distribution](#621-bernoulli-distribution)
    * [6.2.2 Binomial Distribution](#622-binomial-distribution)
    * [6.2.3 Poisson Distribution](#623-poisson-distribution)
  * [6.3 Normal Distribution](#63-normal-distribution)
    * [6.3.1 Examples of Normal Distributions](#631-examples-of-normal-distributions)
    * [6.3.2 Central Limit Theorem](#632-central-limit-theorem)
    * [6.3.3 Distributions and Hypothesis Tests](#633-distributions-and-hypothesis-tests)
  * [6.4 Continuous Distributions Derived from the NormalDistribution](#64-continuous-distributions-derived-from-the-normaldistribution)
    * [6.4.1 t-Distribution](#641-t-distribution)
    * [6.4.2 Chi-Square Distribution](#642-chi-square-distribution)
    * [6.4.3 F-Distribution](#643-f-distribution)
  * [6.5 Other Continuous Distributions](#65-other-continuous-distributions)
    * [6.5.1 Lognormal Distribution](#651-lognormal-distribution)
    * [6.5.2 Weibull Distribution](#652-weibull-distribution)
    * [6.5.3 Exponential Distribution](#653-exponential-distribution)
    * [6.5.4 Uniform Distribution](#654-uniform-distribution)
  * [6.6 Exercises](#66-exercises)

<!-- tocstop -->


## 6.1 Characterizing a Distribution

### 6.1.1 Distribution Center

* a) Mean


```python
import numpy as np
x = np.arange(10)
np.mean(x)
```




    4.5




```python
xWithNan = np.hstack( (x, np.nan) ) # append nan
np.mean(xWithNan)
```




    nan




```python
np.nanmean(xWithNan)
```




    4.5



* b) Median


```python
np.median(x)
```




    4.5



* c) Mode


```python
from scipy import stats
data = [1, 3, 4, 4, 7]
stats.mode(data)
```




    ModeResult(mode=array([4]), count=array([2]))



* d) Geometric Mean


```python
x = np.arange(1,101)
stats.gmean(x)
```




    37.992689344834304



### 6.1.2 Quantifying Variability

* a) Range


```python
range = np.ptp(x)
range
```




    99



* b) Percentiles


```python

```

* c) Standard Deviation and Variance


```python
data = np.arange(7,14)
data
```




    array([ 7,  8,  9, 10, 11, 12, 13])




```python
np.std(data, ddof=0)
```




    2.0




```python
np.std(data, ddof=1)
```




    2.1602468994692869



* d) Standard Error


```python

```

* e) Confidence Intervals


```python

```

### 6.1.3 Parameters Describing the Form of a Distribution

* a) Location

* b) Scale

* c) Shape Parameters

### 6.1.4 Important Presentations of Probability Densities


```python
import numpy as np
from scipy import stats
myDF = stats.norm(5,3) # Create the frozen distribution
x = np.linspace(-5, 15, 101)
y = myDF.cdf(x) # Calculate the corresponding CDF
```

## 6.2 Discrete Distributions

### 6.2.1 Bernoulli Distribution


```python
from scipy import stats
p = 0.5
bernoulliDist = stats.bernoulli(p)
```


```python
p_tails = bernoulliDist.pmf(0)
p_heads = bernoulliDist.pmf(1)
```


```python
trials = bernoulliDist.rvs(10)
trials
```




    array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1])



### 6.2.2 Binomial Distribution


```python
from scipy import stats
import numpy as np
(p, num) = (0.5, 4)
binomDist = stats.binom(num, p)
```


```python
binomDist.pmf(np.arange(5))
```




    array([ 0.0625,  0.25  ,  0.375 ,  0.25  ,  0.0625])




```python

```

* b) Example: Binomial Test

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/06_Distributions/binomialTest

### 6.2.3 Poisson Distribution

## 6.3 Normal Distribution

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/06_Distributions/distDiscrete.


```python
import numpy as np
from scipy import stats
mu = -2
sigma = 0.7
myDistribution = stats.norm(mu, sigma)
significanceLevel = 0.05
myDistribution.ppf([significanceLevel/2, 1-significanceLevel/2] )
```




    array([-3.37197479, -0.62802521])



* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/06_Distributions/distNormal

### 6.3.1 Examples of Normal Distributions

### 6.3.2 Central Limit Theorem

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/06_Distributions/centralLimitTheorem

### 6.3.3 Distributions and Hypothesis Tests


```python
from scipy import stats
nd = stats.norm(3.5, 0.76)
nd.cdf(2.6)
```




    0.11816486815719918



## 6.4 Continuous Distributions Derived from the NormalDistribution

### 6.4.1 t-Distribution

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/06_Distributions/distContinuous


```python
import numpy as np
from scipy import stats
n = 20
df = n-1
alpha = 0.05
stats.t(df).isf(alpha/2)
```




    2.0930240544082634




```python
stats.norm.isf(alpha/2)
```




    1.9599639845400545




```python
alpha = 0.95
df = len(data)-1
ci = stats.t.interval(alpha, df,
                      loc=np.mean(data), scale=stats.sem(data))
```

### 6.4.2 Chi-Square Distribution

* a) Definition

* b) Application Example


```python
import numpy as np
from scipy import stats
data = np.r_[3.04, 2.94, 3.01, 3.00, 2.94, 2.91, 3.02,
             3.04, 3.09, 2.95, 2.99, 3.10, 3.02]
sigma = 0.05
chi2Dist = stats.chi2(len(data)-1)
statistic = sum( ((data-np.mean(data))/sigma) **2 )
chi2Dist.sf(statistic)
```




    0.19293306654285153



### 6.4.3 F-Distribution

* a) Definition

* b) Application Example


```python
import numpy as np
from scipy import stats
method1 = np.array([20.7, 20.3, 20.3, 20.3, 20.7, 19.9,
                    19.9, 19.9, 20.3, 20.3, 19.7, 20.3])
method2 = np.array([ 19.7, 19.4, 20.1, 18.6, 18.8, 20.2,
                    18.7, 19. ])
fval = np.var(method1, ddof=1)/np.var(method2, ddof=1)
fd = stats.f(len(method1)-1,len(method2)-1)
p_oneTail = fd.cdf(fval) # -> 0.019
if (p_oneTail<0.025) or (p_oneTail>0.975):
    print('There is a significant difference'
          ' between the two distributions.')
else:
    print('No significant difference.')
```

    There is a significant difference between the two distributions.


## 6.5 Other Continuous Distributions

### 6.5.1 Lognormal Distribution

### 6.5.2 Weibull Distribution

### 6.5.3 Exponential Distribution

### 6.5.4 Uniform Distribution

## 6.6 Exercises


```python

```
