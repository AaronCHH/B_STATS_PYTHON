
# 3 Statistics
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [3 Statistics](#3-statistics)
  * [3.1 Introduction](#31-introduction)
  * [3.2 Python Modules for Statistics](#32-python-modules-for-statistics)
    * [3.2.1 Scipy Statistics Module](#321-scipy-statistics-module)
    * [3.2.2 Sympy Statistics Module](#322-sympy-statistics-module)
    * [3.2.3 Other Python Modules for Statistics](#323-other-python-modules-for-statistics)
  * [3.3 Types of Convergence](#33-types-of-convergence)
    * [3.3.1 Almost Sure Convergence](#331-almost-sure-convergence)
    * [3.3.2 Convergence in Probability](#332-convergence-in-probability)
    * [3.3.3 Convergence in Distribution](#333-convergence-in-distribution)
    * [3.3.4 Limit Theorems](#334-limit-theorems)
  * [3.4 Estimation Using Maximum Likelihood](#34-estimation-using-maximum-likelihood)
    * [3.4.1 Setting Up the Coin Flipping Experiment](#341-setting-up-the-coin-flipping-experiment)
    * [3.4.2 Delta Method](#342-delta-method)
  * [3.5 Hypothesis Testing and P-Values](#35-hypothesis-testing-and-p-values)
    * [3.5.1 Back to the Coin Flipping Example](#351-back-to-the-coin-flipping-example)
    * [3.5.2 Receiver Operating Characteristic](#352-receiver-operating-characteristic)
    * [3.5.3 P-Values](#353-p-values)
    * [3.5.4 Test Statistics](#354-test-statistics)
    * [3.5.5 Testing Multiple Hypotheses](#355-testing-multiple-hypotheses)
  * [3.6 Confidence Intervals](#36-confidence-intervals)
  * [3.7 Linear Regression](#37-linear-regression)
    * [3.7.1 Extensions to Multiple Covariates](#371-extensions-to-multiple-covariates)
  * [3.8 Maximum A-Posteriori](#38-maximum-a-posteriori)
  * [3.9 Robust Statistics](#39-robust-statistics)
  * [3.10 Bootstrapping](#310-bootstrapping)
    * [3.10.1 Parametric Bootstrap](#3101-parametric-bootstrap)
  * [3.11 Gauss Markov](#311-gauss-markov)
  * [3.12 Nonparametric Methods](#312-nonparametric-methods)
    * [3.12.1 Kernel Density Estimation](#3121-kernel-density-estimation)
    * [3.12.2 Kernel Smoothing](#3122-kernel-smoothing)
    * [3.12.3 Nonparametric Regression Estimators](#3123-nonparametric-regression-estimators)
    * [3.12.4 Nearest Neighbors Regression](#3124-nearest-neighbors-regression)
    * [3.12.5 Kernel Regression](#3125-kernel-regression)
    * [3.12.6 Curse of Dimensionality](#3126-curse-of-dimensionality)
  * [References](#references)

<!-- tocstop -->


## 3.1 Introduction

## 3.2 Python Modules for Statistics

### 3.2.1 Scipy Statistics Module


```python
import scipy.stats # might take awhile
n = scipy.stats.norm(0,10) # create normal distrib
n
```




    <scipy.stats._distn_infrastructure.rv_frozen at 0x762f7c9940>




```python
n.mean() # we already know this from its definition!
```




    0.0




```python
n.moment(4)
```




    30000.0




```python
n.pdf(0)
```




    0.039894228040143268




```python
n.cdf(0)
```




    0.5




```python
n.rvs(10)
```




    array([ -3.34335197,  16.10507687,   3.08586178,   4.23101862,
            13.63803945,  -4.3565258 ,  27.38408865,   9.44491787,
             8.66860182,   0.50589409])




```python
scipy.stats.shapiro(n.rvs(100))
```




    (0.9853184819221497, 0.3347501754760742)



### 3.2.2 Sympy Statistics Module


```python
from sympy import stats
X = stats.Normal('x',0,10) # create normal random variable
from sympy.abc import x
stats.density(X)(x)
stats.cdf(X)(0)
stats.P(X>0) # prob X >0?
stats.E(abs(X)**(1/2)).evalf()
```




    2.59995815363879



### 3.2.3 Other Python Modules for Statistics

## 3.3 Types of Convergence

### 3.3.1 Almost Sure Convergence


```python
from scipy import stats
u = stats.uniform()
xn = lambda i: u.rvs(i).max()
xn(5)
```




    0.83090085498038646




```python
import numpy as np
np.mean([xn(60) > 0.95 for i in range(1000)])
```




    0.95599999999999996




```python
print(np.log(1-.99)/np.log(.95))
```

    89.7811349607



```python
import numpy as np
np.mean([xn(90) > 0.95 for i in range(1000)])
```




    0.98799999999999999



### 3.3.2 Convergence in Probability


```python
make_interval= lambda n: np.array(zip(range(n+1),range(1,n+1)))/n
```


```python
intervals = np.vstack([make_interval(i) for i in range(1,5)])
print(intervals)
```


```python
bits= lambda u:((intervals[:,0] < u) & (u<=intervals[:,1])).astype(int)
bits(u.rvs())
```


```python
print(np.vstack([bits(u.rvs()) for i in range(10)]))
```


```python
np.vstack([bits(u.rvs()) for i in range(1000)]).mean(axis=0)
```

### 3.3.3 Convergence in Distribution

### 3.3.4 Limit Theorems

## 3.4 Estimation Using Maximum Likelihood

### 3.4.1 Setting Up the Coin Flipping Experiment


```python
from scipy.stats import bernoulli
p_true=1/2.0 # estimate this!
fp=bernoulli(p_true) # create bernoulli random variate
xs = fp.rvs(100) # generate some samples
print(xs[:30]) # see first 30 samples
```

    [1 0 0 0 0 1 1 1 0 0 1 1 0 1 1 0 1 1 0 1 1 1 0 1 1 0 1 1 0 0]



```python
import sympy
x,p,z=sympy.symbols('x p z', positive=True)
phi=p**x*(1-p)**(1-x) # distribution function
L=np.prod([phi.subs(x,i) for i in xs]) # likelihood function
print(L) # approx 0.5?
```

    p**51.0*(-p + 1)**49



```python
logL = sympy.expand_log(sympy.log(L))
sol, = sympy.solve(sympy.diff(logL,p),p)
print(sol)
```

    0.510000000000000



```python
fig,ax=subplots()
x=np.linspace(0,1,100)
ax.plot(x,map(sympy.lambdify(p,logJ,'numpy'),x),'k-',lw=3)
ax.plot(sol,logJ.subs(p,sol),'o',
        color='gray',ms=15,label='Estimated')
ax.plot(p_true,logJ.subs(p,p_true),'s',
        color='k',ms=15,label='Actual')
ax.set_xlabel('$p$',fontsize=18)
ax.set_ylabel('Likelihood',fontsize=18)
ax.set_title('Estimate not equal to true value',fontsize=18)
ax.legend(loc=0)
```


```python
from scipy.stats import binom
# n=100, p = 0.5, distribution of the estimator phat
b = binom(100,.5)
# symmetric sum the probability around the mean
g = lambda i:b.pmf(np.arange(-i,i)+50).sum()
print(g(10)) # approx 0.95
```

    0.953955933071



```python
from scipy.stats import bernoulli
b=bernoulli(0.5) # coin distribution
xs = b.rvs(100) # flip it 100 times
phat = np.mean(xs) # estimated p
print(abs(phat-0.5) < 0.5*0.20 )# make it w/in interval?
```

    True



```python
out=[]
b=bernoulli(0.5) # coin distribution

for i in range(500): # number of tries
    xs = b.rvs(100) # flip it 100 times
    phat = np.mean(xs) # estimated p
    out.append(abs(phat-0.5) < 0.5*0.20 ) # within 20% ?

# percentage of tries w/in 20\,% interval
print(100*np.mean(out))
```

    97.2



```python
import numpy as np
from scipy import stats
rv = stats.uniform(0,1) # define uniform random variable
mle = rv.rvs((100,500)).max(0) # max along row-dimension
print(np.mean(mle)) # approx n/(n+1) = 100/101 ˜= 0.99
print(np.var(mle)) #approx n/(n+1)**2/(n+2) ˜= 9.61E-5
```

    0.98969419136
    8.04200833005e-05


### 3.4.2 Delta Method


```python
from scipy import stats
# compute MLE estimates
d=stats.bernoulli(0.1).rvs((10,5000)).mean(0)
# avoid divide-by-zero
d=d[np.logical_not(np.isclose(d,1))]
# compute odds ratio
odds = d/(1-d)
print('odds ratio=',np.mean(odds),'var=',np.var(odds))
```

    odds ratio= 0.123205555556 var= 0.0175364184171



```python
from scipy import stats
d = stats.bernoulli(.5).rvs((10,5000)).mean(0)
d = d[np.logical_not(np.isclose(d,1))]
print('odds ratio=',np.mean(d),'var=',np.var(d))
```

    odds ratio= 0.501161161161 var= 0.0247834364895


## 3.5 Hypothesis Testing and P-Values

### 3.5.1 Back to the Coin Flipping Example


```python
from sympy.stats import P, Binomial
theta = S.symbols('theta',real=True)
X = Binomial('x',100,theta)
beta_function = P(X>60)
print(beta_function.subs(theta,0.5)) # alpha
print(beta_function.subs(theta,0.70))
```


```python
from scipy import stats
rv=stats.bernoulli(0.5) # true p = 0.5
# number of false alarms ˜ 0.018
print(sum(rv.rvs((1000,100)).sum(axis=1)>60)/1000)
```

    0.018


### 3.5.2 Receiver Operating Characteristic

### 3.5.3 P-Values

### 3.5.4 Test Statistics


```python
import sympy as S
from sympy import stats
s = stats.Normal('s',1,1) # signal+noise
n = stats.Normal('n',0,1) # noise
x = S.symbols('x',real=True)
L = stats.density(s)(x)/stats.density(n)(x)
```


```python
g = S.symbols('g',positive=True) # define gamma
v = S.integrate(stats.density(n)(x),
                (x,S.Rational(1,2)+S.log(g),S.oo))
```


```python
print(S.nsolve(v-0.01,3.0)) # approx 6.21
```

    6.21116124253284



```python
from scipy.stats import binom, chi2
import numpy as np
# some sample parameters
p0,p1,p2 = 0.3,0.4,0.5
n0,n1,n2 = 50,180,200
brvs= [ binom(i,j) for i,j in zip((n0,n1,n2),(p0,p1,p2))]
def gen_sample(n=1):
    'generate samples from separate binomial distributions'
    if n==1:
        return [i.rvs() for i in brvs]
    else:
        return [gen_sample() for k in range(n)]
```


```python
k0,k1,k2 = gen_sample()
print(k0,k1,k2)
```

    13 64 101



```python
pH0 = sum((k0,k1,k2))/sum((n0,n1,n2))
numer = np.sum([np.log(binom(ni,pH0).pmf(ki)) for ni,ki in zip((n0,n1,n2),(k0,k1,k2))])
print(numer)
```

    -14.9487619239



```python
denom = np.sum([np.log(binom(ni,pi).pmf(ki)) for ni,ki,pi in zip((n0,n1,n2),(k0,k1,k2),(p0,p1,p2))])
print(denom)
```

    -8.67115888386



```python
chsq=chi2(2)
logLambda =-2*(numer-denom)
print(logLambda)
```

    12.5552060801



```python
print(1- chsq.cdf(logLambda))
```

    0.00187789644647



```python
c = chsq.isf(.05) # 5% significance level
out = []

for k0,k1,k2 in gen_sample(100):
    pH0 = sum((k0,k1,k2))/sum((n0,n1,n2))
    numer = np.sum([np.log(binom(ni,pH0).pmf(ki)) for ni,ki in zip((n0,n1,n2),(k0,k1,k2))])

    denom = np.sum([np.log(binom(ni,pi).pmf(ki)) for ni,ki,pi in zip((n0,n1,n2),(k0,k1,k2),(p0,p1,p2))])
    out.append(-2*(numer-denom)>c)

print(np.mean(out)) # estimated probability of detection
```

    0.55



```python
x=binom(10,0.3).rvs(5) # p=0.3
y=binom(10,0.5).rvs(3) # p=0.5
z = np.hstack([x,y]) # combine into one array
t_o = abs(x.mean()-y.mean())
out = [] # output container

for k in range(1000):
    perm = np.random.permutation(z)
    T=abs(perm[:len(x)].mean()-perm[len(x):].mean())
    out.append((T>t_o))

print('p-value = ', np.mean(out))

```

    p-value =  0.06



```python
from scipy import stats
theta0 = 0.5 # H0
k=np.random.binomial(1000,0.3)
theta_hat = k/1000. # MLE
W = (theta_hat-theta0)/np.sqrt(theta_hat*(1-theta_hat)/1000)
c = stats.norm().isf(0.05/2) # z_{alpha/2}
print(abs(W)>c) # if true, reject H0
```

    True



```python
theta0 = 0.5 # H0
c = stats.norm().isf(0.05/2.) # z_{alpha/2}
out = []

for i in range(100):
    k=np.random.binomial(1000,0.3)
    theta_hat = k/1000. # MLE
    W = (theta_hat-theta0)/np.sqrt(theta_hat*(1-theta_hat)/1000.)
    out.append(abs(W)>c) # if true, reject H0

print(np.mean(out)) # detection probability
```

    1.0


### 3.5.5 Testing Multiple Hypotheses

## 3.6 Confidence Intervals


```python
from scipy import stats
import numpy as np
b= stats.bernoulli(.5) # fair coin distribution
nsamples = 100
# flip it nsamples times for 200 estimates
xs = b.rvs(nsamples*200).reshape(nsamples,-1)
phat = np.mean(xs,axis=0) # estimated p
# edge of 95\,% confidence interval
epsilon_n=np.sqrt(np.log(2/0.05)/2/nsamples)
pct=np.logical_and(phat-epsilon_n<=0.5,
                   0.5 <= (epsilon_n +phat)
                  ).mean()*100
print('Interval trapped correct value ', pct,'% of the time')
```

    Interval trapped correct value  100.0 % of the time



```python
# compute estimated se for all trials
se=np.sqrt(phat*(1-phat)/xs.shape[0])
# generate random variable for trial 0
rv=stats.norm(0, se[0])
# compute 95\,% confidence interval for that trial 0
np.array(rv.interval(0.95))+phat[0]

def compute_CI(i):
    return stats.norm.interval(0.95,loc=i,
                               scale=np.sqrt(i*(1-i)/xs.shape[0]))
lower,upper = compute_CI(phat)
```

## 3.7 Linear Regression


```python
import numpy as np
a = 6;b = 1 # parameters to estimate
x = np.linspace(0,1,100)
y = a*x + np.random.randn(len(x))+b
p,var_=np.polyfit(x,y,1,cov=True) # fit data to line
y_ = np.polyval(p,x) # estimated by linear regression
```


```python
x0, xn =x[0],x[80]
# generate synthetic data
y_0 = a*x0 + np.random.randn(20)+b
y_1 = a*xn + np.random.randn(20)+b
# mean along sample dimension
yhat = np.array([y_0,y_1]).mean(axis=1)
a_,b_=np.linalg.solve(np.array([[x0,1],
                                [xn,1]]),yhat)
```


```python
from scipy import stats
slope,intercept,r_value,p_value,stderr = stats.linregress(x,y)
```


```python
import statsmodels.formula.api as smf
from pandas import DataFrame
import numpy as np
d = DataFrame({'x':np.linspace(0,1,10)}) # create data
d['y'] = a*d.x+ b + np.random.randn(*d.x.shape)
```


```python
results = smf.ols('y ˜ x', data=d).fit()
```

### 3.7.1 Extensions to Multiple Covariates


```python
fit = lambda i,x,y: np.polyval(np.polyfit(x,y,1),i)
omit = lambda i,x: ([k for j,k in enumerate(x) if j !=i])
def cook_d(k):
    num = sum((fit(j,omit(k,x),omit(k,y))- fit(j,x,y))**2 for j in x)
    den = sum((y-np.polyval(np.polyfit(x,y,1),x))**2/len(x)*2)
    return num/den
```

## 3.8 Maximum A-Posteriori


```python
import sympy
from sympy import stats as st
from sympy.abc import p,k,n
obj = sympy.expand_log(sympy.log(p**k*(1-p)**(n-k)* st.density(st.Beta('p',6,6))(p)))
sol = sympy.solve(sympy.simplify(sympy.diff(obj,p)),p)[0]
sol
```




    (k + 5)/(n + 10)



## 3.9 Robust Statistics


```python
import statsmodels.api as sm
from scipy import stats
data=np.hstack([stats.norm(10,1).rvs(10),stats.norm(0,1).rvs(100)])
```


```python
huber = sm.robust.scale.Huber()
loc,scl=huber(data)
```

## 3.10 Bootstrapping


```python
import numpy as np
from scipy import stats
rv = stats.beta(3,2)
xsamples = rv.rvs(50)
```


```python
yboot = np.random.choice(xsamples,(100,50))
yboot_mn = yboot.mean()
```


```python
np.std(yboot.mean(axis=1)) # approx sqrt(1/1250)
```




    0.024178108705133088




```python
import sympy as S
import sympy.stats

for i in range(50): # 50 samples
    # load sympy.stats Beta random variables
    # into global namespace using exec
    execstring = "x%d = S.stats.Beta('x'+str(%d),3,2)"%(i,i)
    exec(execstring)

# populate xlist with the sympy.stats random variables
# from above
xlist = [eval('x%d'%(i)) for i in range(50) ]
# compute sample mean
sample_mean = sum(xlist)/len(xlist)
# compute expectation of sample mean
sample_mean_1 = S.stats.E(sample_mean)
# compute 2nd moment of sample mean
sample_mean_2 = S.stats.E(S.expand(sample_mean**2))
# standard deviation of sample mean
# use sympy sqrt function
sigma_smn = S.sqrt(sample_mean_2-sample_mean_1**2) # 1/sqrt(1250)
print(sigma_smn)

```

    sqrt(-1/(20000*beta(3, 2)**2) + 1/(1500*beta(3, 2)))



```python
from scipy import stats
import numpy as np
p= 0.25 # true head-up probability
x = stats.bernoulli(p).rvs(10)
print(x)
```

    [0 0 0 0 0 0 0 0 0 0]



```python
phat = x.mean()
print(phat)
```

    0.0



```python
print((1-2*phat)**2*(phat)**2/10.)
```

    0.0



```python
phat_b = np.random.choice(x,(50,10)).mean(1)
print(np.var(phat_b*(1-phat_b)))
```

    0.0



```python
import sympy as S
from sympy.stats import E, Bernoulli
xdata =[Bernoulli(i,p) for i in S.symbols('x:10')]
ph = sum(xdata)/float(len(xdata))
g = ph*(1-ph)
```


```python
print(E(g**2) - E(g)**2)
```

    0.00442968750000000


### 3.10.1 Parametric Bootstrap


```python
rv = stats.norm(0,2)
xsamples = rv.rvs(45)
# estimate mean and var from xsamples
mn_ = np.mean(xsamples)
std_ = np.std(xsamples)
# bootstrap from assumed normal distribution with
# mn_,std_ as parameters
rvb = stats.norm(mn_,std_) #plug-in distribution
yboot = rvb.rvs(1000)
```


```python
# MLE-Plugin Variance of the sample mean
print(2*(std_**2)**2/9.) # MLE plugin
```

    2.33433879437



```python
# Bootstrap variance of the sample mean
print(yboot.var())
```

    3.1688336913



```python
# True variance of sample mean
print(2*(2**2)**2/9.)
```

    3.5555555555555554


## 3.11 Gauss Markov


```python
Q = np.eye(3)*0.1 # error covariance matrix
# this is what we are trying estimate
beta = np.matrix(np.ones((2,1)))
W = np.matrix([[1,2],
               [2,3],
               [1,1]])
ntrials = 50
epsilon = np.random.multivariate_normal((0,0,0),Q,ntrials).T
y=W*beta+epsilon
```


```python
from matplotlib.patches import Ellipse
U,S,V = linalg.svd(bm_cov)
err = np.sqrt((matrix(bm))*(bm_cov)*(matrix(bm).T))
theta = np.arccos(U[0,1])/np.pi*180

ax.add_patch(Ellipse(bm,err*2/np.sqrt(S[0]),
                     err*2/np.sqrt(S[1]),
                     angle=theta,color=’gray’))
```

## 3.12 Nonparametric Methods

### 3.12.1 Kernel Density Estimation


```python
def generate_samples(n,ntrials=500):
    phat = np.zeros((nbins,ntrials))
    for k in range(ntrials):
        d = rv.rvs(n)
        phat[:,k],_=histogram(d,bins,density=True)
    return phat
```

### 3.12.2 Kernel Smoothing


```python
from scipy.integrate import quad
from scipy import stats
rv= stats.beta(2,2)
n=100 # number of samples to generate
d = rv.rvs(n)[:,None] # generate samples as column-vector
```


```python
train,test,_,_ = train_test_split(d,d,test_size=0.5)
kdes=[KernelDensity(bandwidth=i).fit(train) for i in [.05,0.1,0.2,0.3]]
```


```python
import numpy as np
for i in kdes:
    f = lambda x: np.exp(i.score_samples(x))
    f2 = lambda x: f(x)**2
    print('h=%3.2f\t %3.4f'%(i.bandwidth,quad(f2,0,1)[0]
                             -2*np.mean(f(test))))
```


```python
class KernelDensityWrapper(KernelDensity):
    def predict(self,x):
        return np.exp(self.score_samples(x))
    def score(self,test):
        f = lambda x: self.predict(x)
        f2 = lambda x: f(x)**2
        return -(quad(f2,0,1)[0]-2*np.mean(f(test)))
```


```python
from sklearn.grid_search import GridSearchCV
params = {'bandwidth':np.linspace(0.01,0.5,10)}
clf = GridSearchCV(KernelDensityWrapper(), param_grid=params,cv=2)
clf.fit(d)
print(clf.best_params_)

from pprint import pprint
pprint(clf.grid_scores_)
```

### 3.12.3 Nonparametric Regression Estimators

### 3.12.4 Nearest Neighbors Regression


```python
import numpy as np
from numpy import cos, pi
xi = np.linspace(0,1,100)[:,None]
xin = np.linspace(0,1,12)[:,None]
f0 = 1 # init frequency
BW = 5
y = cos(2*pi*(f0*xin+(BW/2.0)*xin**2))
```


```python
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(2)
knr.fit(xin,y)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=2, p=2,
              weights='uniform')




```python
knr=KNeighborsRegressor(3)
knr.fit(xin,y)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=3, p=2,
              weights='uniform')




```python
from sklearn.cross_validation import LeaveOneOut
loo=LeaveOneOut(len(xin))
```


```python
pprint(list(LeaveOneOut(3)))
```

    Pretty printing has been turned OFF



```python
out=[]
for train_index, test_index in loo:
    _=knr.fit(xin[train_index],y[train_index])
    out.append((knr.predict(xi[test_index])-y[test_index])**2)
print('Leave-one-out Estimated Risk: ',np.mean(out),)
```

    Leave-one-out Estimated Risk:  1.03517136627



```python
_= knr.fit(xin,y) # fit on all data
S=(knr.kneighbors_graph(xin)).todense()/float(knr.n_neighbors)
```


```python
print(S[:5,:5])
```

    [[ 0.33333333  0.33333333  0.33333333  0.          0.        ]
     [ 0.33333333  0.33333333  0.33333333  0.          0.        ]
     [ 0.          0.33333333  0.33333333  0.33333333  0.        ]
     [ 0.          0.          0.33333333  0.33333333  0.33333333]
     [ 0.          0.          0.          0.33333333  0.33333333]]



```python
print(np.hstack([knr.predict(xin[:5]),(S*y)[:5]]))#columns match
```

    [[ 0.55781314  0.55781314]
     [ 0.55781314  0.55781314]
     [-0.09768138 -0.09768138]
     [-0.46686876 -0.46686876]
     [-0.10877633 -0.10877633]]



```python
print(np.allclose(knr.predict(xin),S*y))
```

    True


### 3.12.5 Kernel Regression


```python
from kernel_regression import KernelRegression
```


```python
kr = KernelRegression(gamma=np.linspace(6e3,7e3,500))
kr.fit(xin,y)
```

### 3.12.6 Curse of Dimensionality


```python
import numpy as np
v=np.random.rand(1000,2)-1/2.
```


```python
for d in [2,3,5,10,20,50]:
    v=np.random.rand(5000,d)-1/2.
    hist([np.linalg.norm(i) for i in v])
```

## References


```python

```
