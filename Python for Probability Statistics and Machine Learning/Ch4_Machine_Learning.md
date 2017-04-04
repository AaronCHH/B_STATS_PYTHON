
# 4 Machine Learning
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [4 Machine Learning](#4-machine-learning)
  * [4.1 Introduction](#41-introduction)
  * [4.2 Python Machine Learning Modules](#42-python-machine-learning-modules)
  * [4.3 Theory of Learning](#43-theory-of-learning)
    * [4.3.1 Introduction to Theory of Machine Learning](#431-introduction-to-theory-of-machine-learning)
    * [4.3.2 Theory of Generalization](#432-theory-of-generalization)
    * [4.3.3 Worked Example for Generalization/Approximation Complexity](#433-worked-example-for-generalizationapproximation-complexity)
    * [4.3.4 Cross-Validation](#434-cross-validation)
    * [4.3.5 Bias and Variance](#435-bias-and-variance)
    * [4.3.6 Learning Noise](#436-learning-noise)
  * [4.4 Decision Trees](#44-decision-trees)
    * [4.4.1 Random Forests](#441-random-forests)
  * [4.5 Logistic Regression](#45-logistic-regression)
    * [4.5.1 Generalized Linear Models](#451-generalized-linear-models)
  * [4.6 Regularization](#46-regularization)
    * [4.6.1 Ridge Regression](#461-ridge-regression)
    * [4.6.2 Lasso](#462-lasso)
  * [4.7 Support Vector Machines](#47-support-vector-machines)
    * [4.7.1 Kernel Tricks](#471-kernel-tricks)
  * [4.8 Dimensionality Reduction](#48-dimensionality-reduction)
    * [4.8.1 Independent Component Analysis](#481-independent-component-analysis)
  * [4.9 Clustering](#49-clustering)
  * [4.10 Ensemble Methods](#410-ensemble-methods)
    * [4.10.1 Bagging](#4101-bagging)
    * [4.10.2 Boosting](#4102-boosting)
  * [References](#references)

<!-- tocstop -->


## 4.1 Introduction

## 4.2 Python Machine Learning Modules


```python
import numpy as np
from matplotlib.pylab import subplots
from sklearn.linear_model import LinearRegression
X = np.arange(10) # create some data
Y = X+np.random.randn(10) # linear with noise
```


```python
from sklearn.linear_model import LinearRegression
lr=LinearRegression() # create model
```


```python
X,Y = X.reshape((-1,1)), Y.reshape((-1,1))
lr.fit(X,Y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
lr.coef_
```




    array([[ 1.15014932]])




```python
lr.score(X,Y)
```




    0.91175077484361922




```python
xi = np.linspace(0,10,15) # more points to draw
xi = xi.reshape((-1,1)) # reshape as columns
yp = lr.predict(xi)
```


```python
X=np.random.randint(20,size=(10,2))
Y=X.dot([1, 3])+1 + np.random.randn(X.shape[0])*20
```


```python
lr=LinearRegression()
lr.fit(X,Y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
print(lr.coef_)
```

    [ 2.01977455  2.18349479]



```python
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(10).reshape(-1,1) # create some data
Y = X+X**2+X**3+ np.random.randn(*X.shape)*80
```


```python
qfit = PolynomialFeatures(degree=2) # quadratic
Xq = qfit.fit_transform(X)
print(Xq)
```

    [[  1.   0.   0.]
     [  1.   1.   1.]
     [  1.   2.   4.]
     [  1.   3.   9.]
     [  1.   4.  16.]
     [  1.   5.  25.]
     [  1.   6.  36.]
     [  1.   7.  49.]
     [  1.   8.  64.]
     [  1.   9.  81.]]



```python
lr=LinearRegression() # create linear model
qr=LinearRegression() # create quadratic model
lr.fit(X,Y) # fit linear model
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
qr.fit(Xq,Y) # fit quadratic model
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
lp = lr.predict(xi)
qp = qr.predict(qfit.fit_transform(xi))
```

## 4.3 Theory of Learning

### 4.3.1 Introduction to Theory of Machine Learning


```python
import pandas as pd
import numpy as np
from pandas import DataFrame
df=DataFrame(index=pd.Index(['{0:04b}'.format(i) for i in range(2**4)],
                            dtype='str',
                            name='x'),columns=['f'])
```


```python
df.f=np.array(df.index.map(lambda i:i.count('0'))
              > df.index.map(lambda i:i.count('1')),dtype=int)
df.head(8) # show top half only
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f</th>
    </tr>
    <tr>
      <th>x</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000</th>
      <td>1</td>
    </tr>
    <tr>
      <th>0001</th>
      <td>1</td>
    </tr>
    <tr>
      <th>0010</th>
      <td>1</td>
    </tr>
    <tr>
      <th>0011</th>
      <td>0</td>
    </tr>
    <tr>
      <th>0100</th>
      <td>1</td>
    </tr>
    <tr>
      <th>0101</th>
      <td>0</td>
    </tr>
    <tr>
      <th>0110</th>
      <td>0</td>
    </tr>
    <tr>
      <th>0111</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.random.seed(12)
def get_sample(n=1):
    if n==1:
        return '{0:04b}'.format(np.random.choice(range(8)*2+range(8,16)))
    else:
        return [get_sample(1) for _ in range(n)]
```


```python
train=df.f.ix[get_sample(8)] # 8-element training set
train.index.unique().shape # how many unique elements?
```


```python
df['fhat']=df.f.ix[train.index.unique()]
df.fhat
```


```python
df.fhat.fillna(0,inplace=True) #final specification of fhat
```


```python
test= df.f.ix[get_sample(50)]
(df.ix[test.index][’fhat’] != test).mean()
```


```python
pd.concat([test.groupby(level=0).mean(),
           train.groupby(level=0).mean()],
          axis=1,
          keys=[’test’,’train’])
```


```python
train=df.f.ix[get_sample(63)]
del df['fhat']
df[’fhat’]=df.f.ix[train.index.unique()]
df.fhat.fillna(0,inplace=True) #final specification of fhat
test= df.f.ix[get_sample(50)]
(df.fhat.ix[test] != df.f.ix[test]).mean() # error rate
```

### 4.3.2 Theory of Generalization

### 4.3.3 Worked Example for Generalization/Approximation Complexity


```python
train=DataFrame(columns=['x','y'])
train['x']=np.sort(np.random.choice(range(2**10),size=90))
train.x.head(10) # first ten elements
```




    0     25
    1     49
    2     58
    3     60
    4     74
    5     82
    6     84
    7     89
    8    100
    9    104
    Name: x, dtype: int32




```python
train.x.reshape(10,-1)
```

    C:\Anaconda36\lib\site-packages\ipykernel\__main__.py:1: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
      if __name__ == '__main__':





    array([[ 25,  49,  58,  60,  74,  82,  84,  89, 100],
           [104, 110, 119, 134, 146, 156, 175, 198, 203],
           [203, 204, 208, 225, 241, 245, 253, 259, 269],
           [278, 291, 300, 309, 335, 352, 355, 371, 373],
           [384, 390, 414, 417, 418, 419, 432, 447, 458],
           [459, 470, 473, 480, 480, 491, 539, 571, 577],
           [585, 619, 621, 621, 630, 635, 642, 642, 653],
           [667, 670, 672, 682, 685, 693, 699, 699, 752],
           [758, 772, 824, 836, 838, 843, 844, 901, 933],
           [946, 948, 958, 963, 963, 964, 980, 984, 985]])




```python
f_target=np.vectorize(lambda i:sum(map(int,i)))
```


```python
train['xb']= train.x.map('{0:010b}'.format)
train.y=train.xb.map(f_target)
train.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>xb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>3</td>
      <td>0000011001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>3</td>
      <td>0000110001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58</td>
      <td>4</td>
      <td>0000111010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>4</td>
      <td>0000111100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74</td>
      <td>3</td>
      <td>0001001010</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.y.reshape(10,-1).mean(axis=1)
```

    C:\Anaconda36\lib\site-packages\ipykernel\__main__.py:1: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
      if __name__ == '__main__':





    array([ 3.33333333,  4.33333333,  4.55555556,  4.77777778,  4.66666667,
            5.22222222,  5.11111111,  5.66666667,  5.11111111,  6.22222222])




```python
le,re=train.x.reshape(10,-1)[:,[0,-1]].T
print(le) # left edge of group
print(re)# right edge of group
```

    [ 25 104 203 278 384 459 585 667 758 946]
    [100 203 269 373 458 577 653 752 933 985]


    C:\Anaconda36\lib\site-packages\ipykernel\__main__.py:1: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
      if __name__ == '__main__':



```python
val = train.y.reshape(10,-1).mean(axis=1).round()
func = pd.Series(index=range(1024))
func[le]=val # assign value to left edge
func[re]=val # assign value to right edge
func.iloc[0]=0 # default 0 if no data
func.iloc[-1]=0 # default 0 if no data
func.head()
```

    C:\Anaconda36\lib\site-packages\ipykernel\__main__.py:1: FutureWarning: reshape is deprecated and will raise in a subsequent release. Please use .values.reshape(...) instead
      if __name__ == '__main__':





    0    0.0
    1    NaN
    2    NaN
    3    NaN
    4    NaN
    dtype: float64




```python
fi=func.interpolate('nearest')
fi.head()
```




    0    0.0
    1    0.0
    2    0.0
    3    0.0
    4    0.0
    dtype: float64




```python
test=pd.DataFrame(columns=['x','xb','y'])
test['x']=np.random.choice(range(2**10),size=500)
test.xb= test.x.map('{0:010b}'.format)
test.y=test.xb.map(f_target)
test.sort(columns=['x'],inplace=True)
```

    C:\Anaconda36\lib\site-packages\ipykernel\__main__.py:5: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)



```python
from sklearn.metrics import confusion_matrix
cmx=confusion_matrix(test.y.values,fi[test.x].values)
print(cmx)
```

    [[ 0  0  0  0  0  0  0  0  0  0  0]
     [ 2  0  0  2  1  1  0  0  0  0  0]
     [ 0  0  0 10  5  7  0  0  0  0  0]
     [ 0  0  0 10 14 34  2  0  0  0  0]
     [ 0  0  0 12 18 61  8  0  0  0  0]
     [ 0  0  0  6 19 73 17  0  0  0  0]
     [ 0  0  0  1 10 61 29  0  0  0  0]
     [ 0  0  0  0  4 44 19  0  0  0  0]
     [ 5  0  0  0  0 11 10  0  0  0  0]
     [ 1  0  0  0  0  1  1  0  0  0  0]
     [ 1  0  0  0  0  0  0  0  0  0  0]]



```python
print(cmx.diagonal()/cmx.sum(axis=1))
```

    [        nan  0.          0.          0.16666667  0.18181818  0.63478261
      0.28712871  0.          0.          0.          0.        ]


    C:\Anaconda36\lib\site-packages\ipykernel\__main__.py:1: RuntimeWarning: invalid value encountered in true_divide
      if __name__ == '__main__':



```python
print((cmx.sum(axis=0) - cmx.diagonal())/(cmx.sum()-cmx.sum(axis=1)))
```

    [ 0.018       0.          0.          0.07045455  0.13216958  0.57142857
      0.14285714  0.          0.          0.          0.        ]


### 4.3.4 Cross-Validation


```python
import numpy as np
from sklearn.cross_validation import KFold
data =np.array(['a',]*3+['b',]*3+['c',]*3) # example
print(data)
```

    ['a' 'a' 'a' 'b' 'b' 'b' 'c' 'c' 'c']


    C:\Anaconda36\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
for train_idx,test_idx in KFold(len(data),3):
    print(train_idx,test_idx)
```

    [3 4 5 6 7 8] [0 1 2]
    [0 1 2 6 7 8] [3 4 5]
    [0 1 2 3 4 5] [6 7 8]



```python
for train_idx,test_idx in KFold(len(data),3):
    print('training', data[ train_idx ])
    print('testing' , data[ test_idx ])
```

    training ['b' 'b' 'b' 'c' 'c' 'c']
    testing ['a' 'a' 'a']
    training ['a' 'a' 'a' 'c' 'c' 'c']
    testing ['b' 'b' 'b']
    training ['a' 'a' 'a' 'b' 'b' 'b']
    testing ['c' 'c' 'c']



```python
xi = np.linspace(0,1,30)
yi = np.sin(2*np.pi*xi)
```


```python
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
Xi = xi.reshape(-1,1) # refit column-wise
Yi = yi.reshape(-1,1)
lf = LinearRegression()
scores = cross_val_score(lf,Xi,Yi,cv=4,
                         scoring=make_scorer(mean_squared_error))
print(scores)
```

    [ 0.3554451   0.33131438  0.50454257  0.45905672]



```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
polyfitter = Pipeline([('poly', PolynomialFeatures(degree=3)),
                       ('linear', LinearRegression())])
polyfitter.get_params()
```




    {'linear': LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False),
     'linear__copy_X': True,
     'linear__fit_intercept': True,
     'linear__n_jobs': 1,
     'linear__normalize': False,
     'poly': PolynomialFeatures(degree=3, include_bias=True, interaction_only=False),
     'poly__degree': 3,
     'poly__include_bias': True,
     'poly__interaction_only': False,
     'steps': [('poly',
       PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)),
      ('linear',
       LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False))]}




```python
from sklearn.grid_search import GridSearchCV
gs=GridSearchCV(polyfitter,{'poly__degree':[1,2,3]},cv=4)
```

    C:\Anaconda36\lib\site-packages\sklearn\grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)



```python
_ = gs.fit(Xi,Yi)
gs.grid_scores
```

### 4.3.5 Bias and Variance


```python
from scipy import stats
def gen_sindata(n=2):
    x=stats.uniform(-1,1) # define random variable
    v = x.rvs((n,1)) # generate sample
    y = np.sin(np.pi*v) # use sample for sine
    return (v,y)
```


```python
lr = LinearRegression(fit_intercept=False)
lr.fit(*gen_sindata(2))
```




    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)




```python
lr.coef_
```




    array([[ 2.29103149]])




```python
a_out=[] # output container
for i in range(100):
    _=lr.fit(*gen_sindata(2))
    a_out.append(lr.coef_[0,0])
np.mean(a_out) # approx 1.43
```




    1.5011839851982378



### 4.3.6 Learning Noise


```python
def est_errors(d=3,n=10,niter=100):
    assert n>d
    wo = np.matrix(arange(d)).T
    Ein = list()
    Eout = list()
    # choose any set of vectors
    X = np.matrix(np.random.rand(n,d))
    for ni in xrange(niter):
        y = X*wo + np.random.randn(X.shape[0],1)
        # training weights
        w = np.linalg.inv(X.T*X)*X.T*y
        h = X*w
        Ein.append(np.linalg.norm(h-y)**2)
        # out of sample error
        yp = X*wo + np.random.randn(X.shape[0],1)
        Eout.append(np.linalg.norm(h-yp)**2)
    return (np.mean(Ein)/n,np.mean(Eout)/n)
```


```python
import numpy as np
d=10
xi = np.arange(d*2,d*10,d//2)
ei,eo = np.array([est_errors(d=d,n=n,niter=100) for n in xi]).T
```

## 4.4 Decision Trees


```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
```


```python
import numpy as np
M=np.fromfunction(lambda i,j:j>=2,(4,4)).astype(int)
print(M)
```

    [[0 0 1 1]
     [0 0 1 1]
     [0 0 1 1]
     [0 0 1 1]]



```python
i,j = np.where(M==0)
x=np.vstack([i,j]).T # build nsamp by nfeatures
y = j.reshape(-1,1)*0 # 0 elements
print(x)
```

    [[0 0]
     [0 1]
     [1 0]
     [1 1]
     [2 0]
     [2 1]
     [3 0]
     [3 1]]



```python
print(y)
```

    [[0]
     [0]
     [0]
     [0]
     [0]
     [0]
     [0]
     [0]]



```python
i,j = np.where(M==1)
x = np.vstack([np.vstack([i,j]).T,x ]) # build nsamp x nfeatures
y = np.vstack([j.reshape(-1,1)*0+1,y]) # 1 elements
```


```python
clf.fit(x, y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')




```python
clf.score(x,y)
```




    1.0




```python
M[1,0]=1 # put in different class
print(M)# now contaminated
```

    [[0 0 1 1]
     [1 0 1 1]
     [0 0 1 1]
     [0 0 1 1]]



```python
i,j = np.where(M==0)
x=np.vstack([i,j]).T
y = j.reshape(-1,1)*0
i,j = np.where(M==1)
x=np.vstack([np.vstack([i,j]).T,x])
y = np.vstack([j.reshape(-1,1)*0+1,y])
clf.fit(x,y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')




```python
y[x[:,1]>1.5] # first node on the right
```




    array([[1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1]], dtype=int64)




```python
y[x[:,1]<=1.5] # first node on the left
```




    array([[1],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0]], dtype=int64)




```python
np.logical_and(x[:,1]<=1.5,x[:,1]>0.5)
```




    array([False, False, False, False, False, False, False, False, False,
           False,  True,  True, False,  True, False,  True], dtype=bool)




```python
y[np.logical_and(x[:,1]<=1.5,x[:,1]>0.5)]
```




    array([[0],
           [0],
           [0],
           [0]], dtype=int64)



### 4.4.1 Random Forests


```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=4,max_depth=2)
rfc.fit(X_train,y_train.flat)
```

## 4.5 Logistic Regression


```python
import numpy as np
from matplotlib.pylab import subplots
v = 0.9
@np.vectorize
def gen_y(x):
    if x<5:
        return np.random.choice([0,1],p=[v,1-v])
    else:
        return np.random.choice([0,1],p=[1-v,v])
xi = np.sort(np.random.rand(500)*10)
yi = gen_y(xi)
```


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(np.c_[xi],yi)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
x0,x1=np.random.rand(2,20)*6-3
X = np.c_[x0,x1,x1*0+1] # stack as columns
```


```python
beta = np.array([1,-1,1]) # last coordinate for affine offset
prd = X.dot(beta)
probs = 1/(1+np.exp(-prd/np.linalg.norm(beta)))
c = (prd>0) # boolean array class labels
```


```python
lr = LogisticRegression()
_=lr.fit(X[:,:-1],c)
```


```python
betah = np.r_[lr.coef_.flat,lr.intercept_]
```


```python
lr = LogisticRegression(C=1000)
```

### 4.5.1 Generalized Linear Models


```python

```

## 4.6 Regularization


```python
import sympy as S
S.var('x:2 l',real=True)
J=S.Matrix([x0,x1]).norm()**2 + l*(1-x0-2*x1)
sol=S.solve(map(J.diff,[x0,x1,l]))
print(sol)
```

    {l: 2/5, x0: 1/5, x1: 2/5}



```python
from cvxpy import Variable, Problem, Minimize, norm1, norm2
x=Variable(2,1,name='x')
constr=[np.matrix([[1,2]])*x==1]
obj=Minimize(norm1(x))
p= Problem(obj,constr)
p.solve()
```


```python
print(x.value)
```


```python
constr=[np.matrix([[1,2]])*x==1]
obj=Minimize(norm2(x)) #L2 norm
p= Problem(obj,constr)
p.solve()
```


```python
print(x.value)
```


```python
x=Variable(4,1,name='x')
constr=[np.matrix([[1,2,3,4]])*x==1]
obj=Minimize(norm1(x))
p= Problem(obj,constr)
p.solve()
```


```python
print(x.value)
```


```python
constr=[np.matrix([[1,2,3,4]])*x==1]
obj=Minimize(norm2(x))
p= Problem(obj,constr)
p.solve()
```


```python
print(x.value)
```

### 4.6.1 Ridge Regression


```python
import sympy as S
from sympy import Matrix
X = Matrix([[1,2,3],
            [3,4,5]])
y = Matrix([[1,2]]).T
```


```python
b0,b1,b2=S.symbols('b:3',real=True)
beta = Matrix([[b0,b1,b2]]).T # transpose
```


```python
obj=(X*beta -y).norm(ord=2)**2
```


```python
sol=S.solve([obj.diff(i) for i in beta])
beta.subs(sol)
```




    Matrix([
    [         b2],
    [-2*b2 + 1/2],
    [         b2]])




```python
obj.subs(sol)
```




    0




```python
beta.subs(sol).norm(2)
```




    sqrt(2*b2**2 + (2*b2 - 1/2)**2)




```python
S.solve((beta.subs(sol).norm()**2).diff())
```




    [1/6]




```python
betaL2=beta.subs(sol).subs(b2,S.Rational(1,6))
betaL2
```




    Matrix([
    [1/6],
    [1/6],
    [1/6]])




```python
from sklearn.linear_model import Ridge
clf = Ridge(alpha=100.0,fit_intercept=False)
clf.fit(np.array(X).astype(float),np.array(y).astype(float))
```




    Ridge(alpha=100.0, copy_X=True, fit_intercept=False, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001)




```python
print(clf.coef_)
```


```python
from scipy.optimize import minimize
f = S.lambdify((b0,b1,b2),obj+beta.norm()**2*100.)
g = lambda x:f(x[0],x[1],x[2])
out = minimize(g,[.1,.2,.3]) # initial guess
out.x
```




    array([ 0.0428641 ,  0.06113005,  0.07939601])




```python
betaLS=X.T*(X*X.T).inv()*y
betaLS
```




    Matrix([
    [1/6],
    [1/6],
    [1/6]])




```python
X*betaLS-y
```




    Matrix([
    [0],
    [0]])




```python
print(betaLS.norm().evalf(), np.linalg.norm(clf.coef_))
```

    0.288675134594813 0.108985964126



```python
print((y-X*clf.coef_.T).norm()**2)
```

    1.86870864136429



```python
# create chirp signal
xi = np.linspace(0,1,100)[:,None]
# sample chirp randomly
xin= np.sort(np.random.choice(xi.flatten(),20,replace=False))[:,None]
# create sampled waveform
y = cos(2*pi*(xin+xin**2))
# create full waveform for reference
yi = cos(2*pi*(xi+xi**2))
# create polynomial features
qfit = PolynomialFeatures(degree=8) # quadratic
Xq = qfit.fit_transform(xin)
# reformat input as polynomial
Xiq = qfit.fit_transform(xi)
lr=LinearRegression() # create linear model
lr.fit(Xq,y) # fit linear model
# create ridge regression model and fit
clf = Ridge(alpha=1e-9,fit_intercept=False)
clf.fit(Xq,y)
```

### 4.6.2 Lasso


```python
X = np.matrix([[1,2,3],[3,4,5]])
y = np.matrix([[1,2]]).T
from sklearn.linear_model import Lasso
lr = Lasso(alpha=1.0,fit_intercept=False)
_=lr.fit(X,y)
print(lr.coef_)
```

    [ 0.          0.          0.32352941]



```python
from scipy.optimize import fmin
obj = 1/4.*(X*beta-y).norm(2)**2 + beta.norm(1)*l
f = S.lambdify((b0,b1,b2),obj.subs(l,1.0))
g = lambda x:f(x[0],x[1],x[2])
fmin(g,[0.1,0.2,0.3])
```

    Optimization terminated successfully.
             Current function value: 0.360297
             Iterations: 121
             Function evaluations: 221





    array([  2.27469304e-06,   4.02831864e-06,   3.23134859e-01])




```python
o=[]
alphas= np.logspace(-3,0,10)
for a in alphas:
    clf = Lasso(alpha=a,fit_intercept=False)
    _=clf.fit(X,y)
    o.append(clf.coef_)
```

## 4.7 Support Vector Machines


```python
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
sv = SVC(kernel='linear')
```


```python
X,y=make_blobs(n_samples=200, centers=2, n_features=2,
               random_state=0,cluster_std=.5)
sv.fit(X,y)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



### 4.7.1 Kernel Tricks


```python
import sympy as S
x0,x1=S.symbols('x:2',real=True)
y0,y1=S.symbols('y:2',real=True)
```


```python
psi = lambda x,y: (x**2,y**2,x*y,x*y)
kern = lambda x,y: S.Matrix(x).dot(y)**2
```


```python
print(S.Matrix(psi(x0,x1)).dot(psi(y0,y1)))
```

    x0**2*y0**2 + 2*x0*x1*y0*y1 + x1**2*y1**2



```python
print(S.expand(kern((x0,x1),(y0,y1))))# same as above
```

    x0**2*y0**2 + 2*x0*x1*y0*y1 + x1**2*y1**2


## 4.8 Dimensionality Reduction


```python
from sklearn import decomposition
import numpy as np
pca = decomposition.PCA()
```


```python
x = np.linspace(-1,1,30)
X = np.c_[x,x+1,x+2] # stack as columns
pca.fit(X)
print(pca.explained_variance_ratio)
```


```python
x = np.linspace(-1,1,30)
X = np.c_[np.sin(2*np.pi*x),
          2*np.sin(2*np.pi*x)+1,
          3*np.sin(2*np.pi*x)+2]
pca.fit(X)
```




    PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
print(pca.explained_variance_ratio_)
```

    [  1.00000000e+00   3.61863467e-32   2.56837706e-33]


### 4.8.1 Independent Component Analysis


```python
from numpy import matrix, c_, sin, cos, pi
t = np.linspace(0,1,250)
s1 = sin(2*pi*t*6)
s2 =np.maximum(cos(2*pi*t*3),0.3)
s2 = s2 - s2.mean()
s3 = np.random.randn(len(t))*.1
# normalize columns
s1=s1/np.linalg.norm(s1)
s2=s2/np.linalg.norm(s2)
s3=s3/np.linalg.norm(s3)
S =c_[s1,s2,s3] # stack as columns
# mixing matrix
A = matrix([[ 1, 1,1],
[0.5, -1,3],
[0.1, -2,8]])
X= S*A.T # do mixing
```


```python
from sklearn.decomposition import FastICA
ica = FastICA()
# estimate unknown S matrix
S_=ica.fit_transform(X)
```

## 4.9 Clustering


```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
from scipy.spatial.distance import cdist
m_distortions=[]
for k in range(1,7):
    kmeans = KMeans(n_clusters=k)
    _=kmeans.fit(X)
    tmp=cdist(X,kmeans.cluster_centers_,'euclidean')
    m_distortions.append(sum(np.min(tmp,axis=1))/X.shape[0])
```


```python
from sklearn.metrics import silhouette_score
```

## 4.10 Ensemble Methods


```python

```

### 4.10.1 Bagging


```python
from sklearn.linear_model import Perceptron
p=Perceptron()
p
```




    Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
          n_iter=5, n_jobs=1, penalty=None, random_state=0, shuffle=True,
          verbose=0, warm_start=False)




```python
from sklearn.ensemble import BaggingClassifier
bp = BaggingClassifier(Perceptron(),max_samples=0.50,n_estimators=3)
bp
```




    BaggingClassifier(base_estimator=Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
          n_iter=5, n_jobs=1, penalty=None, random_state=0, shuffle=True,
          verbose=0, warm_start=False),
             bootstrap=True, bootstrap_features=False, max_features=1.0,
             max_samples=0.5, n_estimators=3, n_jobs=1, oob_score=False,
             random_state=None, verbose=0, warm_start=False)



### 4.10.2 Boosting


```python
from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier(Perceptron(),n_estimators=3,
                       algorithm='SAMME',
                       learning_rate=0.5)
clf
```




    AdaBoostClassifier(algorithm='SAMME',
              base_estimator=Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
          n_iter=5, n_jobs=1, penalty=None, random_state=0, shuffle=True,
          verbose=0, warm_start=False),
              learning_rate=0.5, n_estimators=3, random_state=None)



## References


```python

```
