
# 14 Bayesian Statistics
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [14 Bayesian Statistics](#14-bayesian-statistics)
  * [14.1 Bayesian vs. Frequentist Interpretation](#141-bayesian-vs-frequentist-interpretation)
    * [14.1.1 Bayesian Example](#1411-bayesian-example)
  * [14.2 The Bayesian Approach in the Age of Computers](#142-the-bayesian-approach-in-the-age-of-computers)
  * [14.3 Example: Analysis of the Challenger Disaster with a Markov-Chain–Monte-Carlo  Simulation](#143-example-analysis-of-the-challenger-disaster-with-a-markov-chainmonte-carlo-simulation)
  * [14.4 Summing Up](#144-summing-up)

<!-- tocstop -->


## 14.1 Bayesian vs. Frequentist Interpretation

### 14.1.1 Bayesian Example

## 14.2 The Bayesian Approach in the Age of Computers

For more information on that topic, check out (in order of rising complexity)
* Wikipedia, which has some nice explanations under “Bayes :: :”
* Bayesian Methods for Hackers (http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/), a nice, free ebook, provid- ing a practical introduction to the use of PyMC (see below).
* The PyMC User Guide (http://pymc-devs.github.io/pymc/): PyMC is a very powerful Python package which makes the application of MCMC techniques very simple.
* Pattern Classification, does not avoid the mathematics, but uses it in a practical manner to help you gain a deeper understanding of the most important machine learning techniques (Duda 2004).
* Pattern Recognition and Machine Learning, a comprehensive, but often quite technical book by Bishop (2007).



## 14.3 Example: Analysis of the Challenger Disaster with a Markov-Chain–Monte-Carlo  Simulation


```python
# --- Perform the MCMC-simulations ---
temperature = challenger_data[:, 0]
D = challenger_data[:, 1] # defect or not?
# Define the prior distributions for alpha and beta
# 'value' sets the start parameter for the simulation
# The second parameter for the normal distributions is the
# "precision", i.e. the inverse of the standard deviation
beta = pm.Normal("beta", 0, 0.001, value=0)
alpha = pm.Normal("alpha", 0, 0.001, value=0)
# Define the model-function for the temperature
@pm.deterministic
def p(t=temperature, alpha=alpha, beta=beta):
    return 1.0 / (1. + np.exp(beta * t + alpha))
# connect the probabilities in `p` with our observations
# through a Bernoulli random variable.
observed = pm.Bernoulli("bernoulli_obs", p, value=D,
observed=True)
# Combine the values to a model
model = pm.Model([observed, beta, alpha])
# Perform the simulations
map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(120000, 100000, 2)
```

* https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/14_Bayesian/bayesianStats.

## 14.4 Summing Up


```python

```
