---
layout: post
title:  "Reverse Engineering Expected Goals"
date:   2020-02-29 13:15:16 +0200
categories: statistics
---
In this post we are going to look again a the 2018-2019 Premier League season and try to reverse engineer bookmakers odds to get the expected goals for each team. Then we are going to try to improve on these models and reduce our reliance on bookmakers odds.

We are going to use publicly available data, downloaded from [here](https://datahub.io/sports-data/english-premier-league#readme). In case the link disappears, here is a [local copy]({{site.url}}/assets/season-1819_csv.csv) of the csv file used for this analysis. The explanation for the columns can be found [here]({{site.url}}/assets/data_explanation.txt)

### Reverse Engineering Expected Goals Using Direct Optimization

We are looking independently at each line, the match odds for home-draw-away and over-under, and optimize the mean squared error loss function for each match. We are going to consider the match goals as Poisson distributed.

```python
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# https://datahub.io/sports-data/english-premier-league#readme

data = pd.read_csv("season-1819_csv.csv") # todo: add index on team names!

rows = len(data)
teams = int(np.sqrt(rows)) + 1

###
# BbAvH = Betbrain average home win odds
# BbAvD = Betbrain average draw win odds
# BbAvA = Betbrain average away win odds

# BbAv>2.5 = Betbrain average over 2.5 goals
# BbAv<2.5 = Betbrain average under 2.5 goals

# FTHG and HG = Full Time Home Team Goals
# FTAG and AG = Full Time Away Team Goals
###

data = data[['HomeTeam', 'AwayTeam','BbAvH', 'BbAvD', 'BbAvA', 'BbAv>2.5', 'BbAv<2.5', 'FTHG',  'FTAG']]

### xG reverse engineering
def hda_ou25(xGH, xGA):
    
    h_distrib = np.array([poisson.pmf(x, xGH) for x in range(0, 10)])
    a_distrib = np.array([poisson.pmf(x, xGA) for x in range(0, 10)])
    
    ret = [0, 0, 0, 0, 0]
    
    for i in range (0, len(h_distrib)):
        for j in range (0, len(a_distrib)):
            p = h_distrib[i] * a_distrib[j]
            
            # hda part
            if i > j:
                ret[0] += p
            elif i == j:
                ret[1] += p
            elif i < j:
                ret[2] += p
                
            # ou 2.5 part
            if i + j > 2.5:
                ret[3] += p
            elif i + j < 2.5:
                ret[4] += p
                
    return 1/np.array(ret)

def loss_hdaou(xG,hda_ou_target):
    v = hda_ou25(xG[0], xG[1]) - hda_ou_target
    return np.dot(v, v) # square the error
    

def compute_xg(row):    
    hda_ou_target = np.array([row[c] for c in ['BbAvH', 'BbAvD', 'BbAvA', 'BbAv>2.5', 'BbAv<2.5']]).flatten()
    ret = minimize(loss_hdaou, [1.25, 1.25], args=(hda_ou_target), method='COBYLA').x
    return pd.Series(ret, index=['xGH', 'xGA'])
    
xg = data.apply(compute_xg, axis=1)

data['xGH_optim'] = xg['xGH']
data['xGA_optim'] = xg['xGA']
```

### Reverse Engineering Expected Goals Using Poisson Regression

The next step is to do exactly the same thing, but this time using Poisson regression. Poisson regression is very useful for things like counts. It aims to compute a set of regression coefficients such that `lambda = sum(regression_coef_i * predictor_variable_i)`. Poisson regression is solved through the Maximum Likelihood Estimate method, as shown below:

```python
# poisson regression:
def poisson_compute_xg(betas, factors, goals):

    # compute the lambda for each row based on the proposed betas 
    lmbdas = np.dot(betas, factors.T)
    
    if(np.min(lmbdas) <= 0e-5):
        print("violated constraint")
        return 1e10
    
    # compute the probability for obtaining the observed counts (goals)
    # given the lambda
    psn =  [poisson.pmf(g, l) for (g,l) in zip(goals, lmbdas)]

    # return the MLE sum of logarithms
    return -np.sum(np.log(psn))

def poisson_regress(X, y):
    factors = X.to_numpy()

    # start with 1 for all factors
    betas = [1] * factors.shape[1]

    # MLE
    return minimize(poisson_compute_xg, betas, args=(factors, y), method='cobyla').x

# transform from odds to probabilities to get a better regression
X = 1/data[['BbAvH', 'BbAvD', 'BbAvA', 'BbAv>2.5', 'BbAv<2.5']]

y_h = data['FTHG']
y_a = data['FTAG']

betas_h = poisson_regress(X, y_h)
betas_a = poisson_regress(X, y_a) 

data['xGH_regress'] = np.dot(X, betas_h)
data['xGA_regress'] = np.dot(X, betas_a)
```

The results for the two methods are highlighted in the dataframe below:

![Expected Goals Reverse Engineered]({{site.url}}/assets/xg_1.png)

Now let's look at the RMSE for the two methods:

```python

# in sample RMSE
rmse_optim = np.sqrt(np.sum((data['FTHG'] - data['xGH_optim']) ** 2 + (data['FTAG'] - data['xGA_optim']) ** 2) / len(data))
rmse_regress = np.sqrt(np.sum((data['FTHG'] - data['xGH_regress']) ** 2 + (data['FTAG'] - data['xGA_regress']) ** 2) / len(data))

# out of sample RMSE
X['xGH_optim'] = data['xGH_optim']
X['FTHG'] = data['FTHG']
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y_h)

betas_h = poisson_regress(X_train_h[['BbAvH', 'BbAvD', 'BbAvA', 'BbAv>2.5', 'BbAv<2.5']], y_train_h)
result = np.dot(X_test_h[['BbAvH', 'BbAvD', 'BbAvA', 'BbAv>2.5', 'BbAv<2.5']], betas_h)

from sklearn.metrics import mean_squared_error
rmse_optim = np.sqrt(mean_squared_error(X_test_h['FTHG'], X_test_h['xGH_optim']))
rmse_regress = np.sqrt(mean_squared_error(X_test_h['FTHG'], result))
```

Let's look at the results now. For in-sample RMSE, we always have, based on this dataset, at least, lower RMSE for the Poisson regression. However, the difference is very small:

```
rmse_optim = np.sqrt(np.sum((data['FTHG'] - data['xGH_optim']) ** 2 + (data['FTAG'] - data['xGA_optim']) ** 2) / len(data))
rmse_regress = np.sqrt(np.sum((data['FTHG'] - data['xGH_regress']) ** 2 + (data['FTAG'] - data['xGA_regress']) ** 2) / len(data))

rmse_optim
Out[190]: 1.5902709407482063
rmse_regress
Out[191]: 1.5836188549138615
```

For out-of-sample RMSE, on the test dataset we usually have lower RMSE for Poisson regression, but this is not always the case. Again, the difference is small and we can consider the two methods roughly equivalent.

```
X['xGH_optim'] = data['xGH_optim']
X['FTHG'] = data['FTHG']
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y_h)

betas_h = poisson_regress(X_train_h[['BbAvH', 'BbAvD', 'BbAvA', 'BbAv>2.5', 'BbAv<2.5']], y_train_h)
result = np.dot(X_test_h[['BbAvH', 'BbAvD', 'BbAvA', 'BbAv>2.5', 'BbAv<2.5']], betas_h)

from sklearn.metrics import mean_squared_error

rmse_optim = np.sqrt(mean_squared_error(X_test_h['FTHG'], X_test_h['xGH_optim']))
rmse_regress = np.sqrt(mean_squared_error(X_test_h['FTHG'], result))

rmse_optim
Out[196]: 1.162618442066351

rmse_regress
Out[197]: 1.1526039105430579
```
