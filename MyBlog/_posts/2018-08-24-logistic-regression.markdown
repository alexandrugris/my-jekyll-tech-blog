---
layout: post
title:  "Logistic Regression"
date:   2018-08-24 13:15:16 +0200
categories: statistics
---
While linear regression is about predicting effects given a set of causes, logistic regression predicts the probability of certain effects. This way, its main applications are classification and forecasting. Logistic regression helps find how probabilities are changed by our actions.

### The problem

The problem we want to solve is: given a vector of factors `X=[X1 ... Xn]`, find a model that predicts the probability of a binary outcome to occur, `P(X, outcome = 1)`. The input for the model are the previous observations of `X` and whether the expected outcome was `0` or `1`.

One can obviosly try to reduce the problem directly to a linear regression problem, as exemplified in the PDF file [here]({{site.url}}/assets/logistic_regression_1.pdf) but the shortcomings are immediately apparent:

- R^2 is low
- There is not an obvious way to link the results to probability, the regression line goes above 1 and below 0.
- Residuals are obviously not normally distributed

If in this precise case, one predictor variable, one result, one can simply find a cut-off point, the problem becomes more difficult if we include several predictor variables in the equation.

### The Logistic Regression Approach

We notice that the Logistic function, `f(x) = 1 / (1+e^-(ax+b))`, has properties that naturally map to our problem:

- it grows assimptotically from 0 to 1 - which are the natural boundaries of probabilty
- it is continuous
- has an inflection point from which `f(x)` grows quickly from close to 0 to close to 1

[Logistic function](https://en.wikipedia.org/wiki/Logistic_function)

Therefore, we will find a mathematical model that uses the logistic function to map from causes to probabilities. The model aims to find a vector `B` that, when plugged into the logistic function multiplied by the factors `X`, the observed results match as closely as possible to the results determined by our model. In this case, our logistic function looks like `f(X_i) = 1 / (1+e^-(b1 * xi_1 + ... bn * xi_n)`, where `Xi = [xi_1, ... xi_n]` are the observed `X` factors at point `i`.

We will further define our problem like this:

 - `Xi = [xi_1 ... xi_n]` - factors, with i between 1 and k
 - `B = [b1 ... bn]` - coefficients
 - `Y = [y1 ... yk]` - a vector of 1 and 0 of k elements
 - `dot(Xi, B) == Xi_1 * b1 + ... Xi_n * bn`

 This is a fancy way of describing a table with n colums defined by X, the regression factors, and 1 column defined by Y, the observed result, with k rows.

The core of the model is assuming that the probability of Y being 1 is given by the logistic function:
 - `P(1|Xi) = f(Xi) = 1 / (1+e^-dot(Xi,B))` 

Which makes:
 - `P(0|Xi) = 1 - P(1|Xi)`

So, in compact form:
 - `P(yi | Xi) = (P(1 | Xi) ^ yi) * (P(0 | Xi) ^ (1 - yi)`), since `yi` is either `0` or `1`

We conclude that `P(yi | X)` is another column, that of probabilities, but we cannot fully compute it because we are missing the vector `B`.

A perfect fit means `P(yi | Xi) == 1`, whatever `i=1..k`, which reads *given the set of inputs Xi we estimate with absolute certainty that the result is yi, 0 or 1, just as observed in the real world.* This leads to a *real world* equation of `PRODUCT(P(yi | Xi), for i = 0..k) == 1`

 But since we only aim to model the real world, we are happy if this product is merely maximized by our estimated probability function (logistic) with `B` plugged in. That is,

`PRODUCT(P(yi | Xi), for i = 0..k) = P(y1 | X1) * P(y2 | X2) * ...` is maximized

which is equivalent to 

`log(PRODUCT) = log(P(y1 | X1)) + ...` is maximized - we converted from product to sum. Now we can use gradient descent to find the vector `B`. Unfortunately this method is slow because for every incremental change in `B`, all the data points need to be recomputed, which leads to a slow step.

