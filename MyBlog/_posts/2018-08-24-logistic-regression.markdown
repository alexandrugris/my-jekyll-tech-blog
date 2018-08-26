---
layout: post
title:  "Logistic Regression"
date:   2018-08-24 13:15:16 +0200
categories: statistics
---
While linear regression is about predicting effects given a set of causes, logistic regression predicts the probability of certain effects. This way, its main applications are classification and forecasting. Logistic regression helps find how probabilities are changed by our actions or by various changes in the factors included in the regression.

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

![Logistic function]({{site.url}}/assets/logistic_regression_8.png)

And the wikipedia link: [Logistic function](https://en.wikipedia.org/wiki/Logistic_function)

Therefore, we will find a mathematical model that uses the logistic function to map from causes to probabilities. The model aims to find a vector `B` that, when plugged into the logistic function multiplied by the factors `X`, the observed results match as closely as possible to the results determined by our model. In this case, our logistic function looks like `f(X_i) = 1 / (1+e^-(b1 * xi_1 + ... bn * xi_n)`, where `Xi = [xi_1, ... xi_n]` are the observed `X` factors at point `i`.

We will further define our problem like this:

 - `Xi = [xi_1 ... xi_n]` - factors, with i between 1 and k. Attention, one of the factors needs to be `1`, the intercept - see the excel example below
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

`PRODUCT(P(yi | Xi), for i = 0..k) = P(y1 | X1) * P(y2 | X2) * ...` is maximized, as close as possible to 1.

which is equivalent to 

`log(PRODUCT) = SUM(log(P(yi | Xi)) = SUM( yi * log(f(Xi)) + (1-yi) * log(1-f(Xi)))` where `f(Xi) = 1 / (1+e^-dot(Xi,B))` is maximized as close as possible to 0. 

Now we can use gradient descent to find the vector `B`. Unfortunately this method might be slow for large datasets because, for every incremental change in any `B`, all the exponentials and the logarithms for each line in the dataset need to be recomputed.

### Implementation in Excel on the Iris dataset

1. We download the Iris dataset.
2. We setup 3 variables, one for each type of flower. They are 1 if the flower is of that type and 0 otherwise.
3. We split the data into trainig and test dataset.
4. Add a new column for intercept.
5. Add a new row for the `B` vector. These are the values we want to find.
6. Add a new column for `P(setosa)`, the probabilities we want as result from the logistic regression. `P(setosa) = 1/1+EXP(-SUMPRODUCT(B, X))`
7. Add a new column, the log column, `log(P(yi | Xi)`, the column for which the sum we want to optimize as close as possible to 0. The formula for each of the elements in the column is `yi *LN(P(setosa)+0.0001) + (1-yi)*LN(1-P(setosa)+0.0001)`. I added a `0.0001` correction factor because, as the maximization algorithm goes on, we might end up with `NaN` values due to computing `P(setosa) = 0` which leads to `LN(0)` which is `-infinity`.
8. Add a summary value for the column at 7, as `SUM(log(P(yi | Xi)))`.
9. Use Excel Solver plugin to maximize the value of this summary value to as close to 0 as possible.
10. Because the setosa is strictly separated from the other two types, the solver steps continues until the regression becomes very tight and the coefficients very large. In order for the `1/1+EXP(-SUMPRODUCT)` not to overflow, I added a constraint for the `-SUMPRODUCT()` to be less or equal to 50. This does not seem to impact the quality of the regression.
11. Check the test data.

Excel file [here]({{site.url}}/assets/iris.xlsx)

![Solver Dialog]({{site.url}}/assets/logistic_regression_1.png)

*Note:*

Performing the same steps for *virginica* leads also to very good results, while for *versicolor* the results are barely above random choice. 

In the second iteration of the file, I have modified the regression so that it operates on standardized factors. This way, we can interpret the coefficients and see which factor contributes the most for classification. The picture below also contains the `log(odds)` vs `most important factor` chart, sorted by `odds = P(1) / P(0)`, the ratio between the probability of `1` and `0`. This is also a reminder that data for logistic regression should pe pepared in a similar manner to that for linear regression. 

![Standardized Data]({{site.url}}/assets/logistic_regression_2.png)

### Linear Regression and Logistic Regression

By the choice for the logistic function, `p = 1 / (1+EXP(-SUMPROD(Beta, X)))`, with `Beta` the coefficients for each of the factors X included in the regression, including the intercept, Logistic Regression hints stongly towards the linear regression. 

We introduce the function `Odds(p) = p / 1-p` with its inverse, `p = Odds / (1+Odds)`.

By simple arithmetic, we conclude that `Odds(p) = EXP(SUMPRODUCT(Beta, X))` which leads to `LN(Odds(p)) = SUMPRODUCT(Beta, X)` which is a linear function.

Please remember that `p` is the probability for the event to happen, that is proability for the flower to be setosa, in the example above.

This function, `ln(Odds(p)) = ln(p/1-p)`, called the *logit function*, maps the probability space `[0,1]` to a linear space `[-inf, inf]` in which we can do the regression.

![Logit function]({{site.url}}/assets/logistic_regression_3.png)

In order to use linear regression to compute the probabilities and thus build our classifier, we need in our input data probabilities as well. The Iris dataset, as presented above, has only 0 and 1 inputs, so we need to collapse it to intervals on which meaninful probabilities can be computed. 

In this example we will use another dataset which is already collapsed:

![Hypertensive men]({{site.url}}/assets/logistic_regression_4.png)

The data is described as follows:
- Number of men analysed
- Number of men with hypertension
- Wether the man is smoking, obese or snores

We want to:
- Find the real probabilities for a man to be hypertensive given the factors above - please note that we are talking about a small sample of the population, so the real probabilities are not simply the division between men with hypertension and men.
- Map these probabilities to individual conditions of a patient - he may be fat but not obese or he might just be a casual smoker, metrics which can be represented as fractions of the smoking, obese, snores variables above.

### Steps

Linear regression on the raw, unprocessed, original data. If we include an intercept, the intercept will reflect the amount of men included in the test, so, for a single man, the results for the prediction will be completely off. We will see that these results are pretty close to what we will obtain from logistic regression, because the probabilities themselves are in the lower part of the spectrum, where the logistic regression itself is quite linear. However, we expect that, as the risk factors increase significantly, for instance by codifying a "heavy smoker" or a "highly obese", the results from the linear regression to diverge from the real probabilities.

Please see how the variables for smoking, obese and snoring are codified. We want them to affect the slope of the regression, not the intercept.

![Linear Regression]({{site.url}}/assets/logistic_regression_5.png)

We will do two types of logistic regression - one that does not account for the amount of men included in the sample and one which does. The problem with not ballancing the regression for the amount of men is equivalent to discarding the precision of the initial etimation of probability given by the sample (no of hypertensives / no of men). E.g., the more men you include, the more confident one can be that the expected value of the sample is closer to the actual expected value of the population.

First step is to remove from the regression the line where only 2 men are counted. This line does not contain enough information to be able to codify a probability out of it or, better said, the margin of error is too high.

Second step, we add the following columns:

- `Log(Odds Observed) = LN(P_observed / 1 - (P_observed))` where `P_observed = hypertensives / total number of men in that category`.
- Smoking, obesity, snoring, codified as 1 if the person is smoking, obese or snoring.
- Intercept, all values equal to 1.

As result, we have:
- `Log(Odds regressed)`, the result of the linear regression
- `P Logistic` which is the probability as computed from the logistic regression.

We also added two more tables:

- One under the logistic regression for drawing and validating how probabilities change for various fractions of Snoring, Smoking and Obese predictor variables.
- One under the linear regression to validate how many standard deviations the observed ratio (probability) is from the theoretical probability obtained from the logistic regression. That is, assuming the probabilities from the regression are correct, how likely it is to have observed the value we have observed. For this, we used the theoretical mean and variance for the *binomial distribution*, `mean = sample_size * p` and `variance = sample_size * p * (1-p)`

First run, the results without taking into consideration the sample size:

![Logistic Regression]({{site.url}}/assets/logistic_regression_6.png)

The second time, we took into consideration the sample size by weightning the square of the residuals with the sample size when computing the square sum of the residuals we want to minimize. This is equivalent to having 1 row in the regression for each men that was considered in the regression. We see that this also minimizes the sum of the square standard scores, meaning that the results are now closer to the reality that was observed in the field. As a side note, I tried computing the `Bs` by minimizing directly the sum of the square standard scores and the results were very close to the ones predicted by the logistic regression.

![Weighted Logistic Regression]({{site.url}}/assets/logistic_regression_7.png)

Excel file [here](({{site.url}}/assets/hypertensives.xlsx))

### Conclusion

We used logistic regression to build a clasifier on the Iris dataset and to predict the probability of a person having hypertension given a set of predictors. We used two ways to compute the regression coefficients: one by maximizing directly a probability function using the gradient descent, the other by applying linear regression to the log-odds function and then computing the probabilities from it.