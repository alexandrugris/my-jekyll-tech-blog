---
layout: post
title:  "Linear Regression Done Right"
date:   2018-08-15 13:15:16 +0200
categories: statistics
---
In several of my previous posts I wrote about linear regression, but in none I wrote about when and how to use it correctly and how to interpret the results. Computing a simple regression line is easy, but applying it blindly will surely lead to incorrect results.

### Residuals

Let's assume the following two vectors, `X=[x1, .., xn] and Y=[y1, ... yn]`. Let us assume the regression line `y=ax+b`. We have the following definitions:

1. *Residuals* are a vector `R=[r1, ...,rn]` with `ri = a * xi + b - yi`, that is, the difference between the points given by the regression line and the observed values of `Y`

2. *Least square method* minimizes the function `f(a, b) = sum(ri^2)` by solving the system of equations `df(a,b)/da = 0` and `df(a, b)/db = 0`

3. *R^2*, the coefficient of determination, is the proportion of the variance in the dependent variable that is predictable from the independent variable(s) and is defined by `R^2 = Var(ax+b) / Var(Y)`. It is located between 0 and 1.

4. *Lag 1 autocorrelation* defined as `CORREL([x2, .. ,xn], [x1, .. xn-1])` and is a measure of how much values at point `i` are similar to the values at point `i-1`

In order to check that a linear regression model is correct, we need to examine the residuals. These residuals need to have the following properties:

 - Have 0 mean
 - All of them should have the same variance, meaning all the residuals are drawn from the same distribution
 - Be independent of each other
 - Be independent of X
 - Be normally distributed

In general:

- Low R^2, plot of Y vs X has no pattern - no cause-effect relationship
- Hi R^2, residuals not independent of each other - not properly defined relathionship, maybe non-linear
- Low R^2, residuals not independent of X - incomplete relationship, there is something other factor to be taken into consideration.

 ### A good regression example

Let's do an example in Excel. Let's consider `X = RANDBETWEEN(10,100)` and `Y=100 + 3 × X + RANDBETWEEN(−100, 100)`. This leads to Y being in linear relationship with X with the paramenters `b=100` and `a=3`. In Excel, slope (`a'`) and intercept (`b'`) are computed with the formulas `a' = SLOPE(Y, X)` and `b' = INTERCEPT(Y, X)`. Running these lead to `a'=3.06` and `b'=91.56` and an `R^2=0.81`, where `R^2=Var(Regressed Y) / Var(Y)` as noted before. This means that 81% of the variation of Y is explained by the regression model. We can also consider a rough interval of confidence of 95% as `(y' - 3 * stddev(y'), y' + 3 * stddev(y'))

Let's look now at the residuals:

- Lag 1 autocorrelation of the residuals is 0.12, which is low enough (check that residuals are independent of each other)
- R^2 is high enough, but not too high. An R^2 higher than 90% almost surely means that the lag 1 autocorrelation is too high and we need to revise the model
- And then we inspect the distribution of residuals vs X. Should be highly uncorrelated.

![Residuals vs X]({{site.url}}/assets/regression_1.png)

and the regression line

![Regression Line]({{site.url}}/assets/regression_2.png)

### A not so good regression example

Let's consider now the same X as before, but this time Y defined in a square relationship to Y. We apply the regression steps described before and we obtain:

- Regression line `y = 326.66 * x - 6660`
- `R^2 = 0.9586` - very high, over `0.9` which might hint the model is problematic
- `Lag 1 autocorrelation = 0.16` - low, but is low from a mistake. We should have sorted by X the data series before to see the real correlation.

But when we inspect the scatter plot of residuals vs X we see a pattern where should be none:

![Residuals vs X]({{site.url}}/assets/regression_3.png)

And the regression line showing also that the linear relationship does not properly capture the data:

![Regression Line]({{site.url}}/assets/regression_4.png)

### Improving the regression

When we have a series that has an underlying pattern, we need to deflate the series in order to bring the residuals as close as possible to their desired properties. Usually a series that has an underlying pattern exhibits large variations towards its ends. Examples of such series is price evolution of stocks which need to be deflated by the inflation, prices vs demand, any time series as they usually tend to follow either periodic patterns or exhibit some kind of growth or both. If we don't want to (or can't) find such underlying patterns, the most simple general method of deflating the series with good results is switching to percentage returns. These can be calculated by one of the following formulas:

- `%return(t) = ln(value(t)) - ln(value(t-1))` or
- `%return(t) = (value(t) - value(t-1)) / value(t-1)`

For small variations, up to 20%, the two formulas give almost identical results due to the mathematical properties of the natural logarithm. For higher variation, the logarithm shows excessive dampening of the values. See the chart below for `%returns` for a series computed through both methods.

![Percentage returns ln vs difference]({{site.url}}/assets/regression_5.png)

Better than mine explanations [here](http://people.duke.edu/~rnau/411log.htm):

>Logging a series often has an effect very similar to deflating: it straightens out exponential growth patterns and reduces heteroscedasticity (i.e., stabilizes variance).
>Logging is therefore a "poor man's deflator" which does not require any external data (or any head-scratching about which price index to use).
>Logging is not exactly the same as deflating--it does not eliminate an upward trend in the data--but it can straighten the trend out so that it can be better fitted by a linear model.  
>Deflation by itself will not straighten out an exponential growth curve if the growth is partly real and only partly due to inflation.

And

>The logarithm of a product equals the sum of the logarithms, i.e., LOG(XY) = LOG(X) + LOG(Y), regardless of the logarithm base. Therefore, logging converts multiplicative relationships to additive relationships, and by the same token it converts exponential (compound growth) trends to linear trends.

And

> `LN(X * (1+r))  =  LN(X) + LN(1+r)  ≈ LN(X) + r`
> Thus, when X is increased by 5%, i.e., multiplied by a factor of `1.05`, the natural log of X changes from `LN(X)` to `LN(X) + 0.05`, to a very close approximation.  Increasing X by 5% is therefore (almost) equivalent to adding 0.05 to LN(X).

### Going back to the regression

We will use logging in this example. Steps below:

1. Sort the series by the X term. Only doing this skyrockets the lag 1 autocorrelation to 0.75.
2. Add an additional column with formula `X' = ln(X(t)) - ln(X(t-1))`
3. Add an additional column with formula `Y' = ln(Y(t)) - ln(Y(t-1))`
4. Do the regression on `Y' = aX' + b`, which basically say *for an increase of x% in X, corresponds a linear increase of y% in Y*, this being the reason for transforming both X and Y to percentage returns.
5. Inspect the residuals of the linear regression
6. If the residuals exhibit the desired properties, compute Y as regression from X (undo the transformations). Please note that since the regression was based on increases, the whole process now is integrative, depending on the previous value. Therefore, the errors tend to accumulate the more periods we aim to forecast but also due to the high amplitude of the exponential transform. `Y-regressed(t) = EXP(y' + LN(y(t-1)))`
7. Finally inspect the residuals of the full regression

![All computation]({{site.url}}/assets/regression_6.png)

Results inspection:

Low `R^2`, lag 1 autocorrelation of only `0.08`, highly independent of X, seem rather normally distributed:
![Log residuals]({{site.url}}/assets/regression_7.png)

Final residuals show a tendency to accumulate errors towards the right extreme.
![Final residuals]({{site.url}}/assets/regression_11.png)

Plots of originally observed Y vs X and regressed Y vs X show good capture of the fundamental X^2 coefficient as well as a good fit
![Observed data]({{site.url}}/assets/regression_8.png)
and
![Regressed data]({{site.url}}/assets/regression_10.png)

And finally, observed Y vs Y regressed show a strong linear relatinon of slope almost 1, but with visibly increasing errors towards the right extreme:
![Y observed vs Y regressed]({{site.url}}/assets/regression_9.png)

### Conclusions

1. Residuals _must_ be inspected in order to make sure the regression is correct and captures the underlying movement of data.
2. Timeseries usually need to be transformed to percentage gains (or returns)
3. Obviously, while the transformation towards percentage returns still tends to accumulate errors as X grows, it still captures much better the data.
4. In the example above, a significant part of the increase in error is due to the way I generated the Y vector in the first place: `Y = 3 * X * (X+RANDBETWEEN(-10, 10)) + RANDBETWEEN(-200, 200)`. Simply by using this formula, the errors increase towards the end because the generated (observed) data has proportionally more variation towards the end.
5. As with any model, visual inspection of the end result is very important.
6. Using the percentage gain difference will lead to slightly different results.
