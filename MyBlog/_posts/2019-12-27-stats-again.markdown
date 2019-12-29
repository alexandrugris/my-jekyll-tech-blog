---
layout: post
title:  "Stats Again"
date:   2019-12-01 13:15:16 +0200
categories: statistics
---
A summary of notions not found anywhere else in this blog.

### Descriptive Statistics

For unimodal distributions we define skewness, which is a measure of whether the distribution has its its peak towards the left or the right.

```
Skewness = 1/n * sum( (xi - x_mean) / stdev)^3
```

Negative value means the sample has its mode to the left, positive to the right. Left-skewed means the tail is on the left and the mode is on the right. Right-skewed, the opposite. 

There is an equality which says:

```
mode < mean < median if the distribution is right skewed and
mean < median < mode if the distribution is left skewed
```

For bivariate datasets we have correlation and covariance, measures of how X and Y move together monotonically.

```
sample_covariance = 1/(n-1) * sum((xi - x_mean) (yi-y_mean))
```

```
sample_correlation = r = sample_covariance / (x_stddev * y_stddev)
```

Sample correlation, r, is also called Pearson's correlation coefficient.

Perfect correlation, 1 or -1, mean that X and Y move together on a straight line. We can derive the regression line between X and Y as follows:

```
Y = aX + b
a = covar(X, Y) / Var(X)
b = mean(Y) - a * mean(X) # the regression line always passes through the means of X and Y
```

For the regression, we define: 

```
r_squared = r^2 = 1 - Var(residuals) / Var(Y)
```

For linear least squares regression with an intercept term and a single explanatory, it is the same `r` as the Pearson's correlation coefficient defined above. It measures the amount of explained variance of Y for this particular regression.

Similarly to the Pearson coefficient which is defined for the values of `X` and `Y`, we define the Spearman correlation coefficient but instead of values we take the ranks. Spearman is more robust, thus less sensitive to outliers.

For nominal (categorical) values, we can also introduce a measure of dependency and correlation. We use the Chi2 statistic to measure the independence of two categorical variables. The *Chi2 test* works on contingency tables. 

```python
from scipy.stats import chi2_contingency

# first create the contingency table based on the two categorical variables
df_for_test = pd.crosstab(df['categorical_variable_1'], df['categorical_variable_2'])
chi2, p_value, degrees_of_freedom, expected_values = chi2_contingency(df_for_test.values)
```

Return values described below:
- `expected_values` - how would the data look like if it were independent
- `chi2` - the test statistic, the higher it is, the lower the `p-value`, the probability that the values are independent
- `degrees_of_freedom` - `dof = observed.size - sum(observed.shape) + observed.ndim - 1` the degrees of freedom for the Chi2 distribution from which the p-value is computed given the test statistic.

### Binomial and Hypergeometric Distributions

The binomial distribution is frequently used to model the number of successes in a sample of size n drawn with replacement from a population of size N. If the sampling is carried out without replacement, the draws are not independent and so the resulting distribution is a hypergeometric distribution, not a binomial one. However, for N much larger than n (a rule of thumb is 10%), the binomial distribution remains a good approximation.

If n is large enough, then the skew of the distribution is not too great. In this case a reasonable approximation to `B(n, p)` is given by the normal distribution `N(n*p, n*p*(1-p))`. A commonly used rule is that both `n*p` and `n*(1-p)` are larger than 10.

The binomial distribution converges towards the Poisson distribution as the number of trials goes to infinity while the product `n*p` remains fixed or at least p tends to zero. Therefore, the Poisson distribution with parameter `λ = n*p` can be used as an approximation to `B(n, p)` of the binomial distribution if `n` is sufficiently large and `p` is sufficiently small. According to two rules of thumb, this approximation is good if `n >= 20` and `p <= 0.05`, or if `n >= 100` and `n*p <= 10`.


### Inferential Statistics

*Maximum Likelihood Estimate*

MLE starts from the premise that if you observe a certain sample, then its probability must be high. It is used to estimate a parameter of the distribution of the population given the observations. 

The MLE problem is described as follows:

```
maximize(P(x1 | distrib(parameter)) * P(x2 | distrib(parameter) * .... * P(xn | distrib(parameter))))
```

which is equivalent to 
```
maximize(log(P(x1 | ...)) + log(P(x2 |...)) + .... )
```

Since MLE usually uses gradient descent to find the maximum, it helps if the distribution for which we plan to estimate the parameter is continuous. Otherwise, the maximum must be found through different means.

Below is solved the following problem with MLE. We extract 1000 samples from a `t` distribution with 5 degrees of freedom. We want to estimate the degrees of freedom knowing that the samples come from a `t` distribution. 

```python
import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from functools import partial

# the samples we extract
sample = np.random.standard_t(5, 1000)
plt.hist(sample)

# problem solved below
def log_likelihood(f, sample: np.array) -> float:
    return np.sum(np.log(f(sample)))

def t_distrib(df : int, sample: np.array) -> np.array:
    return stats.t.pdf(sample, df=df)

ret = minimize(lambda df: -log_likelihood(partial(t_distrib, df), sample), 10.0)
print(ret.x[0])
```

A good estimator has low variance and low bias, preferably 0. Bias is the difference between the true value of the parameter and the expected value of its estimator. 


*Hypothesis Testing*

H0 - null hypothesis, it is considered true until proven false. It is usually in the form 'there is no relationship' or 'there is no effect'.

H1 - alternative hypothesis, it asserts that there is a relationship or a significant effect.

Hypothesis testing does not prove the H1, but rather says there is a low probability (usually under 5%) that H0 can occur. There is still a possibility, usually under 5%, that H0 still occurs, but it is considered low. The p-values, which are used in hypothesis testing, represent the probability that we observe H1 given that H0 is true. The 5% above is called significance level and it depends on the application.

As an example, let's assume that someone can tell if [the milk was poured before or after the tea in a cup](https://en.wikipedia.org/wiki/Lady_tasting_tea). There are 8 cups, 4 with milk poured before and 4 with milk poured after. Assuming the person manages to correctly identify which one is which in this experiment, can we conclude that the person is able to tell whether the milk was poured before or after?

H0: correct identification in the experiment is purely due to chance
H1: the person is able to identify which is which

```
p_observed_results_given_h0 = 1 / Combinations of 8 taken by 4 = 1/70 < 5% 
```

We can conclude that it is unlikely that the person made the choices purely by chance.

*T-Tests*

It is used to learn about averages across two categories, for instance height of males vs height of females in sample. It has several forms:

- *One sample location test* - used to test whether a population mean is significantly different from some hypothesized value. The degrees of freedom is `DF = n - 1` where `n` is the number of observations in the sample.

```
t = (sample_mean - hypothesized value) / Standard Error 
t = sqrt(n) * (sample_mean - hypothesized value) / (sample_std_dev)
```

In python we use

```python
scipy.stats.ttest_1samp(a, popmean, axis=0, nan_policy='propagate')
```

If the sample size is large (e.g. > 30 observations) and the population standard deviation is known, you can assume the test statistics follows the normal distribution instead of the t-distribution with n-1 degrees of freedom, thus one can apply the Z-test.

- *Two sample location test* - used to test whether the population means of two samples are significantly different. E.g. mean of the the height of adults from town A is different from the mean of heights of adults from town B. The sample size should be the same and the variance in each sample should be equal. If the variances differ, one needs to apply the *Welch t-test*. To test for the equality of variances between the two populations, one can use the *Levene* test. For the *Levene* test, the null hypothesis is that the sample variances are equal, so we can use the two sample location test if we cannot discard the null hypothesis, that is the p-values from the *Levene* tests are higher than 5%.

```python
# we assume we have the two samples, sample1 and sample2 already extracted
# sample1 and sample2 are of the same type, same unit of measure, same scale

from sklearn.preprocessing import scale # for z-scoring

stats.levene(sample1, sample2) # here we need to accept the null, which means p>0.05

diff = scale(sample1 - sample2) # this diff should be normally distributed
plt.hist(diff)
stats.probplot(diff, plot=plt, dist='norm') # check the QQ plot
stats.shapiro(diff) #test for normality, if the test statistic is not significant, p>0.05, then the population is normally distributed

stats.ttest_ind(sample1, sample2) # perform the test
```

If the population variances are not equal, we need to use:

```python
stats.ttest_ind(sample1, sample2, equal_var=False) # Welch t-test
```

- *Paired difference test* - the previous two t-tests assume the variables are independent. If they are not independent, e.g. babies from the same town or the same sample before and after treatment to check if the treatment was successful, one needs to use the paired difference test.

Same assumptions as before, in python use
```python
stats.ttest_rel(sample1, sample2)
```

- *Regression coefficient tests* - this tests whether the coefficients from a regression are significantly different from 0.

Assumptions of the t-test:
- Populations are normal
- Samples are representative
- Samples randomly drawn

T tests work well for two group comparison, for multiple groups one needs to use *ANOVA*. 

*Test for correlation / dependence*

If sample (X, Y) come from a 2 dimensional normal distribution, `Corr(X, Y) == 0` means independence. 

*Kolmogorov-Smirnov Goodness of Fit test*

Tells us if the samples come from a specified distribution. In python, we can use `stats.kstest`

*One-way ANOVA*