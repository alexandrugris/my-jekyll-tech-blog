---
layout: post
title:  "P-Values And Hypothesis Testing"
date:   2017-05-14 13:15:16 +0200
categories: maths
---
This post is about p-values and hypothesis testing. What they are, why they are needed, how to compute them and how to use them. It also includes a worked example, how to validate that an A/B test indeed produces a significant outcome. The article follows quite closely a chapter from "Data Science from Scratch" by Joel Grus and it is annotated with my own research and observations.

## Definitions

1. *Null hypothesis (H0)* - the hypothesis we want to test against. Equivalent to "there is nothing out of the ordinary". For a coin toss, for instance, the hypothesis that the coin is fair. For a medicine, that it has no more effect than a sugar pill. 

2. *Alternative hypothesis (H1)* - the hypothesis we want to test for. Something is happening, the coin is unfair or that the medicine has a significant effect.

3. *Significance (probability)* - how willing are we to make a false positive claim, to reject H0 even if it is true. Due to historical reasons and scientific prudence, it is usually set to 1% or 5%.

4. *Type 1 Error* - false positive. Reject H0 even if it is true. Say that there is something when there isn't.

5. *Type 2 Error* - the opposite of type 1; failure to reject the null hypothesis even though there is something. 

6. *Power of the test* - probability of not making a type 2 error.

7. *P-value* - instead of choosing bounds based on some probability cutoff (99% or 95%), we compute the probability— assuming H0 is true—that we would see a value at least as extreme as the one we actually observed. [wikipedia](https://en.wikipedia.org/wiki/P-value)

Before going forward, I would like to point out these two links:
- [OpenIntro Statistics Book](https://www.boundless.com/users/233402/textbooks/openintro-statistics/)
- [OpenIntro Statistics Hub](https://www.boundless.com/statistics/)

## Code example - unfair coin

We are going to test if a coin is unfair (slightly biased towards the head, with a `p(head) = 0.55`). We are going to use two sample sizes and see how the sample size affects the results. 

Before that, here is some prerequisite code:

*Notes:* 

- The coin toss is represented by a `Binomial(n, p)` distribution, where `n` is the number of trials and `p` the probability of hitting head.

 - A common rule of thumb is that if both `n*p` and `n * (1 – p)` are greater than `5`, the binomial distribution may be approximated by the normal distribution. 

```python
import math

#The binomial distribution with 
#parameters n and p is the discrete probability distribution of the number of 
#successes in a sequence of n independent experiments
def normal_approximation_to_binomial(n, p):    
    """finds mu and sigma corresponding to a Binomial(n, p)""" 

    if n * p < 5 or n * (1 - p) < 5:
        raise Exception("Cannot be approximated by a normal distribution")
    
    mu = p * n    
    sigma = math.sqrt(p * (1 - p) * n)    
    return mu, sigma

def normal_probability_below(value, mu, sigma):
    """ Considering the normal distribution of coin tosses and the central limit theorem,
    computes the probability of a value to be below a certain given value. Same as normal_cdf"""
    return (1 + math.erf((value - mu) / math.sqrt(2) / sigma)) / 2     

def normal_probability_above(value, mu, sigma):
    return 1 - normal_probability_below(value, mu, sigma)

def normal_probability_between(v1, v2, mu, sigma):
    v1, v2 = (v2, v1) if v1 > v2 else (v1, v2)
    return normal_probability_below(v2, mu, sigma) - normal_probability_below(v1, mu, sigma)


#Inverse through binary search; not optimal for more than a few values
def interval_probability_centered(p, mu, sigma):
    """
    returns the interval (lower, upper) of values for which the probability 
    of the result to be in it is equal to p.
    """
    hw = 9 * sigma # half width of intw
    interval_low, interval_high = mu - hw, mu + hw
    current_prob = 1.0
    
    while abs(current_prob - p) > 1e-12:
        hw /= 2
        if p < current_prob:
            interval_low, interval_high = interval_low + hw, interval_high - hw            
        else:
            interval_low, interval_high = interval_low - hw, interval_high + hw            

        current_prob = normal_probability_between(interval_low, interval_high, mu, sigma)
        
    return interval_low, interval_high

def normal_upper_bound(p, mu, sigma):
    """ interval (-oo, value) for which sum of probabilities == p. Equivalent to inverse_normal_cdf"""   

    delta = 9 * sigma
    intw_high = mu + delta
    current_prob = 1.0
    while abs(current_prob - p) > 1e-12:
        
        delta /= 2

        while p < current_prob:
            intw_high -= delta
            current_prob = normal_probability_below(intw_high, mu, sigma)
            
        while p > current_prob:
            intw_high += delta
            current_prob = normal_probability_below(intw_high, mu, sigma)

    return intw_high

def normal_lower_bound(p, mu, sigma):
    return normal_upper_bound(1-p, mu, sigma)

```

Let's run some tests with the code above:

```python
normal_approximation_to_binomial(1000, 0.5)
Out[47]: (500.0, 15.811388300841896)

normal_approximation_to_binomial(100, 0.5)
Out[49]: (50.0, 5.0)
```

Considering the coin fair (`p=0.5`), the average is 500 heads in case of 1000 trials and 50 heads in case of 100 trials - obviously. The dispersion (sigma), if we consider it as a percentage of the number of trials, is much higher in the case of 100 trials versus the 1000. For 1000 trials, the distribution is significantly narrower.

Assuming 100 trials,

```python
normal_probability_below(50, mu, sigma)
Out[53]: 0.5

normal_probability_between(45, 55, mu, sigma)
Out[54]: 0.6826894921370861
``` 

50% of the results will be below 50 (first line) and the probability for the result (no. of observed heads) to be between 45 and 55 is 0.68.

```python
interval_probability_centered(0.95, mu, sigma)
Out[55]: (40.20018007732688, 59.79981992267312)
```

For the same `n=100` trials, the number of observed heads will be between 40 and 60 for 95% of the experiment runs.

First hypothesis - coin is biased, but we don't specify towards which side:

```python
# if coin not biased, considering the mu and sigma computed above, for 100 trials and p(head) = 0.5
low, high = interval_probability_centered(0.95, mu, sigma)

# consider the coin biased towards the head, with p(head) = 0.55
mu_biased, sigma_biased = normal_approximation_to_binomial(100, 0.55)
```

Now we run the following test: what is the probability that we will get a value between `low` and `high` as the margins of 95% confidence if the coin is unbiased, if we consider the coin biased. As we see below, the higher the number of trials, the higher the confidence we have in rejecting the null hypothesis. Power of the test below is the probability of not making a type 2 error, in which we fail to reject H0 even though it’s false.

```python
normal_probability_between(low, high, mu_biased, sigma_biased)
Out[57]: 0.8312119922463654

1 - normal_probability_between(low, high, mu_biased, sigma_biased) # power of the test
Out[58]: 0.16878800775363456
```

Obviously, we cannot discharge the null hypothesis. p-value = 0.83 and the power of the test only 0.168. However, if we run the code for `n=1000` and `p_biased(head) = 0.55`, we get a different picture due to the more narrow distribution. We still cannot reject the null hypothesis, though.

```python
normal_probability_between(low, high, mu_biased, sigma_biased)
Out[62]: 0.11345221890056983

1 - normal_probability_between(low, high, mu_biased, sigma_biased) # power of the test
Out[63]: 0.8865477810994302
```

If we had 2000 trials on the other hand,

```python
normal_probability_between(low, high, mu_biased, sigma_biased)
Out[65]: 0.005787749170240164

1 - normal_probability_between(low, high, mu_biased, sigma_biased) # power of the test
Out[66]: 0.9942122508297598
```
We can pretty confidently reject the null hypothesis.

Now we assume the hypothesis we want to test is a little bit more specific - the coin is biased towards the head. The difference from the previous scenario is that now we make a one-sided test. In the previous case we were less specific, thus the two-sided test has wider margins.

Considering 1000 trials,

```python
hi = normal_upper_bound(0.95, mu, sigma)

normal_probability_below(hi, mu_biased, sigma_biased)
Out[74]: 0.06362100214309152

1 - 0.06362100214309152
Out[75]: 0.9363789978569085
```

While we still cannot confidently reject the null hypothesis with this test, but it is obviously a more powerful one.

## P-values:

Instead of choosing bounds based on some probability cutoff, we compute the probability, assuming H0 is true, that we would see a value at least as extreme as the one we actually observed. From wikipedia,

![pvalue](https://upload.wikimedia.org/wikipedia/commons/3/3a/P-value_in_statistical_significance_testing.svg)

```python
def two_sided_p_value(x, mu=0, sigma=1):    
    if x >= mu:
        # if x is greater than the mean, the tail is what's greater than x        
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # if x is less than the mean, the tail is what's less than x        
        return 2 * normal_probability_below(x, mu, sigma)

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below 
```

The first thing that is obvious is that the `two_sided_p_value` features a multiplication by 2, whereas the single-sided functions don't. We talked about two sided versus one sided tests in the previuos chapter. There is one very important result to consider: we can only decide to opt for a single-sided test _IF_ we didn't look at the data before. That means, that the hypothesis we want to test is not derived from our understanding of the data, but rather is a pure theoretical (blind) speculation. If we already have an idea of what the data is about, we need to opt for the two sided test. This comes from the fact that the data might be already unintentionally biased towards our result (or in the opposite side of our result) by precisely half of our confidence interval. 

Here is a more detailed explanation: [hypothesis testing with p-values](https://www.boundless.com/users/233402/textbooks/openintro-statistics/foundations-for-inference-4/hypothesis-testing-35/two-sided-hypothesis-testing-with-p-values-174-13787/)

Let's use the p-values. While in the previous chapter we estimated the validity of the test for a certain hypothesis, now we are looking at actual results. The problem sounds like: assuming that we obtain `530` tails in a `1000` trials experiment, is the coin biased? 

```python
two_sided_p_value(529.5, mu, sigma) # continuity correction
Out[76]: 0.06207721579598857
```

Unfortunately we cannot dismiss the null hypothesis. Had we observed `531` tails, then the result would have looked different.

*Note:* decreasing `530` by `0.5` in the test above is called continuity correction. It basically asserts that `p(530)` is better estimated by the average of `p(529.5)` and `p(530.5)` than by the average of `p(530)` and `p(531)`. [Continuity Correction - Wikipedia](https://en.wikipedia.org/wiki/Continuity_correction)

## Example - A/B testing

Let's put all these things together in a worked example - computing the success of an A/B test campaign.

Let's assume we have a banner for which we want to make a change and we want to understand if the change brings more clicks and thus profit. Our null hypothesis is "no, the change does not impact the number of clicks".

We split the clients in two cohorts (`A` and `B`). For cohort `A` we keep the old banner, for cohort `B` we have the banner changed. After `N(A)` views for banner `A` we have `n(A)` click-throughs and after `N(B)` views for banner `B` we have `n(B)` click-throughs. Let's assume `N(A) == 1000`, `N(B) == 1000`, `n(A) == 180`, `n(B) == 200`. Is `B` a better campaign?

Obviously we have a binomial variable with two posible outcomes: click (1) or not click (0), with a probability to click of `p`. Thus, for one trial, we have:

```
mu = p #expected value
sigma = sqrt(1/2 * ((0 - p)^2 + (1 - p)^2)) = sqrt(p * (1 - p))
```

For `N` trials we apply the central limit theorem which states:

```
mu_clt = 1/N * (x0 + x1 +... + xN) = 1/N * (1 + 0 + 0 + ... +1) = n/N = mu = p
sigma_clt = sigma / sqrt(N) = sqrt(p * (1 - p)) / sqrt(N) = sqrt(p * (1 - p) / N)
```

Now we want to test that distribution for `A` is the same as distribution for `B`, that is `p(A) == p(B)` - null hypothesis.

`p(A) == p(B)` means `p(A) - p(B) == 0` or, more precisely, has `mu == 0`. But `p(A) - p(B)` [is a normally distributed random variable](https://en.wikipedia.org/wiki/Normal_distribution#Operations_on_normal_deviates), thus 

```
mu(p(A) - p(B)) = mu(A) - mu(B)
sigma(p(A) - p(B)) = sqrt(sigma(A) ^ 2 + sigma(B) ^ 2)
```

If we are to [`z-score`](http://stattrek.com/statistics/dictionary.aspx?definition=Normal_random_variable) this distribution, we should obtain a distribution with `mu == 0` and `sigma == 1` - which is a mathematical expression of our null hypothesis. If our experimental `p(A) - p(B)` is within the constraints of this distribution, then `B` most likely does not provide any improvement (or worsening) over `A`. Now we can use our `two_sided_p_value` to test our hypothesis.

The code:

```python
def abtest_estimated_params_trial(N, n):
    mu = n / N
    sigma = math.sqrt(mu * ( 1- mu) / N)
    return (mu, sigma)

def pA_minus_pB(Na, na, Nb, nb):
    muA, sigmaA = abtest_estimated_params_trial(Na, na)
    muB, sigmaB = abtest_estimated_params_trial(Nb, nb)
    
    # the following number should be normally distributed with mu = 0 and sigma == 1
    return (muA - muB) / math.sqrt(sigmaA ** 2 + sigmaB ** 2)

two_sided_p_value(pA_minus_pB(1000, 200, 1000, 150), 0, 1)
```

With the results:

```python
two_sided_p_value(pA_minus_pB(1000, 200, 1000, 250), 0, 1)
Out[6]: 0.007313777221710671 # > 0.05, we can safely reject the null hypothesis

two_sided_p_value(pA_minus_pB(1000, 200, 1000, 350), 0, 1)
Out[7]: 2.531308496145357e-14 # > 0.5, we can safely reject the null hypothesis

two_sided_p_value(pA_minus_pB(1000, 200, 1000, 210), 0, 1)
Out[8]: 0.5796241923602059 # cannot reject the null

two_sided_p_value(pA_minus_pB(1000, 200, 1000, 180), 0, 1)
Out[9]: 0.25414197654223614 # cannot reject the null

two_sided_p_value(pA_minus_pB(1000, 200, 1000, 170), 0, 1)
Out[10]: 0.08382984264631776 # still cannot reject the null

two_sided_p_value(pA_minus_pB(1000, 200, 1000, 150), 0, 1)
Out[11]: 0.003189699706216853 # we can safely reject the null
```

