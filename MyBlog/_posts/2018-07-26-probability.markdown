---
layout: post
title:  "Probability Notes"
date:   2018-07-26 13:15:16 +0200
categories: statistics
---
A short introduction to probability. A collection of notes I took for myself, published here.

### Counting

*Given 3 locations, A, B, C, and 3 roads from A to B and 2 roads from B to C, how many possible routes are from A to C?*  `3 x 2 = 6`

*Given a bag with 3 balls, R, G and B. Provided that we take balls out of the bag one by one and don't put them back (no replacement), how many possible ways of extracting the balls there are?*  `3 x 2 x 1 = 6`, called permutations.

*Given a bag with `N` balls, provided that we take k balls from the bag without replacement, how many possible ways of extracting the balls are there?*  `N x (N-1) x .. x k = N! / (N-k)!` and the number is called arrangements.

*What if I put the balls back?*  `N^k`

*Given a room with `k` people, `k < 365`, what is the probability of at least 2 people in the room have the same birthday?*  there are `365^k` ways of arranging birthdays (sample space). Number of arrangements when each birthday is different is `365 x 364 x .. x k` (first one can be born any day of the year, second any day except the day of the first one etc). This leads to a probability of `P = (N! / (N-k)!) / (N^k) = A(365, k) / 365^k` of no two people being born in the same day. Thus, the answer is `P = 1 - P`

```python
import matplotlib.pyplot as plt
from functools import reduce

def arrangements(n, k):
    """ We will not use this function as it will overflow for large values. Provided just fo example"""
    return reduce(lambda a, b: a * b, (n-kk for kk in range(0, k)))

def at_least_two_people_in_same_day(k):
    """ If we use arrangements function, it will overflow """
    return 1 - reduce(lambda a, b: a * b, ((365-kk) / 365 for kk in range(0, k)))

at_least_two_people_in_same_day(1)
at_least_two_people_in_same_day(360)

plt.plot(range(1, 100), [at_least_two_people_in_same_day(i) for i in range(1, 100)])
```

Note: at aroung 60 people in the room, the probability is basically 1 for two people to have the same birthday.

*The number of subsets of a set* is a similar problem to extractions without replacement, but this time the order does not matter. That is, given the example with the bag of `R`, `G`, `B` balls earlier, the `R-G-B` and `B-G-R` extractions are the same. How many possible ways of extracting the balls are there? 
- We take the number of picks (arrangements): `A(N, k) = N x (N-1) x ... (N-k) = N! / (N-k)!`
- We divide this number by the total number of permutations possible in each pick as we count each pick as one:  `A(N, k) / k!`
- We call this number *combinations* `C(N, k) = N! / (k! * (N-k)!)`, sometimes also called *choose k out of N*.
Small note: `C(N, k) = C(N, N-k)`

*Given a set of `N` coin tosses, what is the probability of getting exactly `k` heads?* 
- There are `2^N` possible ways of extracting the coins - extraction with replacement.
- There are precisely `C(N, k)` ways you can get `k` heads in a string of `N` tosses
- `P = n / sample-space-size = C(N, k) / 2^N`

Obiously, if we want to compute the probability of having k heads or less, we get:

```
P = P(0) + P(1) +.. +P(k) = (C(N, 0) + C(N, 1) + ... C(N, k)) / 2^N`
```

*Given a class of `N` boys and `M` girls, pick `k` children at random, what is the probability of picking precisely `g` girls?* 

```
P(g girls out of M girls) = C(M, g)
P(k-g boys ot of N boys) = C(N, k-g)
```

The total number of possible picks is `C(M+N, k)`

```
P = (C(M, g) + C(N, k-g)) / C(M+N, k)
```

*Given a group of `N` people, how any ways can people be assigned to `n` groups of `k1, k2.., kn` people respectively?* 

```
G1 = C(N, k1)
G2 = C(N-k1, k2)
...
Gn = C(N - sum(k1, .., kn-1), kn)

Total count: 
T = G1 x G2 x ... x Gn = N! / (k1! x k2! x ... x kn!)
```

*Given `a` a's, `b` b's, `c` c's, how many ways can these be arranged?*
We have `N = a + b + c` numbers
`G1 = C(N, a)`
`G2 = C(N, b)`
`G3 = C(N, c)`

Total count: `T = G1 x G2 x G3 = (a+b+c)! / (a! x b! x c!) = C(N, a, b, c)` - multinomial coefficient.

### Conditional Probability

By definition, `P(A given B) = P (A | B) = P(A and B) / P(B)`, is called conditional probability.

We say that two events are independent if *any* of the following holds:

 - `P(A | B) = P(A)`
 - `P(B | A) = P(B)`
 - `P(A and B) = P(A) x P(B)`

If *none* of the above holds true, we say the events are dependent.

*Given 3 events and a 6-sided die, A: odd number rolled, B: even number rolled, C: {1, 2}, what is the relationship of dependence / independence between these events?*

- `P(A | B) = P(odd number rolled given even number roll) = 0`. `P(A and B) = 0` which is different from `P(A) * P(B)`, thus the events are *dependent*.
- `P(A | C) = 1/2 = P(A)`, thus the two events are *independent*

Obviously:
`P(A and B) = P(B) x P(A given B) = P(A) x P(B given A)`

*Law of total probability* says that, given a probability space `S` and a partition `S1,..., Sn` with `S = S1 U S2 U ... U Sn`, `P(A) = sum(P(Bk) * P(A | Bk), for k from 1 to n)`. In other words, probability of an event A is equal to the sum of probabilities of us being in a partition multiplied by the probability of A given we are in that partition.

*Bayes theorem:* given a sample space `S` and events `B1, ... ,Bk` forming a partition over `S`, and a random event A, 

```
P(Bi | A) = P(Bi) * P(A | Bi) / sum(P(Bj) * P(A | Bj), j = 1..k)
```

Let's apply this theorem to the well know problem of tests and probability of disease. Given the following:

 - `P(positive | have the disease) = 0.9`
 - `P(positive | you don't have the disease) = 0.1`
 - `P(you have the disease) = 1 in 10000`

If you test positive, what i the probability that you have the disease?

- The sample space `S = B1 + B2`, you have the disease or you don't have the disease, the partition fully covers the space.
- Event A = you tested positive

```
P(B1 = you have the disease) = P(B1) * P(A | B1) / sum (...) 
P(B1) = (1/10000 * 0.9) / (P(B1) * P(A | B1)) + P(B2) * P(A | B2))
P(B1) = (0.9 / 10000) / (1/10000 * 0.9 + 9999/10000 * 0.1)
p(B1) = approx 1 in 1000
```

### Random variables and distributions

*Probability function* is defined as `f(x) = P(X = x)`, where X is a random variable. E.g. : `f(x) = { 1/6 for x in 1..6, 0 otherwise }`. This is also *called probability mass function (pmf)*. 

*Uniform distribution* has `pmf(X = x, a <= x <= b, a, b natural numbers) = { 1/(b-a +1) for x = a...b, 0 otherwise }

*Binomial distribution* deals with items that can be in in two distinctive states, each with probability `p` or `1-p`. A single experiment is called a *Bernoulli trial*. For instance, given a string of `N` coin tosses, with `p` the probability of turning head, the probability of getting exactly `k` heads and thus `N-k` tails is `C(N, k) x p^k x (1-p)^(N-k)`.

- Probability of a single string of events is `p^k x (1-p)^(N-k)`
- There are `A(N, k)` arrangements of such strings, each representing one of the `k!` possible permutations, which is precisely the definition for `C(N, k)`.

*Geometric distribution* still refers to a string of Bernoulli trials, just like the binomial distribution, but, unlike the binomial distribution which answers the question *what is the probability of an event to occur X times*, the geometric distribution answers the question *what is the probability that the first success occurs after the X trials*.

The events are:

```
E1 = Success (x=1, p(E1) = p)
E2 = Failure, Success (x=2, P(E2) = (1-p) * p)
E3 = Failure, Failure, Success (x=3, P(E3) = (1-p) * (1-p) * p)`
...

Thus `pmf(x) = (1-p)^(x-1) * p`
```

*An example:*

Suppose we have an airplane which has a probability to crash on each flight equals to `p`, with this probability independent on the number of flights previously performend. Thus, on the first flight probability to crash is `p`, on the second flight probability to crash if `(1-p) * p` and so on. The question is, what is the probability the plane will at least N flights? Answer: `P(N flights) = 1 - SUM( p(x), x = 1...N )`.

5. *Hypergeometric distribution* still deals with items that can be in two possible states, but unlike the binomial distribution, extractions are without replacement. For instance, assuming a bag of balls each of two colors, red or blue, we would use the binomial distribution to compute the probability of N red balls extracted if the red ball is put back in the bag and the hypergeometric distribution if the ball is not. 

*Assuming we have a bag with `N` balls, `R` red and `B` blue, with `R+B=N` and we extract balls from the bag without putting them back, after extracting `n` balls from the bag, what is the probability of having extracted `k` red balls?*

`pmf(x) = C(R, x) * C(N-R, n-x) / C(N, n)` where:

- `R` is the number of success states (in our case the number of red balls)
- `n` is the number of draws
- `x` is the number of observed successes, in this case we would replace it with `k`
- `N` is the population size

*Assuming a a box with `N` newly built CPUs. We know that 1 in `X` is defective. We are invited to pick `K`. What is the probability of none being defective?*

`P(none defective) = C(N - N/X, K) * C(N/X, 0) / C(N, K)` with `R = N - N/X`, `x = K` and `n = K` if we are to use the terminology from the formula above.

*Continuous distributions* are those distributions for which `P(X = x) = 0` thus it would only make sense to talk about `P(a <= X <= b)`. For continuous distributions we define the *cummulative distribution function*, `cdf`, as `cdf(x) = P(X <= x)` and *probability density function*, `pdf`, as `pdf(x) = cdf(x)dx / dx = cdf'(x)` with `integral(-infinty, infinity)(pdf(x)) = 1`. Looping back to computing the `P(a <= X <= b )`, `P(a <= x <= b) = integral(a, b)(pdf(x))`.

*Standard normal distribution* has the expected value `miu = 0` and standard deviation `sigma = 1`. One can transform any normal distribution of a given `miu` and `sigma` to the standard by applying `z=(x-miu)/sigma` as in the following example: assuming that the scores for an exam are centered around 75%, with a standard deviation of 10%, what fraction of the scores lie between 80% and 90%? 

```
z1 = (80-75)/10 = 0.5
z2 = (90-75)/10 = 1.5
answer = cdf_normal(1.5) - cdf_normal(0.5)
```

*Gamma distribution*, link [here](https://en.wikipedia.org/wiki/Gamma_distribution), is a two-parameter family of continuous probability distributions. The exponential distribution, Erlang distribution, and chi-squared distribution are special cases of the gamma distribution.  

*Beta distribution*, link [here](https://en.wikipedia.org/wiki/Beta_distribution), used in Bayesian inference to solve problems which follow a pattern similar to the following: "given our belief of the world at a certain point of time, if new evidence becomes available, how do these affect our beliefs?". A good example is spam filtering.

### Expectation

Expected value is defined as `E[V] = sum(p1*v1 + p2*v2+ ... pn*vn)` where `pi` is the probability for `vi` to occur. It is called also mean or average. For example, the expectation for a binomial distribution with the probability `p` for `1`  and `1-p` for `0` is `E[X] = (1-p)*0 + 1*p = p`. 

For continuous distributions, the `E[X] = integral(-inf, +inf, dx)(x * pdf(x))`.

Considering a part that has the fail probability density function `pdf = 2*x, for 0<=x<=1 and 0 otherwise`, the expected time to fail is `E[X] = integral(0, 1, dx)(2 * x^2) = 2/3 * x^3 for x = 1 - 2/3 * x^3 for x = 0 = 2/3`. Note: `2*x` is a valid pdf, with `cdf=x^2`, monotonically increasing, and `integral(0,1)(pdf) = cdf(1) - cdf(0) = 1`. It can be explained as the probability of defect as it reaches the end-of-life increases squarely.

[Law of the unconscious statistician](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician) helps calculate the expected value of a function g(X) of a random variable X when one knows the probability distribution of X but one does not explicitly know the distribution of g(X). 

For the discrete case, the formula is:

`E[g[X]] = sum(g(x) * f_X(x))`

for the continuous case: 

`E[g[X]] = integral(-inf, inf, dx)(g(x) * f_X(x))`

*Variance* is a measure of how spread out the values are spread around the mean. We define the variance for a random variable as `V[X] = E[(x - mean)^2]`. We define also the standard deviation as `sigma[X] = sqrt(V[X])`.

For a binomial distribution, with `p=0.5`:

```
mean = E[X] = 0 * 0.5 + 1 * 0.5 = 0.5
V[X] = E[(X-mean)^2] = 0.5 * (0-0.5)^2 + 0.5 * (1-0.5)^2 = 0.5 * 0.25 + 0.5 * 0.25 = 0.25
sigma = sqrt(0.25) = 0.5`
```

Another way to computing the variance is:

```
V[X] = E[(X-mean)^2] = E[X^2 - 2*X*mean + mean^2] = E[X^2] - 2 * E[X] * mean + mean^2 = E[X^2] - mean^2
```

A nice property of variance is that `V[a1X1 + a2X2 + ... +b] = a1^2 * V[X1] + ... + an^2 * V[Xn]`.

*The joint probability mass function for two random variables X and Y* is defined as `P_XY(x,y) = P(X=x, Y=y)` with `sum(P_XY(x, y)) = 1`

To compute the measure of two variables varying together, we introduce the notion of *covariance*, which standardizes to *correlation* (covariance is dependend of the values of the two variables).

`cov(X, Y) = sigma(x, y) = E[(X-mean(X)) * (Y - mean(Y))]`

and

`corr(X, Y) = cov(X, X) / (stddev(X) * stddev(Y))`

Correlation is constrained to the interval `(-1, 1)`, with `0` meaning that `X` and `Y` are totally uncorrelated. A value over `0.6` means strongly correlated, `0.2` weakly correlated or almost uncorrelated. One point to note is that correlation only measures linear relationships. Thus a relationship like `(X, X^2)` would score very low on the correlation spectrum.

### Other interesting distributions

*Poisson distribution* models the number of occurences of an event within a given period of time. E.g. goals per match.

*Lognormal distribution* has very interesting applications, described [here](https://en.wikipedia.org/wiki/Log-normal_distribution#Occurrence_and_applications). For instance, In reliability analysis, the lognormal distribution is often used to model times to repair a maintainable system.
The lognormal distribution is a distribution of a random variable `X` whose `log(X)` is normally distributed.  







