---
layout: post
title:  "Risk-based Odds Adjustment Using Bayesian Inference"
date:   2017-06-11 13:15:16 +0200
categories: statistics
---
This blog introduces the basics of Bayesian inference and attempts to sketch a solution to a problem a bookmaker might have: starting from a predefined set of odds, how do we adjust them so that the risk to the bookmaker is minimized, by taking into consideration the financial stakes the bettors have placed on a game. We will only look at a 1x2 market (home-draw-away) to keep the code clean. 

### Bayesian Inference basics

In simple terms, Bayesian Inference tries to solve problems which revolve around the following pattern: "given our belief of the world at a certain point of time, if new evidence becomes available, how do these affect our beliefs"? Or, put differenly, "given a model of the world with a set of a priori parameters, if new evidence becomes available, how will the parameters change to incorporate this new evidence?"  

*Examples of problems that can be solved through Bayesian Inference*

- One might believe a coin is slightly biased with and heads will turn with a probability PH while tails will turn with a probability PT. By starting a set of trial,
 one can adjust the PH and PT according to the results of each trial, having the model migrate towards the real probabilities. 

- Evaluating the success of a marketing campaign, by counting the clicks on a set of ads, so that the most successful ad is presented to the customer. In such a case, we can start with a "all the same" approach and then adjust display probabilities according to click data that comes through in real time and with minimum computational overhead. 

- Given a number of clicks or misses on an add, what is the probability of the next user to click on it?

- Archaeological dating based on the types of objects found on the site [wikipedia](https://en.wikipedia.org/wiki/Bayesian_inference#Making_a_prediction)

*Bayes Theorem*

Vocabulary:

 - Posterior probability P(hypothesis | evidence): what we want to know, the probability of a hypothesis given the observed evidence.
 - Prior probability P(hypothesis): the estimate of the probability before the evidence appears
 - Probability of observing the evidence given the hypothesis P(evidence | hypothesis)
 - Probability of the event to happen: P(E)

```
P(H | E) = P(E | H) * P(H) / P(E)
```

Worked examples: 

1. Given that a medical test offers 99% accuracy in detecting a desease and given that the desease appears in 1 in 100 individuals, if you score positive on the test, what is the probability of you actually having the disease?
90% accuracy means that 99% of the people tested and have the disease will test positive, while 1% of the people that don't have the disease will also test positive. We will also consider the opposite to be true: if the test gives a negative result, 99% probability you don't have the disease.

 - P(have the disease given if you scored positive on the test) = P(hypothesis | evidence)
 - P(you have the disease from the whole population) = P(hypothesis) = 1/100
 - P(scored positive on the test if you had the disease) = P(evidence | hypothesis) = 99/100
 - P(event) = P(scored positive on the test) = P(have the disease and scored positive) + P(don't have the disease and scored positive) = (99/100) * (1/100) + (1/100) * (99/100) 

```
P(H|E) = (P(E|H) * P(H)) / P(E)
P(H|E) = ((99/100) * (1/100)) / ((99/100) * (1/100) + (1/100) * (99/100) = 1/2 = 50%
```

2. A family has two children. The first born is a boy. What is the probability that the second one is a boy?

Obviously, two independent events, the answer is 1/2.

3. A family has two children. One of them is a boy. What is the probability that the family has two boys?

```
P(the family has two boys | one of them is a boy) = 
    P(if a family has two boys, the probabiliy that one of them is a boy) * P(a family has two boys) / 
    P(at least one of the children is a boy in a family with two children)

P(H|E) = (1 * 1/4) / (1 - P(both children are girls)) = (1/4) / (3/4) = 1/3
```

*Beta distribution*

In Bayesian inference, the beta distribution is the conjugate prior probability distribution for the Bernoulli, binomial, negative binomial and Geometric distribution [Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution).
This means that, given our initial knowledge of a system, as new events come, we can adjust our knowledge by simply updating the beta distribution parameters.

```python
def B(alpha, beta):
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)

def center(alpha, beta):
    return alpha / (alpha + beta)

def beta_pdf_array(alpha, beta, count):
    x_ = [x for x in range(0, count)]
    return x_, [beta_pdf(x / count, alpha, beta) for x in x_]
```

### Sports Betting Vocabulary


### Explanation of code and results