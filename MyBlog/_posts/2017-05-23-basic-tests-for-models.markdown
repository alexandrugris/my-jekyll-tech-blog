---
layout: post
title:  "Basic Tests For ML Models"
date:   2017-05-23 13:15:16 +0200
categories: maths
---
Very short and simple post about metrics used for validating models.

### Confusion matrix

A matrix made of:

- *true positives:* my test says spam and the email is spam (tp)
- *false positives (type 1 error):* my test says spam but the email is not spam (fp)
- *false negatives (type 2 error):* my test says not spam but the email is spam (fn)
- *true negatives:* my test says not spam and the email is not spam (tn)

For the rest of the article let's consider some numbers: tp = 10, fp = 100, fn = 1000, tn = 10000

### Accuracy

 Fraction of correct predictions:
 
 ```python
 def accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)

accuracy(10, 100, 1000, 10000)
Out[1]: 0.900990099009901
```
Very high, but obviously the test is crap, as we will see with the following three metrics.

### Precision

### Recall

### F1 score


