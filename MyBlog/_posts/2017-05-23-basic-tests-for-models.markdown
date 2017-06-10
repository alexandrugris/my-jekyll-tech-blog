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
Very high, but obviously the test is crap, as we will see with the following three metrics. The results are completely biased by the true negatives. An example of such a crap test with high accuracy could be 'you are CEO if your name is Jack'. Obviously, there are a lot of Jacks who are not CEOs.

### Precision

How accurate our positive predictions are.

```python
def precision(tp, fp, fn, tn):
    return tp / (tp + fp)


precision(10, 100, 1000, 10000)
Out[2]: 0.09090909090909091
```
Obviously, accuracy is less than 1%. 

### Recall

What fraction of positives the model identified.

```python
def recall(tp, fp, fn, tn):
    return tp  / (tp + fn)


recall (10, 100, 1000, 10000)
Out[3]: 0.009900990099009901
```

Again, not such a good score.

### F1 score

The harmonic average between the recall and precision scores. The harmonic average is a good choice because it gets a score closer to the lowest result.

```python
def f1_score(tp, fp, fn, tn):
    return 2 / (1/precision(tp, fp, fn, tn) + 1/recall(tp, fp, fn, tn))

f1_score(10, 100, 1000, 10000)
Out[5]: 0.017857142857142856
```

