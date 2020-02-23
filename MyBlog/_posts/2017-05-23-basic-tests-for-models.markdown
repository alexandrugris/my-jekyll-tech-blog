---
layout: post
title:  "Classification"
date:   2017-05-23 13:15:16 +0200
categories: maths
---
Very short and simple post about metrics used for validating classification models.

### Confusion matrix

A matrix made of:

- *true positives:* my test says spam and the email is spam (tp)
- *false positives (type 1 error):* my test says spam but the email is not spam (fp)
- *false negatives (type 2 error):* my test says not spam but the email is spam (fn)
- *true negatives:* my test says not spam and the email is not spam (tn)

For the rest of the article, let's consider a classifier that has the following confusion matrix: `tp = 10`, `fp = 100`, `fn = 1000`, `tn = 10000`. This is an example of an imbalanced dataset (`tp + fn  = 10 + 1000 << fp + tn = 100 + 10000`).

### Accuracy

 The percent of classes classified correctly.
 
 ```python
 def accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)

accuracy(10, 100, 1000, 10000)
Out[1]: 0.900990099009901
```
Very high, but obviously the test is crap, as we will see with the following three metrics. The results are completely biased by the true negatives. An example of such a crap test with high accuracy could be 'you are CEO if your name is Jack'. Obviously, there are a lot of Jacks who are not CEOs.

### Precision

How accurate our positive predictions are. The percent of predicted 1s that are actually 1s.

```python
def precision(tp, fp, fn, tn):
    return tp / (tp + fp)


precision(10, 100, 1000, 10000)
Out[2]: 0.09090909090909091
```
Obviously, accuracy is less than 1%. 

### Recall or Sensitivity

The percentage of 1s that the model correctly classified.

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

### Specificity

The percentage of 0s correctly classified.

```python
def specificity(tp, fp, fn, tn):
    return tn / (tn + fn)
```

### ROC and AUROC (AUC)

The ROC curve captures the tradeoff between specificity and recall (sensitivity). Its axes are `X:(1-specificity)` and `Y: sensitivity`. It is drawn by varying the lambda classification threshold for a specific model. It is used to select such a threshold that gives the best ratio of sensitivity and specificity, which is given by choosing a lambda as close as possible to the top - left corner of the chart.

Below an example of plotting the ROC curve and setting the lambda parameter for a logistic regression.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X, y)

# our lambda parameter
threshold = 0.5 

# ...[0] is probability of not `0` 
y_pred = lr.predict_proba(X).T[0] < threshold

sensitivity = sum((y == 1) & (y_pred == 1)) / sum(y==1)
specificity = sum((y == 0) & (y_pred == 0)) / sum(y==0)

print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")

print(f"Predicted no, actual no: {sum((y==0) & (y_pred==0))}")
print(f"Predicted no, actual yes: {sum((y==1) & (y_pred==0))}")
print(f"Predicted yes, actual no: {sum((y==0) & (y_pred==1))}")
print(f"Predicted yes, actual yes: {sum((y==1) & (y_pred==1))}")
```

Now, let's vary the threshold and plot the ROC curve.

```
def roc(threshold):
    
    y_pred = lr.predict_proba(X).T[1] >= threshold
    sensitivity = sum((y == 1) & (y_pred == 1)) / sum(y==1)
    specificity = sum((y == 0) & (y_pred == 0)) / sum(y==0)
    
    return [sensitivity, 1 - specificity, threshold]

v = np.array([roc(th) for  th in np.arange(0, 1, step=0.01)]).T

# draw the ROC curve
plt.scatter(v[1], v[0])

# the point most distant from the curve
n = np.argmax(v[0] - v[1])
print(f"recommended threshold = {v[2][n]} with sensitivity = {v[0][n]} and specificity={1-v[1][n]}")
```

![ROC]({{site.url}}/assets/classification_1.png)

The AUC (area under the curve) or, in our case, better said AUROC (area under the ROC), is a measure of how good the classifier is. The larger the area, the better the classifier. In our case above, the classifier is pretty weak to start with.

In addition to the ROC curves, it can be very useful to examine the precision-recall curve [(PR-curve)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html). These are especially useful for imbalanced outcomes. Below is an example for the precision / recall curve:

![PR-curve]({{site.url}}/assets/classification_2.png)

In the example above, precision starts to fall sharply around 80% recall. In this case, we probably want to select a precision/recall tradeoff just before that drop for example, at around 60% recall. A good classifier has a PR-curve which gets closer to the top-right corner.

### The Imbalanced Dataset Problem

Imbalanced datasets can lead to a very poor classifier because the underrepresented class might simply be rejected as noise in the training step.

Some ways to deal with imbalanced data are described below:
- Undersample - use less of the prevalent class to train the model
- Oversample - use more of the rare class records to rain the model, usually achieved through bootstrapping
- Up/down weight - add weights to the classification data, so that the probability for each class becomes roughly the same
- Data generation - similar to bootstrapping, but each generated record is slightly different from its source. SMOTE is an algorithm that generates a new record for the record being up-sampled by using the K-nearest neighbors and assigning a randomly selected weight for each feature. 

A classification problem could be turned to a regression problem if an expected value is assigned to each classification record. For instance, instead of trying to predict a credit default, we might want to predict the expected return of a given loan.

