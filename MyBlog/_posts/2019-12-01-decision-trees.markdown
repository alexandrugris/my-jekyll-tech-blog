---
layout: post
title:  "Decision Trees"
date:   2019-12-01 13:15:16 +0200
categories: machine learning
---
A short introduction to decision trees (CART). Decision trees are a supervised machine learning technique used for classification and regression problems. Since the launch of XGBoost, it has been the preferred way of tackling problems at machine learning competitions and not only, because of easy setup, learning speed, straightforward tuning parameters, easy to reason about results and, above all, very good results. Arguably, with current implementations, decision trees outperform many of the other machine learning techniques.

### Classification and Prediction

As with every other supervised learning techniques, the input to the algorithm in its learning phase is a set of predictors, `Xij`, and a set of predicted variables `Yj`, with `i` between `0` and `K`, the features, and `j` between `0` and `N`, the number of points in the training set. 

If the problem is one of classification the `Yj` labels, with `Yj` belonging to a finite set of labels, `L`, with `size(L) << N`.  If the problem is a regression, then `Yj` are  numeric values. 

The output is a decision tree which, for a new input vector, `X'` will predict its class (or the regressed value), `Y'`. If the problem is one of classification, the label predicted will be the label obtained by the majority vote from the points contained in one of its leafs. If the problem is one of regression, the value predicted will be the average values `Yj` of the points contained in the un-split set at one of its leafs. 

### Principles

A decision tree is, as the name implies, a tree. Since in the large majority of the cases this tree is binary, we will consider only the binary case in this post. 

When parsing the tree, at every node a decision is taken on whether to go right and or left. Each leaf represents the decision to which class that particular path belongs to. In the case of regression trees, it represents the predicted value for the specified set of parameters. 

The algorithm works by splitting the nodes based on the value of one of its predictor variables. The algorithm stops usually after some conditions, like the depth of the tree, or the number of leafs, or the number of nodes in a leaf has been been reached, or when all the elements in the leaf belong to a single class. Obviously, the decision trees are very prone either to overfitting by splitting too much, or to underfitting, by taking the decision to stop splitting too early.

These model parameters, which influence when to stop splitting, are tackled by hyperparameter tuning methods, which train trees with various parameters and then pick the most successful ones. 

The bias to given by the training set selection (bagging) or feature choices are tackled by ensemble methods, like random forest or gradient boosted trees, which build several trees based on the input data to query from and then use voting to select the most accurate results.

Below is a very basic example of how to build such a decision tree, together with its visualization:

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=60, min_samples_split=60)
dt.fit(X, y)

plt.figure(figsize=(20, 20))
plot_tree(dt)
plt.show()

```
![Decision Tree Visualization]({{site.url}}/assets/decision_trees_1.png)

The tree above is a regression tree, inputted with numeric features.

- `X[n]` - the feature based on which the split has been made
- `mse` - mean squared error in the node 
- `samples` - number of samples contained in the node before splitting the node further
- `value` - the mean value predicted for the node

In our case, we can see the Sum of Squared Error (SSE) is improved with every split. Taking the first node, for instance, we observe:

- First step: `MSE = 455 => SSE_before_the_split = 455 * 1000`
- After split we have: `SSE_after_the_split = 199 * 565 + 202 * 435 = 200305 < SSE_before_the_split`

With `SSE = sum((corresponding_yi_from_partition_0 - y_mean_0)^2  ) + sum(corresponding_yi_from_partition_1 - y_mean_1)^2) ` where _0 and _1 are the two partitions.

### A little bit of terminology

Here is a little bit of terminology associated with the decision trees:

- Classification tree: a tree which outputs which class the input vector belongs to. It does so by majority selection from the selected leaf.

- Regression tree: a tree which outputs the predicted value for an input vector. It does so by averaging the values from the selected leaf.

- Random forest: a way of improving the predicted values for CARTs by training a number of independent trees and averaging or voting from their outputs. This can be achieved by selecting a different set of features, sampling from the training data or training with the model different hyperparameters.

- Bagging: a way of building a tree from the forest by randomly sampling with replacement from the training data.

- Gradient boosting: a way of building subsequent trees by weighting down the points for which the tree classified correctly and weighting up the misclassified points. In the end result, trees which performed poorly in classification will have a lower voting weight than trees that performed well.

- Post pruning: the opposite of splitting, cutting the nodes that don't bring enough information, to avoid overfitting. 

### Building the tree

Building the decision tree is a recursive process. At each step a feature is selected and a value from the selected feature based on which to do the split. The selection is made such that the highest information gain is attained, meaning each group as as homogenous as possible. 

Nodes are split based on their “impurity”. Impurity is a measure of how badly the observations at a given node fit the model.

In a regression tree, for example, the impurity may be measured by the residual sum of squares within that node - as we've seen in the example above. 

In a classification tree, there are various ways of measuring the impurity, such as the misclassification error, the Gini index, and the entropy.

For building the tree there are 3 general cases, each building on each other:

- All independent (`X`) and dependent (`y`) variables are categorical.
- There is a mix of categorical and continuous independent variables (`X`) but the dependent variable is categorical (`y`).
- The dependent variable is continuous (`y`).

### All independent and dependent variables are categorical

The algorithm goes as follows:

1. Compute the uncertainty at the root node
2. Find the feature that has the highest uncertainty 
3. Find the splitting point
4. Repeat

We are going to analyze the following [dataset](({{site.url}}/assets/telco.csv)), where the predicted variable is `churn`. 

```python
import pandas as pd

df = pd.read_csv('../data/telco.csv', sep='\t')

df_categorical = df[['marital', 'ed', 'gender', 'retire']]
y = df['churn']
```

Entropy (uncertainty) at the root:

```python
import numpy as np

def entropy(s):
    x = s.value_counts().values / s.count()
    return np.sum(-x * np.log2(x))

y_entropy = entropy(y)
```

Calculating the entropy reduction if we split by each feature:

```python
for c in df_categorical.columns:

    ct = pd.crosstab(df_categorical[c], y, normalize='index')
    
    # compute entropy for each of classes 
    entp = pd.DataFrame(np.sum(-np.log2(ct.values) * ct.values, axis=1))
    entp.index = ct.index
    entp = entp.T[sorted(entp.T.columns)].values.flatten()
    
    # compute weights for each of the classes
    w = pd.DataFrame(df_categorical[c].value_counts() / df_categorical[c].count()).T
    w = w[sorted(w.columns)].values.flatten()
    
    # normalize for size of the population to get the weighted entropy
    weighted_entropy = np.dot(entp, w)
    
    print(f'Entropy for {c} is {weighted_entropy}, reduction in entropy is {y_entropy - weighted_entropy}')
```

Given the results below, we would choose feature `ed` for split as it gives the highest entropy decrease.

![Feature selection]({{site.url}}/assets/decision_trees_2.png)

Broken down into individual steps, for the variable `ed` the intermediate results from each step are shown in the image below:

![Entropy gain]({{site.url}}/assets/decision_trees_3.png)

        
    






