---
layout: post
title:  "Decision Trees"
date:   2019-12-01 13:15:16 +0200
categories: machine learning
---
A short introduction to decision trees (CART).

### What are decision trees

Decision trees are a supervised machine learning technique used for classification and regression problems. Since the launch of XGBoost, it has been the preferred way of tackling problems at machine learning competitions and not only, because of easy setup, learning speed, straightforward tuning parameters, easy to reason about results and, above all, very good results. Arguably, with current implementations, decision trees outperform many of the other machine learning techniques.

As with every other supervised learning techniques, the input to the algorithm in its learning phase is a set of predictors, `Xij`, and a set of predicted variables `Yj`, with `i` between `0` and `K`, the features, and `j` between `0` and `N`, the number of points in the training set. If the problem is one of classification the `Yj` labels, with `Yj` belonging to a finite set of labels, `L`, with `size(L) << N`.  If the problem is a regression, then `Yj` are  numeric values. The output is a decision tree which, for a new input vector, `X'` will predict its class (or the regressed value), `Y'`. If the problem is one of classification, the label predicted will be the label obtained by the majority vote from the points contained in one of its leafs. If the problem is one of regression, the value predicted will be the average values `Yj` of the points contained in the un-split set at one of its leafs. 

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

### Building the tree

Building the decision tree is a recursive process. At each step a feature is selected and a value based on which to do the split. The selection is made such that the highest information gain is attained, that is, the set is best split across its categories, meaning each group as as homogenous as possible. 


### Splitting

### TODO: analyze with decision trees the data from telco -> the one with telco from  - telco.csv in data folder