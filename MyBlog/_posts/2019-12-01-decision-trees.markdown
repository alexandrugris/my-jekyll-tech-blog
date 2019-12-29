---
layout: post
title:  "Decision Trees"
date:   2019-12-01 13:15:16 +0200
categories: statistics
---
A short introduction to decision trees.

### What are decision trees

Decision trees are a supervised machine learning technique used for classification and regression problems. Since the launch of XGBoost, it has been the preferred way of tackling problems at machine learning competitions and not only because of easy setup, learning speed, straightforward tuning parameters, easy to reason about results and, above all, very good results. Arguably, with current implementations, decision trees outperform most of the other machine learning techniques.

As with every other supervised learning techniques, the input to the algorithm in its learning phase is a set of predictors, `Xij`, and a set of predicted variables `Yj`, with `i` between `0` and `K`, the features, and `j` between `0` and `N`, the number of points in the training set. If the problem is one of classification the `Yj` labels, with `Yj` belonging to a finite set of labels, `L`, with `size(L) << N`.  If the problem is a regression, then `Yj` are  numeric values. The output is an decision tree which, for a new input vector, `X'` will predict its class (or the regressed value), `Y'`. If the problem is one of classification, the label predicted will be the label obtained by the majority vote from the points contained in one of its leafs. If the problem is one of regression, the value predicted will be the average values `Yj` of the points contained in the un-split set at one of its leafs. 

### Principles

A decision tree is, as the name implies, a tree. Since in the large majority of the cases this tree is binary, we will consider only the binary case in this post. 

When parsing the tree, at every node a decision is taken on whether to go right and or left. Each leaf represents the decision to which class that particular path belongs to. In the case of regression trees, it represents the predicted value for the specified set of parameters. 

The algorithm works by splitting the nodes based on the value of one of its predictor variables. The algorithm stops usually after some conditions, like the depth of the tree, or the number of leafs, or the number of nodes in a leaf has been been reached, or when all the elements in the leaf belong to a single class. Obviously, the decision trees are very prone either to overfitting by splitting too much, or to underfitting, by taking the decision to stop splitting too early.

These model parameters, which influence when to stop splitting, are tackled by hyperparameter tuning methods, which train trees with various parameters and then pick the most successful ones. 

The bias to given by the training set selection or feature choices are tackled by ensemble methods, like random forest or gradient boosted trees, which build several trees based on the input data to query from and then use voting to select the most accurate results.

### Building the tree

Building the decision tree is a recursive process. At each step a feature is selected and a value based on which to do the split. The selection is made such that the highest information gain is attained, that is, the set is best split across its categories, meaning each group as as homogenous as possible. 

### TODO: analyze with decision trees the data from telco -> the one with telco from  - telco.csv in data folder