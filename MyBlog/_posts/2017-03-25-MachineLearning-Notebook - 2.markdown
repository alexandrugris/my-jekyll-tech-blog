---
layout: post
title:  "My Machine Learning Notebook - Part 2 (Regression)"
date:   2017-03-11 14:15:16 +0200
categories: Machine Learning
---
This is the second part of my Machine Learning notebook, following the Udemy course "Machine Learning A-Z™: Hands-On Python & R".

### Simple Linear Regression With Plot

- Pink dots - training set X
- Blue line - regression line on the train set
- Red dots - values to predict (test set)
- Green dots - predicted values for the test set (situated on the blue line)

```python
import pandas as pd
dataset = pd.read_csv(".\\Data\\Salary_Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression().fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt

# scatter plot
plt.scatter(X_train, y_train, color="pink")
plt.scatter(X_test, y_test, color="red")
plt.scatter(X_test, y_pred, color="lightgreen")

# line plot
plt.plot(X_train, regressor.predict(X_train), color="blue")

plt.title("Salary vs Experience")
plt.xlabel("Years")
plt.ylabel("Salary")

plt.show()
```

![Results]({{site.url}}/assets/ml_3_1.png)