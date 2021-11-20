---
layout: post
title:  "Timeseries Analysis in Python"
date:   2021-08-14 09:15:16 +0200
categories: programming
---

This post is about statistical models for timeseries analysis in Python. We will cover the ARIMA model to a certain depth.

### Linear Regression and Timeseries


Using a linear regression with time series is problematic. Linear regression  assumes you have independently and identically distributed data. In time series data, points near in time tend to be strongly correlated with one another. In fact, when there aren’t temporal correlations, time series data is hardly useful for traditional time series tasks,
such as predicting the future or understanding temporal dynamics. Linear regression can be used with timeseries when linear regression assumptions hold. Such a case is when the predicted variable is fully dependent on its predictors, for instance when the timeseries component is entirely embedded in one of the features and the errors preserve the normality assumption with no autocorrelation.


### The Statistics Of Time Series

An excellent introduction on time series can be found [here](https://www.youtube.com/playlist?list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3).

Timeseries bring several concepts of interest

- Stationarity: constant statistical properties of the timeseries (mean, variance, no seasonality)
- Self-correlation: correlation between subsequent values of a timeseries
- Seasonality: time-based patterns tha repeat at set intervals
- Spurious correlations: the propensity of timeseries to correlate with other unrelated timeseries especially when seasonality or trends are present.

A log transformation or a square root transformation are two usually good options for making a timeseries stationary,particularly in the case of changing variance over time. 

Most of the timeseries have a trend, that is the mean is not constant - [trend-stationarity](https://en.wikipedia.org/wiki/Trend-stationary_process) and [difference-stationarity](https://en.wikipedia.org/wiki/Unit_root). Removing a trend is most commonly done by differencing. Sometimes a series must be differenced more than once. However, if you find yourself differencing too much (more than two or three times) it is unlikely that you can fix your stationarity problem with differencing.

If `v(t_i+1) - v(t_i)` is random and stationary, then the process generating the series is a random walk, else a more refined model is required. 

The test that is mainly used for testing stationarity is called the [Augmented Dickey Fuller Test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test). It removes the autocorrelation and tests for equal mean and equal variance throughout the series. The null hypothesis is the non-stationarity. 

The Dickey Fuller test assumes that the time series is an AR1 process (auto-regressive one), that is, it can be written as `y(t) = phi * y(t-1) + constant + error`. The DF test's null hypothesis is `phi >= 1`. The alternate hypothesis is that `phi < 1`.  This `phi == 1` is called a unit root. A good explanation of unit roots can be found [here](https://www.youtube.com/watch?v=ugOvehrTRRw).

ADF extends the test to ARn series and this null hypothesis is that `sum(phi_i)>=1`. The difference between the basic DF test and the ADF test is that the latter makes is to account for more lags. The test of whether a series is stationary is a test of whether a series is integrated. An integrated series of order `n` is a series that must be differenced `n` times to become stationary.

```python
import random
from statsmodels.tsa.stattools import adfuller

def gen_ts(start, coefs, count, error):
  assert(len(start) == len(coefs))
  assert(count > len(start))

  lst = start + [0] * (count - len(start))

  for i in range(len(start), count):
    lst[i] = random.uniform(-0.5, 0.5) * error
    for j in range (1, len(start)+1):
      lst[i] += coefs[j-1] * lst[i-j]

  return lst

v = gen_ts([0, 1, 2], [0.5, 0.2, 0.1], 100, 1.0) # sum of coefficients < 1
plt.plot(v)

# automatically test for the best lag to use
# AIC comes from https://en.wikipedia.org/wiki/Akaike_information_criterion
p_value = adfuller(v, autolag='AIC')[1]
print(p_value)
```

![ADF]({{site.url}}/assets/tsa_adf.jpg)

For detecting auto correlation, we introduce two measures:
- The *ACF* - the autocorrelation between the value at `t` and `t-n`, *including* the intermediary values, `(t-1 .. t-n-1)`. E.g. the effect of prices 2 months ago vs the prices today, including the effect of the prices 2 months ago on the prices 1 month ago and the prices from 1 month ago on today's prices.
- The *PACF* - the autocorrelation between the value at `t` and `t-n` *excluding* the intermediary values. 

The ACF is the Pearson correlation between the values `ti` and the lagged `ti-k` values. For the PACF we do a regression on the values of the timeseries at time `ti` on the `ti-k` of the form `ti=phi_1*ti_-1 + phi_2*ti_-2 + ... + phi_k*ti_-k + error_term` - autoregressive lag `k`. `phi_k` is our PACF(k).

To plot these a real dataset:

```python
import pandas as pd

lynx_df = pd.read_csv("/content/drive/MyDrive/Datasets/LYNXdata.csv", 
    index_col=0, 
    header=0, 
    names=['year', 'trappings'])
```

And then transform it to time-annotated series:

```python
lynx_ts = pd.Series(
    lynx_df["trappings"].values, 
    pd.date_range('31/12/1821', 
    periods=len(lynx_df), 
    freq='A-DEC'))

lynx_ts.plot()
```

![Lynx TS]({{site.url}}/assets/tsa_lynx.jpg)

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(lynx_ts, lags=100)
plot_pacf(lynx_ts, lags=10)
```

![ACF]({{site.url}}/assets/tsa_acf.jpg)

![PACF]({{site.url}}/assets/tsa_pacf.jpg)

We see in these charts that autocorrelation decreases as we look backwards in the series. This means that we are likely dealing with an auto-regressive process. If we are to build an auto-regressive model for this series we'll probably consider the coefficients for the 1, 2, 4 and 6 lags. The blue bands are the error margin; everything within the bands are not statistically significant. The coefficient with index 0 is always 1, as it is the correlation of the timeseries with itself. 

A series can be stationary, that is the mean is 0 and the variance constant with time, but with no lag auto-correlation, no matter the lag value. Such a series is called white noise and it is not predictable. 

Let's plot some generated time series to explore the PCF and PACF charts for various cases.



Since timeseries tend to be noisy, we introduce the concept of moving average. When selecting the smoothing period, one needs to pay attention to aliasing effects that occur when the length of interval is larger than half of the period. 

```python
plt.plot(pd.Series(v).rolling(10).mean()[9:])
```

Another type of window is the Exponentially Weighted Window, (`ewm`) which is similar to an expanding window but with each prior point being exponentially weighted down relative to the current point. Pandas offers also different types of windows, such as triangular, Gaussian or custom. Below an example of a Gaussian smoothing and one Exponentially Weighted.

```python
# exponentially weighted - important to notice 
# that the window is extending, so first few values 
# have more variability
# alpha is a parameter of how much smoothing (0 : 1),
# 1 being no smoothing at all
plt.plot(lynx_ts.ewm(alpha=0.1).mean())
```

```python
# center=True must be specified, otherwise the smoothing shifts the series right
plt.plot(lynx_ts.rolling(window=10, win_type="gaussian", center=True).mean(std=1))
```
![Original Series - EWM - Gaussian]({{site.url}}/assets/tsa_smoothing.jpg)

# ARIMA Models

An ARIMA model has 3 parameters:

- `p` : the order of autoregression (the summation of the weighted lags) - `AR`
- `d` : the degree of differencing (used to make the dataset stationary if it is not) - `I`
- `q` : the order of moving average (the summation of the lags of the forecast errors) - `MA`

Examples (`ARIMA(p, d, q)`):

 - `ARIMA (p=1, d=0, q=0) <=> Y(t) = coef + phi_1 * Y(t-1) + error(t)` - lag 1 autoregressive model (1 is the lag)
 - `ARIMA (p=1, d=0, q=1) <=> Y(t) = coef + phi_1 * Y(t-1) + theta_1 * error(t-1) + error(t)` - this time error is a regression too.
 - `ARIMA (p=0, d=1, q=0) <=> Y(t) = coef + Y(t-1) + error(t)` is a random walk. The differencing equation, `Y(t) - Y(t-1) = coef + error(t)`, is needed so that the remaining `ARMA` model is applied on stationary data. A random walk is not stationary.

 # ARIMA Model Parameter Selection

 First step is to check for stationarity using the Augmented Dickey-Fuller test. If the data is not stationary, we need to set the `d` parameter.

 The second step is to set the `p` and `q` parameters by inspecting the `ACF` and `PACF` plots. The `ACF` plot tells more about the Moving Average parameter `q` while pe `PACF` plot tells more about the Auto-Regressive parameter `p`.

To avoid over-fitting, a rule of thumb is to start the parameter selection with the plot that has the least amount of lags outside of the significance bands and then consider the lowest reasonable amount of lags.

On the Lynx dataset, this looks as the following:

![ACF and PACF Lynx On The Dataset]({{site.url}}/assets/tsa_lynx_pacf.JPG)

Because the PACF chart is mostly inside the benchmark with lags 1 and 2 the most prominent, we start with it. Our first model we aim to fit is an auto regressive (AR) model with lag 2. 

```python
from statsmodels.tsa.arima_model import ARIMA

m = ARIMA(lynx_ts, order=(2,0,0))
results = m.fit()
plt.plot(lynx_ts)
plt.plot(results.fittedvalues, color="orange")
```

Even with this simple model, we have a pretty good fit.

![The first ARIMA fitted model]({{site.url}}/assets/tsa_arima_lynx.png)

Let's check the residuals now. We are searching for:
- 0 mean
- Normal distribution
- See if they have autocorrelation

```python
resid = lynx_ts - results.fittedvalues
plt.hist(resid)

import scipy.stats as stats
# visual inspection of the residuals
plt.hist(resid)
# Shapiro test for normality
stats.shapiro(stats.zscore(resid))[1]
```

![Residuals ACF and Normality Checks]({{site.url}}/assets/tsa_resid.png)

The plots above show little to no autocorrelation, but it is hard to accept that the residuals are normally distributed. 
