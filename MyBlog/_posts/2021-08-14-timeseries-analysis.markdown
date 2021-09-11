---
layout: post
title:  "Timeseries Analysis in Python"
date:   2021-08-14 09:15:16 +0200
categories: programming
---

This post is about timeseries analysis in Python. It uses Google Collab for running Python notebooks.

### The Datasets

Before we dive into the code for analyzing timeseries, let's load the data. We load two  public datasets, one for analyzing non-periodic data, the other one for periodic. The first data set representing [lynx trappings in Canada between 1821 and 1934](https://rdrr.io/cran/fma/man/lynx.html), the second one, [the average monthly temperatures in Notthingham between 1920 and 1939](https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/nottem).

```python
import pandas as pd

lynx_df = pd.read_csv("/content/drive/MyDrive/Datasets/LYNXdata.csv", 
    index_col=0, 
    header=0, 
    names=['year', 'trappings'])

nottem_df = pd.read_csv("/content/drive/MyDrive/Datasets/nottem.csv", 
    index_col=0, 
    header=0, 
    names=['index', 'temp'])
```

And then transform it to time-annotated series:

```python
lynx_ts = pd.Series(
    lynx_df["trappings"].values, 
    pd.date_range('31/12/1821', 
    periods=len(lynx_df), 
    freq='A-DEC'))

nottem_df = pd.Series(
    nottem_df["temp"].values, 
    pd.date_range('31/01/1920', 
    periods=len(nottem_df), 
    freq='M'))

lynx_ts.plot()
nottem_df.plot()
```

![Lynx TS]({{site.url}}/assets/tsa_lynx.jpg)

![Nottem TS]({{site.url}}/assets/tsa_nottem.jpg)

### The Statistics Of Time Series

An excellent introduction on time series can be found [here](https://www.youtube.com/playlist?list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3).

Timeseries have two properties of major interest:

- Stationarity: constant statistical properties of the timeseries (mean, variance, no seasonality)
- Autocorrelation: correlation between subsequent values of a timeseries

Most of the timeseries have a trend, that is the mean is not constant. In this case, we need to remove the trend from the data. Relevant concepts are [trend-stationarity](https://en.wikipedia.org/wiki/Trend-stationary_process) and [difference-stationarity](https://en.wikipedia.org/wiki/Unit_root).

If `v(t_i+1) - v(t_i)` is random and stationary, then the process generating the series is a random walk, else a more refined model is required. 

The test that is mainly used for testing stationarity is called the [Augmented Dickey Fuller Test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test). It removes the autocorrelation and tests for equal mean and equal variance throughout the series. The null hypothesis is non-stationarity, that means that only if the p-value is less than 0.05 we can assume stationarity. Dickey Fuller test assumes that the time series is an AR1 process (auto-regressive one), that is, it can be written as `y(t) = phi * y(t-1) + constant + error`. The ADF test's null hypothesis is `phi >= 1`. The alternate hypothesis is that `phi < 1`. ADF extends the test ARn series and its null hypothesis is that `sum(phi_i)>=1`.

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

A good explanation of unit roots can be found [here](https://www.youtube.com/watch?v=ugOvehrTRRw).

*Other important notions concerning timeseries:*

- *ACF* - the autocorrelation between the value at `t` and `t-n`, *including* the intermediary values, `(t-1 .. t-n-1)`. E.g. the effect of prices 2 months ago vs the prices today, including the effect of the prices 2 months ago on the prices 1 month ago and the prices from 1 month ago on today's prices.
- *PACF* - the autocorrelation between the value at `t` and `t-n` *excluding* the intermediary values. 

ACF is easy to find, you just do a Pearson correlation between the values `ti` and the lagged `ti-k` values. For PACF we do a regression on the values of the timeseries at time `ti` on the `ti-k` of the form `ti=phi_1*ti_-1 + phi_2*ti_-2 + ... + phi_k*ti_-k + error_term` - autoregressive lag `k`. `phi_k` is our PACF(k).

To plot these on our data set:

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(lynx_ts, lags=100)
plot_pacf(lynx_ts, lags=10)
```

![ACF]({{site.url}}/assets/tsa_acf.jpg)

![PACF]({{site.url}}/assets/tsa_pacf.jpg)

We see in these charts that autocorrelation decreases as we look backwards in the series. This means that we are likely dealing with an auto-regressive process. If we are to build an auto-regressive model for this series we'll probably select coefficients 1, 2, 4 and 6.In the charts above, the blue bands are the error margin, everything within the bands are not statistically significant. The coefficient 0 is always 1, as it is the correlation of the timeseries with itself. 

Another concept is white noise: mean is 0, variance constant with time and no lag auto-correlation, no matter the lag value. White noise is not predictable. 

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
 - `ARIMA (p=1, d=0, q=1) <=> Y(t) = coef + phy_1 * Y(t-1) + theta_1 * error(t-1) + error(t)` - this time error is a regression too.
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


