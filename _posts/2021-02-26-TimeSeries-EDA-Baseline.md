---
title: "HackLive IV - Guided Community Hackathon - TimeSeries"
date: 2021-02-18
tags: [machine learning, data science, time series]
excerpt: "Machine Learning, Data Science, Time Series"
---

# Prophet Forecasts Stock Prices

[Link to competition here!](https://datahack.analyticsvidhya.com/contest/hacklive-4-guided-community-hackathon/)

Go there and register to be able to download the dataset and submit your predictions. Click the button below to open this notebook in Google Colab!

<a href="https://colab.research.google.com/github/jpraguer/jpraguer.github.io/master/_posts/TimeSeriesEDABaseline.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A stock market, equity market or share market is the aggregation of buyers and sellers of stocks (also called shares), which represent ownership claims on businesses; these may include securities listed on a public stock exchange, as well as stock that is only traded privately, such as shares of private companies which are sold to investors through equity crowdfunding platforms.

The secret of a successful stock trader is being able to look into the future of the stocks and make wise decisions. Accurate prediction of stock market returns is a very challenging task due to volatile and non-linear nature of the financial stock markets. With the introduction of artificial intelligence and increased computational capabilities, programmed methods of prediction have proved to be more efficient in predicting stock prices.

 Here, you are provided dataset of a public stock market for 104 stocks. Can you forecast the future closing prices for these stocks with your Data Science skills for the next 2 months?


```python
# import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

from fbprophet import Prophet
import multiprocessing
from joblib import Parallel, delayed
```


```python
# load data and set seed
BASE = 'https://drive.google.com/uc?export=download&id='
SEED = 2021

train = pd.read_csv(f'{BASE}1H3EhyeZ5YJi6OHjMtvWO1TpPcS-C-SYG', parse_dates=['Date']) # parse Date column right away
test = pd.read_csv(f'{BASE}1GRJpiV_fLkhE3MUPxpDcdQ-Xz-4OFKcN', parse_dates=['Date'])
ss = pd.read_csv(f'{BASE}1kOhLBDZyeONgF1NmVnc2Q2bq847VGm_V')
```

## EDA

First we look at the first few rows of the train and test dataset. Also double check that dates were parsed correctly.


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>stock</th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>holiday</th>
      <th>unpredictability_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0</td>
      <td>0</td>
      <td>2017-01-03</td>
      <td>82.9961</td>
      <td>82.7396</td>
      <td>82.9144</td>
      <td>82.8101</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_1</td>
      <td>0</td>
      <td>2017-01-04</td>
      <td>83.1312</td>
      <td>83.1669</td>
      <td>83.3779</td>
      <td>82.9690</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_2</td>
      <td>0</td>
      <td>2017-01-05</td>
      <td>82.6622</td>
      <td>82.7634</td>
      <td>82.8984</td>
      <td>82.8578</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_3</td>
      <td>0</td>
      <td>2017-01-06</td>
      <td>83.0279</td>
      <td>82.7950</td>
      <td>82.8425</td>
      <td>82.7385</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_4</td>
      <td>0</td>
      <td>2017-01-09</td>
      <td>82.3761</td>
      <td>82.0828</td>
      <td>82.1473</td>
      <td>81.8641</td>
      <td>0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>stock</th>
      <th>Date</th>
      <th>holiday</th>
      <th>unpredictability_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_713</td>
      <td>0</td>
      <td>2019-11-01</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_714</td>
      <td>0</td>
      <td>2019-11-04</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_715</td>
      <td>0</td>
      <td>2019-11-05</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_716</td>
      <td>0</td>
      <td>2019-11-06</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_717</td>
      <td>0</td>
      <td>2019-11-07</td>
      <td>0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 73439 entries, 0 to 73438
    Data columns (total 9 columns):
     #   Column                  Non-Null Count  Dtype         
    ---  ------                  --------------  -----         
     0   ID                      73439 non-null  object        
     1   stock                   73439 non-null  int64         
     2   Date                    73439 non-null  datetime64[ns]
     3   Open                    73439 non-null  float64       
     4   High                    73439 non-null  float64       
     5   Low                     73439 non-null  float64       
     6   Close                   73439 non-null  float64       
     7   holiday                 73439 non-null  int64         
     8   unpredictability_score  73439 non-null  int64         
    dtypes: datetime64[ns](1), float64(4), int64(3), object(1)
    memory usage: 5.0+ MB
    


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4223 entries, 0 to 4222
    Data columns (total 5 columns):
     #   Column                  Non-Null Count  Dtype         
    ---  ------                  --------------  -----         
     0   ID                      4223 non-null   object        
     1   stock                   4223 non-null   int64         
     2   Date                    4223 non-null   datetime64[ns]
     3   holiday                 4223 non-null   int64         
     4   unpredictability_score  4223 non-null   int64         
    dtypes: datetime64[ns](1), int64(3), object(1)
    memory usage: 165.1+ KB
    


```python
# define ID and target column names

ID_COL, TARGET_COL = 'ID', 'Close'



# define predictors

features = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
```


```python
# look at train and test sizes

train.shape, test.shape
```




    ((73439, 9), (4223, 5))




```python
# plot target distribution

sns.distplot(train[TARGET_COL]);
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_10_1.png)
    



```python
# plot target distribution for a single stock

STOCK_NO = 0

sns.distplot(train.loc[train['stock'] == STOCK_NO, TARGET_COL]).set_title(f'Stock {STOCK_NO}');
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_11_1.png)
    



```python
# plot target distribution for a different stock

STOCK_NO = 42

sns.distplot(train.loc[train['stock'] == STOCK_NO, TARGET_COL]).set_title(f'Stock {STOCK_NO}');
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_12_1.png)
    



```python
# unique values in each variable

train.nunique()
```




    ID                        73439
    stock                       103
    Date                        713
    Open                      60702
    High                      60594
    Low                       61015
    Close                     60352
    holiday                       2
    unpredictability_score       10
    dtype: int64




```python
test.nunique()
```




    ID                        4223
    stock                      103
    Date                        41
    holiday                      2
    unpredictability_score      10
    dtype: int64




```python
# explore holiday variable

print(train['holiday'].value_counts())

sns.boxplot(x = train['holiday'], y = train[TARGET_COL]);
```

    0    69216
    1     4223
    Name: holiday, dtype: int64
    


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_15_1.png)
    



```python
# explore unpredictability_score

print(train['unpredictability_score'].value_counts())

sns.boxplot(x = train['unpredictability_score'], y = train[TARGET_COL]);
```

    9    7843
    4    7843
    0    7843
    8    7130
    7    7130
    6    7130
    5    7130
    3    7130
    2    7130
    1    7130
    Name: unpredictability_score, dtype: int64
    


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_16_1.png)
    



```python
# date ranges

print(train['Date'].min(), train['Date'].max())

print(test['Date'].min(), test['Date'].max())
```

    2017-01-03 00:00:00 2019-10-31 00:00:00
    2019-11-01 00:00:00 2019-12-31 00:00:00
    


```python
# plot holidays in train

train.set_index('Date')[['holiday']].plot(figsize=(16, 4));
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_18_0.png)
    



```python
# plot Close

train.set_index('Date')[[TARGET_COL]].plot(figsize=(16, 4));
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_19_0.png)
    



```python
# plot Close for a single stock

STOCK_NO = 0

train.loc[train['stock'] == STOCK_NO].set_index('Date')[[TARGET_COL]].plot(figsize=(16, 4), title = f'Stock {STOCK_NO}');
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_20_0.png)
    



```python
# plot Close for a different stock

STOCK_NO = 42

train.loc[train['stock'] == STOCK_NO].set_index('Date')[[TARGET_COL]].plot(figsize=(16, 4), title = f'Stock {STOCK_NO}');
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_21_0.png)
    


#### Observations

No null values in these datasets. Training dates are from January 2017 to Halloween 2019, test dates are November and December 2019.

We should predict future stock price for 104 different stocks based only on `Date`, `holiday` flag (1/0) and an `unpredictability_score` (1-9).

Closing amount does not seem to differ too much on `holiday`s compared to normal days. Volatility definitely rises with increasing `unpredictability_score`, as expected. Counts in each score are quite balanced, with 9, 4, and 0 being slightly more common - worth more exploration in the future.

## Baseline FB Prophet model!

Let's define a parallelised Prophet prediction pipeline to speed up the prediction for 104 stocks separately.

Credit for this idea goes to [this kaggle kernel](https://www.kaggle.com/raghvenbhati/prophet-forecasts).


```python
# Forecast function:
def ProphetFC(stock_no: int):
    '''
    Predict test prices for each stock separately.

    :param stock_no: Stock ID to predict
    
    :returns: Forecasted future dataframe, as return by prophet's .predict method
    '''
    # Create Prophet model
    m = Prophet()
    
    # Create df, add features and fit
    tsdf = pd.DataFrame({
      'ds': train.loc[train['stock'] == stock_no, 'Date'].reset_index(drop=True),
      'y': train.loc[train['stock'] == stock_no, TARGET_COL].reset_index(drop=True),
    })
    tsdf['holiday'] = train.loc[train['stock'] == stock_no, 'holiday'].reset_index(drop=True)
    tsdf['unpredictability_score'] = train.loc[train['stock'] == stock_no, 'unpredictability_score'].reset_index(drop=True)
    
    m.add_regressor('holiday')
    m.add_regressor('unpredictability_score')

    m.fit(tsdf)
    
    # create future df and predict
    future = pd.DataFrame({
      'ds': test.loc[test['stock'] == stock_no, 'Date'].reset_index(drop=True),
    })

    future['holiday'] = test.loc[test['stock'] == stock_no, 'holiday'].reset_index(drop=True)
    future['unpredictability_score'] = test.loc[test['stock'] == stock_no, 'unpredictability_score'].reset_index(drop=True)

    fcst = m.predict(future)

    fcst[ID_COL] = test.loc[test['stock'] == stock_no, ID_COL]
    
    fcst['stock'] = stock_no
    return fcst
```


```python
%%time
# how many stocks to predict for
NUM_STOCKS = train['stock'].nunique()

# parallel jobs to forecast
num_cores = multiprocessing.cpu_count()
processed_FC = Parallel(n_jobs=num_cores)(delayed(ProphetFC)(i) for i in range(NUM_STOCKS))
```

    CPU times: user 3 s, sys: 160 ms, total: 3.16 s
    Wall time: 5min 15s
    


```python
# combining obtained dataframes
FCAST = pd.concat(processed_FC, ignore_index=True)
FCAST.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>additive_terms</th>
      <th>additive_terms_lower</th>
      <th>additive_terms_upper</th>
      <th>extra_regressors_additive</th>
      <th>extra_regressors_additive_lower</th>
      <th>extra_regressors_additive_upper</th>
      <th>holiday</th>
      <th>holiday_lower</th>
      <th>holiday_upper</th>
      <th>unpredictability_score</th>
      <th>unpredictability_score_lower</th>
      <th>unpredictability_score_upper</th>
      <th>weekly</th>
      <th>weekly_lower</th>
      <th>weekly_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
      <th>ID</th>
      <th>stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-11-01</td>
      <td>123.836651</td>
      <td>117.992600</td>
      <td>122.187867</td>
      <td>123.836651</td>
      <td>123.836651</td>
      <td>-3.858359</td>
      <td>-3.858359</td>
      <td>-3.858359</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>0.098074</td>
      <td>0.098074</td>
      <td>0.098074</td>
      <td>-3.873307</td>
      <td>-3.873307</td>
      <td>-3.873307</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>119.978292</td>
      <td>id_713</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-11-04</td>
      <td>124.158461</td>
      <td>118.541425</td>
      <td>122.710671</td>
      <td>124.158461</td>
      <td>124.158461</td>
      <td>-3.492853</td>
      <td>-3.492853</td>
      <td>-3.492853</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.073551</td>
      <td>-0.073551</td>
      <td>-0.073551</td>
      <td>-3.336177</td>
      <td>-3.336177</td>
      <td>-3.336177</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120.665608</td>
      <td>id_714</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-11-05</td>
      <td>124.265731</td>
      <td>118.865930</td>
      <td>123.259751</td>
      <td>124.265731</td>
      <td>124.265731</td>
      <td>-3.231049</td>
      <td>-3.231049</td>
      <td>-3.231049</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.007098</td>
      <td>-0.007098</td>
      <td>-0.007098</td>
      <td>-3.140825</td>
      <td>-3.140825</td>
      <td>-3.140825</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>121.034682</td>
      <td>id_715</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-11-06</td>
      <td>124.373002</td>
      <td>119.118231</td>
      <td>123.431604</td>
      <td>124.373002</td>
      <td>124.373002</td>
      <td>-3.067293</td>
      <td>-3.067293</td>
      <td>-3.067293</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.045587</td>
      <td>-0.045587</td>
      <td>-0.045587</td>
      <td>-2.938580</td>
      <td>-2.938580</td>
      <td>-2.938580</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>121.305709</td>
      <td>id_716</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-11-07</td>
      <td>124.480272</td>
      <td>119.611473</td>
      <td>123.998805</td>
      <td>124.480272</td>
      <td>124.480272</td>
      <td>-2.787002</td>
      <td>-2.787002</td>
      <td>-2.787002</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>-0.083126</td>
      <td>0.026466</td>
      <td>0.026466</td>
      <td>0.026466</td>
      <td>-2.730342</td>
      <td>-2.730342</td>
      <td>-2.730342</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>121.693270</td>
      <td>id_717</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# define function to plot predictions

def plot_preds(stock_no: int):

  '''

  PLot Closes for a certain stock separately.



  :param stock_no: Stock ID to plot

    

  :returns: nothing

  '''

  # create temp train df

  train_tmp = train.loc[train['stock'] == stock_no].set_index('Date')[[TARGET_COL]]

  train_tmp['type'] = 'train'



  # create temp test df

  test_tmp = FCAST.loc[FCAST['stock'] == stock_no].rename(columns={"yhat": TARGET_COL, 'ds': 'Date'}).set_index('Date')[[TARGET_COL]]

  test_tmp['type'] = 'test'

  train_tmp.append(test_tmp).groupby('type')[TARGET_COL].plot(figsize=(16, 4), title = f'Stock {stock_no}', sharex=False, legend=True);

  pass
```


```python
# plot stock 0 with preds
plot_preds(0)
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_28_0.png)
    



```python
# plot stock 42 with preds
plot_preds(42)
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_29_0.png)
    



```python
# plot stock 100 with preds
plot_preds(100)
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/TimeSeriesEDABaseline_files/TimeSeriesEDABaseline_30_0.png)
    


#### Observations

Clearly some of these predictions do not start where the train set ends, so further iterations are needed (perhaps targeted on holidays, seasonality, and its fourier order) to fix this problem.


```python
# submission
submission = FCAST[['ID', 'yhat']].rename(columns={'yhat': TARGET_COL})
submission.to_csv('submission_prophet_baseline.csv',index=False)
```


```python
# and we're done!

'Done!'
```




    'Done!'


