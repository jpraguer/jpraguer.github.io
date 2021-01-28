---
title: "Janatahack: Cross-sell Prediction"
date: 2021-01-28
tags: [machine learning, data science, classification]
excerpt: "Machine Learning, Data Science, Classification"
---


[Link to competition here!](https://datahack.analyticsvidhya.com/contest/janatahack-cross-sell-prediction/)

Go there and register to be able to download the dataset and submit your predictions.Click the button below to open this notebook in Google Colab!

{::nomarkdown}
<a href="https://colab.research.google.com/github/jpraguer/jpraguer.github.io/blob/master/assets/ipynbs/ClassificationEDABaseline.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
{:/}

Cross-selling identifies products or services that satisfy additional, complementary needs that are unfulfilled by the original product that a customer possesses. As an example, a mouse could be cross-sold to a customer purchasing a keyboard. Oftentimes, cross-selling points users to products they would have purchased anyways; by showing them at the right time, a store ensures they make the sale.



Cross-selling is prevalent in various domains and industries including banks. For example, credit cards are cross-sold to people registering a savings account. In ecommerce, cross-selling is often utilized on product pages, during the checkout process, and in lifecycle campaigns. It is a highly-effective tactic for generating repeat purchases, demonstrating the breadth of a catalog to customers. Cross-selling can alert users to products they didn't previously know you offered, further earning their confidence as the best retailer to satisfy a particular need.

Your client is an Insurance company that has provided Health Insurance to its customers now they need your help in building a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.

An insurance policy is an arrangement by which a company undertakes to provide a guarantee of compensation for specified loss, damage, illness, or death in return for the payment of a specified premium. A premium is a sum of money that the customer needs to pay regularly to an insurance company for this guarantee.

For example, you may pay a premium of Rs. 5000 each year for a health insurance cover of Rs. 200,000/- so that if, God forbid, you fall ill and need to be hospitalised in that year, the insurance provider company will bear the cost of hospitalisation etc. for upto Rs. 200,000. Now if you are wondering how can company bear such high hospitalisation cost when it charges a premium of only Rs. 5000/-, that is where the concept of probabilities comes in picture. For example, like you, there may be 100 customers who would be paying a premium of Rs. 5000 every year, but only a few of them (say 2-3) would get hospitalised that year and not everyone. This way everyone shares the risk of everyone else.

Just like medical insurance, there is vehicle insurance where every year customer needs to pay a premium of certain amount to insurance provider company so that in case of unfortunate accident by the vehicle, the insurance provider company will provide a compensation (called ‘sum assured’) to the customer.

Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue. 

Now, in order to predict, whether the customer would be interested in Vehicle insurance, you have information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc.


```python
# install catboost

!pip install catboost
```

    Collecting catboost
    [?25l  Downloading https://files.pythonhosted.org/packages/20/37/bc4e0ddc30c07a96482abf1de7ed1ca54e59bba2026a33bca6d2ef286e5b/catboost-0.24.4-cp36-none-manylinux1_x86_64.whl (65.7MB)
    [K     |████████████████████████████████| 65.8MB 49kB/s 
    [?25hRequirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from catboost) (1.15.0)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.6/dist-packages (from catboost) (0.10.1)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from catboost) (3.2.2)
    Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.6/dist-packages (from catboost) (1.19.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from catboost) (1.4.1)
    Requirement already satisfied: plotly in /usr/local/lib/python3.6/dist-packages (from catboost) (4.4.1)
    Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.6/dist-packages (from catboost) (1.1.5)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->catboost) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->catboost) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->catboost) (2.4.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->catboost) (1.3.1)
    Requirement already satisfied: retrying>=1.3.3 in /usr/local/lib/python3.6/dist-packages (from plotly->catboost) (1.3.3)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.24.0->catboost) (2018.9)
    Installing collected packages: catboost
    Successfully installed catboost-0.24.4
    


```python
# import useful libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid") # set seaborn graphing style

import catboost
from catboost import *
```


```python
# mount G-drive to get data
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
# load in data and set seed
BASE = "/content/drive/My Drive/JanataHack/Crossell/data/"
SEED = 2020

train_df = pd.read_csv(f"{BASE}train.csv")
test_df = pd.read_csv(f"{BASE}test.csv")
submit_df = pd.read_csv(f"{BASE}sample_submission.csv")
```

## EDA starts
First we look at the first few rows of all datasets.


```python
train_df.head()
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
      <th>id</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>44</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>&gt; 2 Years</td>
      <td>Yes</td>
      <td>40454.0</td>
      <td>26.0</td>
      <td>217</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>76</td>
      <td>1</td>
      <td>3.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>No</td>
      <td>33536.0</td>
      <td>26.0</td>
      <td>183</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Male</td>
      <td>47</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>&gt; 2 Years</td>
      <td>Yes</td>
      <td>38294.0</td>
      <td>26.0</td>
      <td>27</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Male</td>
      <td>21</td>
      <td>1</td>
      <td>11.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>28619.0</td>
      <td>152.0</td>
      <td>203</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>29</td>
      <td>1</td>
      <td>41.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>27496.0</td>
      <td>152.0</td>
      <td>39</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submit_df.head()
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
      <th>id</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>381110</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>381111</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>381112</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>381113</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>381114</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# look at distribution of target variable
train_df["Response"].value_counts()
```




    0    334399
    1     46710
    Name: Response, dtype: int64




```python
# look at which variables are null and if they were parsed correctly
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 381109 entries, 0 to 381108
    Data columns (total 12 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   id                    381109 non-null  int64  
     1   Gender                381109 non-null  object 
     2   Age                   381109 non-null  int64  
     3   Driving_License       381109 non-null  int64  
     4   Region_Code           381109 non-null  float64
     5   Previously_Insured    381109 non-null  int64  
     6   Vehicle_Age           381109 non-null  object 
     7   Vehicle_Damage        381109 non-null  object 
     8   Annual_Premium        381109 non-null  float64
     9   Policy_Sales_Channel  381109 non-null  float64
     10  Vintage               381109 non-null  int64  
     11  Response              381109 non-null  int64  
    dtypes: float64(3), int64(6), object(3)
    memory usage: 34.9+ MB
    


```python
test_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 127037 entries, 0 to 127036
    Data columns (total 11 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   id                    127037 non-null  int64  
     1   Gender                127037 non-null  object 
     2   Age                   127037 non-null  int64  
     3   Driving_License       127037 non-null  int64  
     4   Region_Code           127037 non-null  float64
     5   Previously_Insured    127037 non-null  int64  
     6   Vehicle_Age           127037 non-null  object 
     7   Vehicle_Damage        127037 non-null  object 
     8   Annual_Premium        127037 non-null  float64
     9   Policy_Sales_Channel  127037 non-null  float64
     10  Vintage               127037 non-null  int64  
    dtypes: float64(3), int64(5), object(3)
    memory usage: 10.7+ MB
    

No nulls, therefore no imputation needed! For examples of imputation, please check out my other templates!

### Looking at categorical columns
Because of all the categorical columns I decided to set a baseline in Catboost. Here are value counts and countplots for all of them, they prove useful to gauge relationship with the target column (if any).


```python
train_df["Gender"].value_counts()
```




    Male      206089
    Female    175020
    Name: Gender, dtype: int64




```python
plt.figure(figsize=(20,5))
sns.countplot(x="Gender", hue="Response", data=train_df);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_16_0.png)
    


Slightly more male than female drivers, proportion of insurance is also higher for men.


```python
train_df["Driving_License"].value_counts()
```




    1    380297
    0       812
    Name: Driving_License, dtype: int64




```python
plt.figure(figsize=(20,5))
sns.countplot(x="Driving_License", hue="Response", data=train_df);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_19_0.png)
    


Looks like everyone in the dataset has a driver's license!


```python
train_df["Region_Code"].value_counts().head(25)
```




    28.0    106415
    8.0      33877
    46.0     19749
    41.0     18263
    15.0     13308
    30.0     12191
    29.0     11065
    50.0     10243
    3.0       9251
    11.0      9232
    36.0      8797
    33.0      7654
    47.0      7436
    35.0      6942
    6.0       6280
    45.0      5605
    37.0      5501
    18.0      5153
    48.0      4681
    14.0      4678
    39.0      4644
    10.0      4374
    21.0      4266
    2.0       4038
    13.0      4036
    Name: Region_Code, dtype: int64




```python
plt.figure(figsize=(20,5))
sns.countplot(x="Region_Code", hue="Response", data=train_df);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_22_0.png)
    


We could add a features that flags the most popular regions. Or the ones with the highest proportion of positive response.


```python
train_df["Previously_Insured"].value_counts()
```




    0    206481
    1    174628
    Name: Previously_Insured, dtype: int64




```python
plt.figure(figsize=(20,5))
sns.countplot(x="Previously_Insured", hue="Response", data=train_df);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_25_0.png)
    



```python
train_df["Vehicle_Age"].value_counts()
```




    1-2 Year     200316
    < 1 Year     164786
    > 2 Years     16007
    Name: Vehicle_Age, dtype: int64




```python
plt.figure(figsize=(20,5))
sns.countplot(x="Vehicle_Age", hue="Response", data=train_df);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_27_0.png)
    


Looks like there a much higher proportion of insured cars in 1-2 year category, than in the < 1 year category.


```python
train_df["Vehicle_Damage"].value_counts()
```




    Yes    192413
    No     188696
    Name: Vehicle_Damage, dtype: int64




```python
plt.figure(figsize=(20,5))
sns.countplot(x="Vehicle_Damage", hue="Response", data=train_df);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_30_0.png)
    


No vehicle damage => no insurance needed!


```python
train_df["Policy_Sales_Channel"].value_counts().head(25)
```




    152.0    134784
    26.0      79700
    124.0     73995
    160.0     21779
    156.0     10661
    122.0      9930
    157.0      6684
    154.0      5993
    151.0      3885
    163.0      2893
    13.0       1865
    25.0       1848
    7.0        1598
    8.0        1515
    30.0       1410
    55.0       1264
    155.0      1234
    11.0       1203
    1.0        1074
    52.0       1055
    125.0      1026
    15.0        888
    29.0        843
    12.0        783
    120.0       769
    Name: Policy_Sales_Channel, dtype: int64




```python
plt.figure(figsize=(20,5))
sns.countplot(x="Policy_Sales_Channel", hue="Response", data=train_df);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_33_0.png)
    


### Analysis of continuous variables
Plotted boxplots by target variable and kernel density estimates for each continuous variable to draw interesting insight.


```python
# insured people are older on average by about 7 years

sns.boxplot(train_df["Response"], train_df["Age"]);
```

    /usr/local/lib/python3.6/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_35_1.png)
    



```python
# bimodal distribution of age

sns.kdeplot(train_df["Age"]);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_36_0.png)
    



```python
# not much of a difference, lots of outliers even with log-transformation of premium paid

sns.boxplot(train_df["Response"], np.log10(train_df["Annual_Premium"]));
```

    /usr/local/lib/python3.6/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_37_1.png)
    



```python
# bimodal distribution of annual premium

sns.kdeplot(np.log10(train_df["Annual_Premium"]));
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_38_0.png)
    



```python
# this does not make much sense, but looking to see if it would be worth it not to encode this variable as continuous

sns.boxplot(train_df["Response"], train_df["Policy_Sales_Channel"]);
```

    /usr/local/lib/python3.6/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_39_1.png)
    



```python
sns.kdeplot(train_df["Policy_Sales_Channel"]);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_40_0.png)
    



```python
# not much of a difference here

sns.boxplot(train_df["Response"], train_df["Vintage"]);
```

    /usr/local/lib/python3.6/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_41_1.png)
    



```python
sns.kdeplot(train_df["Vintage"]);
```


    
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ClassificationEDABaseline_files/ClassificationEDABaseline_42_0.png)
    


Vintage doesn't seem to have an effect on target variable, Policy Sales Channel, Annual Premium, and Age do.

## Baseline Model
Alright, after basic EDA of all variables, it's time to introduce the basic Catboost model with no tuning as a baseline.


```python
# Data preparation
y = train_df['Response'].values
X = train_df.drop(['Response', 'id'], axis=1)
X.head()
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
      <th>Gender</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>44</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>&gt; 2 Years</td>
      <td>Yes</td>
      <td>40454.0</td>
      <td>26.0</td>
      <td>217</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>76</td>
      <td>1</td>
      <td>3.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>No</td>
      <td>33536.0</td>
      <td>26.0</td>
      <td>183</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>47</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>&gt; 2 Years</td>
      <td>Yes</td>
      <td>38294.0</td>
      <td>26.0</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>21</td>
      <td>1</td>
      <td>11.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>28619.0</td>
      <td>152.0</td>
      <td>203</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>29</td>
      <td>1</td>
      <td>41.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>27496.0</td>
      <td>152.0</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Categorical features declaration
cat_features = ["Gender", "Region_Code", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage", "Policy_Sales_Channel"]
print(cat_features)
```

    ['Gender', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']
    


```python
# convert to right data types
print(X[cat_features].info())

X_filled = X.copy()
X_filled["Region_Code"] = X["Region_Code"].astype(np.int16)
X_filled["Policy_Sales_Channel"] = X["Policy_Sales_Channel"].astype(np.int16)

X_filled[cat_features].info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 381109 entries, 0 to 381108
    Data columns (total 6 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   Gender                381109 non-null  object 
     1   Region_Code           381109 non-null  float64
     2   Previously_Insured    381109 non-null  int64  
     3   Vehicle_Age           381109 non-null  object 
     4   Vehicle_Damage        381109 non-null  object 
     5   Policy_Sales_Channel  381109 non-null  float64
    dtypes: float64(2), int64(1), object(3)
    memory usage: 17.4+ MB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 381109 entries, 0 to 381108
    Data columns (total 6 columns):
     #   Column                Non-Null Count   Dtype 
    ---  ------                --------------   ----- 
     0   Gender                381109 non-null  object
     1   Region_Code           381109 non-null  int16 
     2   Previously_Insured    381109 non-null  int64 
     3   Vehicle_Age           381109 non-null  object
     4   Vehicle_Damage        381109 non-null  object
     5   Policy_Sales_Channel  381109 non-null  int16 
    dtypes: int16(2), int64(1), object(3)
    memory usage: 13.1+ MB
    


```python
# split into training and test sets, shuffle and stratify
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_filled, y, train_size=0.8, random_state=SEED, shuffle=True, stratify=y)
```


```python
model = CatBoostClassifier(
    random_seed=SEED,    # make code reproducible
    eval_metric='AUC',   # evaluation metric used in the competition, reference here: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    task_type='GPU'      # use GPU to speed up training!
)
model.fit(
    X_train, y_train,                       # input train features and target
    cat_features=cat_features,              # input which variables to treat as categorical
    use_best_model=True,                    # cache best model to use after training is finished
    eval_set=(X_validation, y_validation),  # evaluate on valudation set
    verbose=50                              # print progress every 50th iteration
)
print('Model is fitted: ' + str(model.is_fitted()))
print('Model params:')
print(model.get_params())
```

    Learning rate set to 0.04379
    0:	learn: 0.7300705	test: 0.7350620	best: 0.7350620 (0)	total: 55.9ms	remaining: 55.8s
    50:	learn: 0.8473518	test: 0.8476181	best: 0.8476181 (50)	total: 2.09s	remaining: 38.9s
    100:	learn: 0.8541753	test: 0.8539259	best: 0.8539259 (100)	total: 4.09s	remaining: 36.4s
    150:	learn: 0.8565141	test: 0.8558266	best: 0.8558266 (150)	total: 6.01s	remaining: 33.8s
    200:	learn: 0.8576283	test: 0.8563952	best: 0.8563952 (200)	total: 7.94s	remaining: 31.6s
    250:	learn: 0.8584694	test: 0.8568907	best: 0.8568907 (250)	total: 9.84s	remaining: 29.4s
    300:	learn: 0.8591383	test: 0.8572520	best: 0.8572520 (300)	total: 11.8s	remaining: 27.3s
    350:	learn: 0.8596287	test: 0.8573664	best: 0.8573697 (349)	total: 13.6s	remaining: 25.2s
    400:	learn: 0.8601131	test: 0.8574828	best: 0.8574834 (397)	total: 15.5s	remaining: 23.1s
    450:	learn: 0.8605216	test: 0.8576151	best: 0.8576182 (449)	total: 17.3s	remaining: 21.1s
    500:	learn: 0.8608967	test: 0.8577058	best: 0.8577085 (498)	total: 19.2s	remaining: 19.2s
    550:	learn: 0.8612487	test: 0.8577447	best: 0.8577550 (546)	total: 21.1s	remaining: 17.2s
    600:	learn: 0.8615830	test: 0.8577949	best: 0.8578038 (595)	total: 23s	remaining: 15.2s
    650:	learn: 0.8618691	test: 0.8578118	best: 0.8578124 (609)	total: 24.8s	remaining: 13.3s
    700:	learn: 0.8622126	test: 0.8578691	best: 0.8578799 (691)	total: 26.7s	remaining: 11.4s
    750:	learn: 0.8625917	test: 0.8579514	best: 0.8579522 (749)	total: 28.6s	remaining: 9.47s
    800:	learn: 0.8628763	test: 0.8579737	best: 0.8579799 (775)	total: 30.4s	remaining: 7.55s
    850:	learn: 0.8630994	test: 0.8579803	best: 0.8579898 (834)	total: 32.2s	remaining: 5.65s
    900:	learn: 0.8633502	test: 0.8579678	best: 0.8580152 (864)	total: 34.1s	remaining: 3.75s
    950:	learn: 0.8635964	test: 0.8579413	best: 0.8580152 (864)	total: 36s	remaining: 1.85s
    999:	learn: 0.8638379	test: 0.8579124	best: 0.8580152 (864)	total: 37.8s	remaining: 0us
    bestTest = 0.8580151796
    bestIteration = 864
    Shrink model to first 865 iterations.
    Model is fitted: True
    Model params:
    {'task_type': 'GPU', 'eval_metric': 'AUC', 'random_seed': 2020}
    

Model did not improve the last ~150 iterations, which suggests it trained for about the right time, not to over- or underfit.


```python
# how many trees did we grow?

print('Tree count: ' + str(model.tree_count_))
```

    Tree count: 865
    


```python
# which features were the most beneficial for the model? Helpful for feature engineering (not covered here).

model.get_feature_importance(prettified=True)
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
      <th>Feature Id</th>
      <th>Importances</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Previously_Insured</td>
      <td>58.060342</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Vehicle_Damage</td>
      <td>19.048923</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Policy_Sales_Channel</td>
      <td>8.422820</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Age</td>
      <td>7.547913</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Region_Code</td>
      <td>3.859257</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vehicle_Age</td>
      <td>2.327715</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Annual_Premium</td>
      <td>0.325198</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Gender</td>
      <td>0.218071</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Driving_License</td>
      <td>0.118190</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Vintage</td>
      <td>0.071571</td>
    </tr>
  </tbody>
</table>
</div>




```python
# prepare data for prediction
X_test = test_df.drop(['id'], axis=1)
X_test.head()
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
      <th>Gender</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>25</td>
      <td>1</td>
      <td>11.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>35786.0</td>
      <td>152.0</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>40</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>33762.0</td>
      <td>7.0</td>
      <td>111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>47</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>40050.0</td>
      <td>124.0</td>
      <td>199</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>24</td>
      <td>1</td>
      <td>27.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>Yes</td>
      <td>37356.0</td>
      <td>152.0</td>
      <td>187</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>27</td>
      <td>1</td>
      <td>28.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>59097.0</td>
      <td>152.0</td>
      <td>297</td>
    </tr>
  </tbody>
</table>
</div>




```python
# convert to right data types in test dataframe
print(X_test[cat_features].info())

X_test_filled = X_test.copy()
X_test_filled["Region_Code"] = X_test["Region_Code"].astype(np.int16)
X_test_filled["Policy_Sales_Channel"] = X_test["Policy_Sales_Channel"].astype(np.int16)

X_test_filled[cat_features].info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 127037 entries, 0 to 127036
    Data columns (total 6 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   Gender                127037 non-null  object 
     1   Region_Code           127037 non-null  float64
     2   Previously_Insured    127037 non-null  int64  
     3   Vehicle_Age           127037 non-null  object 
     4   Vehicle_Damage        127037 non-null  object 
     5   Policy_Sales_Channel  127037 non-null  float64
    dtypes: float64(2), int64(1), object(3)
    memory usage: 5.8+ MB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 127037 entries, 0 to 127036
    Data columns (total 6 columns):
     #   Column                Non-Null Count   Dtype 
    ---  ------                --------------   ----- 
     0   Gender                127037 non-null  object
     1   Region_Code           127037 non-null  int16 
     2   Previously_Insured    127037 non-null  int64 
     3   Vehicle_Age           127037 non-null  object
     4   Vehicle_Damage        127037 non-null  object
     5   Policy_Sales_Channel  127037 non-null  int16 
    dtypes: int16(2), int64(1), object(3)
    memory usage: 4.4+ MB
    


```python
# use catboost Pool class to load the dataset, then predict probabilities of `Response` == 1
test_pool = Pool(data=X_test_filled, cat_features=cat_features)
contest_predictions = model.predict_proba(test_pool)[:,1]
print('Predictions:')
print(contest_predictions)
```

    Predictions:
    [0.00099469 0.30234474 0.29278176 ... 0.0006389  0.00062148 0.00326372]
    


```python
# replace sample submission column
submit_df["Response"] = contest_predictions
submit_df.head()
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
      <th>id</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>381110</td>
      <td>0.000995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>381111</td>
      <td>0.302345</td>
    </tr>
    <tr>
      <th>2</th>
      <td>381112</td>
      <td>0.292782</td>
    </tr>
    <tr>
      <th>3</th>
      <td>381113</td>
      <td>0.005886</td>
    </tr>
    <tr>
      <th>4</th>
      <td>381114</td>
      <td>0.000909</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create a csv to submit to competition!

submit_df.to_csv("Catboost_Baseline.csv", index = False)
```
