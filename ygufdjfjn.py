import time
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import pandas as pd
pd.plotting.register_matplotlib_converters()
from fastai.imports import *
#from fastai.structured import *
from fbprophet import Prophet

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold
from scipy import stats
from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go


import statsmodels.api as sm
# Initialize plotly
init_notebook_mode(connected=True)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
import matplotlib.pylab as plt
import seaborn as sns

warnings.filterwarnings('ignore')


df= pd.read_csv('IM_group.csv')

from datetime import datetime
df= df.drop(['station-code'],axis=1)
df = df.iloc[574:]
df=df.astype(int)
df['Date'] = df.apply(lambda row: datetime(row['year'], row['month'], row['day'],row['hour']), axis=1)

#df[['hour']] = df[['hour']].apply(pd.to_numeric)
#df['hour'] = pd.to_numeric(df['hour'])

print(df.dtypes)
# print('\n\nLength: ', len(df), '\n\n')
# print('\n\nLength: ', len(df), '\n\n')
#
# lvl12_1 = []
# lvl12_2 = []
#
# lvl12 = df['lvl12'].tolist()
#
# for i in lvl12:
#         lvl12_2.append(int(i%10))
#         lvl12_1.append(int(i/10))
#
#
# df = df.drop(['lvl12'], axis=1)
# df['lvl12_1'] = pd.DataFrame(lvl12_1)
#
# values = np.array(df['lvl12_1'])
# df['lvl12_2'] = pd.DataFrame(lvl12_2)
#
# # One Hot Encoding
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#
# df1 = pd.DataFrame(onehot_encoded)
# print(df1.head())
# df1.columns = ['lvl12_1_0', 'lvl12_1_1']
# # print(df1.head())
#
# values = np.array(df['lvl12_2'])
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#
# df2 = pd.DataFrame(onehot_encoded)
# print(df2.head())
# df2.columns = ['lvl12_2_0']
# # print(df2.head())
#
# df = df.drop(['lvl12_1', 'lvl12_2'], axis=1)
# df = df.join(df1)
# df = df.join(df2)
df.index = pd.to_datetime(df.index, unit='s')

print('\n\n')
for i in df.columns:
        print(i, ': ', df[i].unique()
              )

df= df[['wdir','pressure','wspd','gph','npv','day','hour','month','year']]#,'lvl12_1_0','lvl12_1_1']]
ld = df['wspd'].tolist()
ld_7 = []

for i in range(len(ld)):
    if i >= 1 :
        ld_7.append(ld[i-1])
    else:
        ld_7.append(np.mean(ld[:1]))


df['before_6hr'] = pd.DataFrame(ld_7)

lda = df['wdir'].tolist()
ld_7a = []

for i in range(len(lda)):
    if i >= 1 :
        ld_7a.append(lda[i-1])
    else:
        ld_7a.append(np.mean(lda[:1]))


df['before_6hr_wdir'] = pd.DataFrame(ld_7a)

ld2 = df['wspd'].tolist()
ld_9 = []

for i in range(len(ld)):
    if i >= 2:
        ld_9.append(ld[i-2])
    else:
        ld_9.append(np.mean(ld[:2]))


df['before_12hr'] = pd.DataFrame(ld_9)



ld2a = df['wdir'].tolist()
ld_9a = []

for i in range(len(lda)):
    if i >= 2:
        ld_9a.append(lda[i-2])
    else:
        ld_9a.append(np.mean(lda[:2]))


df['before_12hr_wdir'] = pd.DataFrame(ld_9a)


ld = df['wspd'].tolist()
ld_10 = []

for i in range(len(ld)):
    if i >= 3:
        ld_10.append(ld[i-3])
    else:
        ld_10.append(np.mean(ld[:3]))


df['before_18hr'] = pd.DataFrame(ld_10)


lda = df['wdir'].tolist()
ld_10a = []

for i in range(len(lda)):
    if i >= 3:
        ld_10a.append(lda[i-3])
    else:
        ld_10a.append(np.mean(lda[:3]))


df['before_18hr_wdir'] = pd.DataFrame(ld_10a)


ld = df['wspd'].tolist()
ld_11 = []

for i in range(len(ld)):
    if i >= 4:
        ld_11.append(ld[i-4])
    else:
        ld_11.append(np.mean(ld[:4]))


df['before_24hr'] = pd.DataFrame(ld_11)


lda = df['wdir'].tolist()
ld_11a = []

for i in range(len(lda)):
    if i >= 4:
        ld_11a.append(lda[i-4])
    else:
        ld_11a.append(np.mean(lda[:4]))


df['before_24hr_wdir'] = pd.DataFrame(ld_11a)


ld = df['wspd'].tolist()
ld_17 = []

for i in range(len(ld)):
    if i >= 5 :
        ld_17.append(ld[i-5])
    else:
        ld_17.append(np.mean(ld[:5]))


df['before_30hr'] = pd.DataFrame(ld_17)


lda = df['wdir'].tolist()
ld_17a = []

for i in range(len(lda)):
    if i >= 5:
        ld_17a.append(lda[i-5])
    else:
        ld_17a.append(np.mean(lda[:5]))


df['before_30hr_wdir'] = pd.DataFrame(ld_17a)

ld = df['wspd'].tolist()
ld_19 = []

for i in range(len(ld)):
    if i >= 6 :
        ld_19.append(ld[i-6])
    else:
        ld_19.append(np.mean(ld[:6]))


df['before_36hr'] = pd.DataFrame(ld_19)

ld = df['wspd'].tolist()
ld_20 = []

for i in range(len(ld)):
    if i >= 7 :
        ld_20.append(ld[i-7])
    else:
        ld_20.append(np.mean(ld[:7]))


df['before_42hr'] = pd.DataFrame(ld_20)

ld = df['wspd'].tolist()
ld_21 = []

for i in range(len(ld)):
    if i >= 8 :
        ld_21.append(ld[i-8])
    else:
        ld_21.append(np.mean(ld[:8]))


df['before_48hr'] = pd.DataFrame(ld_21)



# Using Pearson Correlation

cor = df.corr()
#print('\n\n', cor)

# Correlation with output variable
cor_target = abs(cor['wspd'])
print("\n\nCorrelation:\n", cor_target, '\n')



#code

#df = df[np.isfinite(df['wspd'])]

is_not_2016 = df['year'] != 2017
train = df[is_not_2016]

is_2016 = df['year'] == 2017
test = df[is_2016]



pd.option_context("display.max_rows", 1000)
pd.option_context("display.max_columns", 1000)
#os.getcwd()
#PATH = '/home/janavi/Desktop/demand_forcast'
#print(os.listdir(PATH))
df_raw = train
df_test = test
#subs = pd.read_csv('/home/janavi/Desktop/demand_forcast/sample_submission.csv')


df_raw.head()

print("Train and Test shape are {} and {} respectively".format(df_raw.shape, df_test.shape))
#### Seasonality Check
# preparation: input should be float type
df_raw['wpsd'] = df_raw['wspd'] * 1.0

# store types
sales_a = df_raw[df_raw.hour == 0]['wspd'].sort_index(ascending = True)
sales_b = df_raw[df_raw.hour  == 6]['wspd'].sort_index(ascending = True) # solve the reverse order
sales_c = df_raw[df_raw.hour  == 12]['wspd'].sort_index(ascending = True)
sales_d = df_raw[df_raw.hour  == 18]['wspd'].sort_index(ascending = True)

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))
c = '#386B7F'


# store types
sales_a.resample('D').sum().plot(color = c, ax = ax1)
sales_b.resample('W').sum().plot(color = c, ax = ax2)
sales_c.resample('W').sum().plot(color = c, ax = ax3)
sales_d.resample('W').sum().plot(color = c, ax = ax4)

#All Stores have same trend... Weird Seems like the dataset is A Synthetic One..;
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))

# Yearly
decomposition_a = sm.tsa.seasonal_decompose(sales_a, model = 'additive', freq = 365)
decomposition_a.trend.plot(color = c, ax = ax1)

decomposition_b = sm.tsa.seasonal_decompose(sales_b, model = 'additive', freq = 365)
decomposition_b.trend.plot(color = c, ax = ax2)

decomposition_c = sm.tsa.seasonal_decompose(sales_c, model = 'additive', freq = 365)
decomposition_c.trend.plot(color = c, ax = ax3)

decomposition_d = sm.tsa.seasonal_decompose(sales_d, model = 'additive', freq = 365)
decomposition_d.trend.plot(color = c, ax = ax4)
date_sales = df_raw.drop(['wspd'], axis=1).copy() #it's a temporary DataFrame.. Original is Still intact..

date_sales.get_ftype_counts()
y = date_sales['wspd'].resample('MS').mean()
#y['2017':] #sneak peak
y.plot(figsize=(15, 6),)


decomposition = sm.tsa.seasonal_decompose(y, model='additive')
decomposition.plot()
decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')
decomposition.plot()