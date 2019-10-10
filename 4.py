import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from hyperopt import hp
from random import sample
df = pd.read_csv("INM00042103.csv")

#df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# for i in df.columns:
#     print(i, ': ', df[i].unique())
#
# print(df['Load'].isna().sum())  # 35064

df= df.replace(-9999,0)
df = df.replace(np.nan, 0)
df['gph'] = df['gph'].map( lambda x : df.gph.mean() if x == 0 else x)
df['pressure'] = df['pressure'].map( lambda x : df.pressure.mean() if x == 0 else x)
df['wdir'] = df['wdir'].map( lambda x : df.wdir.mean() if x == 0 else x)
df['wspd'] = df['wspd'].map( lambda x : df.wspd.mean() if x == 0 else x)
df['temp'] = df['temp'].map( lambda x : df.temp.mean() if x == 0 else x)
df['rh'] = df['rh'].map( lambda x : df.rh.mean() if x == 0 else x)
df['dpdp'] = df['dpdp'].map( lambda x : df.dpdp.mean() if x == 0 else x)
df['reltime'] = df['reltime'].map( lambda x : df.reltime.mean() if x == 0 else x)
df['npv'] = df['npv'].map( lambda x : df.npv.mean() if x == 0 else x)

# print('\n\nLength: ', len(df), '\n\n')

lvl12_1 = []
lvl12_2 = []

lvl12 = df['lvl12'].tolist()

for i in lvl12:
        lvl12_2.append(int(i%10))
        lvl12_1.append(int(i/10))


# print(lvl12_1[:10])
# print(lvl12_2[:10])

df = df.drop(['lvl12'], axis=1)
df['lvl12_1'] = pd.DataFrame(lvl12_1)
df['lvl12_2'] = pd.DataFrame(lvl12_2)

# One Hot Encoding

values = np.array(df['lvl12_1'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

df1 = pd.DataFrame(onehot_encoded)
df1.columns = ['lvl12_1_0', 'lvl12_1_1', 'lvl12_1_2', 'lvl12_1_3']
# print(df1.head())

values = np.array(df['lvl12_2'])
label_encoder = LabelEncoder()


values = np.array(df['lvl12_1'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

df1 = pd.DataFrame(onehot_encoded)
df1.columns = ['lvl12_1_0', 'lvl12_1_1', 'lvl12_1_2', 'lvl12_1_3']
# print(df1.head())

values = np.array(df['lvl12_2'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

df2 = pd.DataFrame(onehot_encoded)
df2.columns = ['lvl12_2_0', 'lvl12_2_1']
# print(df2.head())

df = df.drop(['lvl12_1', 'lvl12_2'], axis=1)
df = df.join(df1)
df = df.join(df2)

ld = df['wspd'].tolist()
ld_7 = []

for i in range(len(ld)):
    if i >= 1:
        ld_7.append(ld[i-1])
    else:
        ld_7.append(np.mean(ld[:1]))


df['load_h1'] = pd.DataFrame(ld_7)

#ld1 = df['Load'].tolist()
# ld_8 = []
#
# for i in range(len(ld)):
#     if i >= 168:
#         ld_8.append(ld[i - 168])
#     else:
#         ld_8.append(np.mean(ld[:168]))
#
# df['load_d7'] = pd.DataFrame(ld_8)
#

ld2 = df['wspd'].tolist()
ld_9 = []

for i in range(len(ld)):
    if i >= 2:
        ld_9.append(ld[i-2])
    else:
        ld_9.append(np.mean(ld[:2]))


df['load_h2'] = pd.DataFrame(ld_9)


ld = df['wspd'].tolist()
ld_10 = []

for i in range(len(ld)):
    if i >= 3:
        ld_10.append(ld[i-3])
    else:
        ld_10.append(np.mean(ld[:3]))


df['load_h3'] = pd.DataFrame(ld_10)

ld = df['wspd'].tolist()
ld_11 = []

for i in range(len(ld)):
    if i >= 4:
        ld_11.append(ld[i-4])
    else:
        ld_11.append(np.mean(ld[:4]))


df['load_h4'] = pd.DataFrame(ld_11)



#df['Year'] = df['Date'].dt.year
#df['Month'] = df['Date'].dt.month
#df['Day'] = df['Date'].dt.day
#df['DayOfYear'] = df['Date'].dt.dayofyear
#df['DayOfWeek'] = df['Date'].dt.dayofweek
#df['WeekOfYear'] = df['Date'].dt.weekofyear
#df = df.drop(['Date'], axis=1)

# print(df.columns)
# print(df.info())

df= df.drop(['etime','station-code','longitude','lattitude'], axis=1)
df = df[np.isfinite(df['wspd'])]

is_not_2017 = df['year'] != 2017
train = df[is_not_2017]

is_2017 = df['year'] == 2017
test = df[is_2017]

print(df.shape)
print(train.shape)
print(test.shape)

# print(train.head())
# print(test.head())
#
# print(train['Year'].unique())
# print(test['Year'].unique())


# model = tf.keras.Sequential()


X_train = pd.DataFrame(train.drop(['wspd'], axis=1))
Y_train = pd.DataFrame(train['wspd'])

X_test = pd.DataFrame(test.drop(['wspd'], axis=1))
Y_test = pd.DataFrame(test['wspd'])

print('train:\n', train.head())
print('test:\n', test.head())
print('X_train:\n', X_train.head())
print('Y_train:\n', Y_train.head())
print('X_test:\n', X_test.head())
print('Y_test:\n', Y_test.head())

numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.dtypes)

corr =numeric_features.corr()
print(corr['wspd'].sort_values(ascending=False))
#print(corr)
#correlation matrix
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=1, square=True)
plt.show()

#
#
#
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

params = {'task':'train', 'boosting_type':'gbdt', 'objective':'regression',
              'metric': {'rmse'}, 'num_leaves': 8, 'learning_rate': 0.05,
              'feature_fraction': 0.8, 'max_depth': 7, 'verbose': 0,
              'num_boost_round':25000, #'early_stopping_rounds':100,
           'nthread':-1}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval)
                #early_stopping_rounds=5)

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(Y_test, y_pred) ** 0.5)

#
df_test = pd.get_dummies(test)
df_test.drop('wspd',axis=1,inplace=True)
X_prediction = df_test.values
#
predictions = gbm.predict(X_prediction,num_iteration=gbm.best_iteration)

sub = test.loc[:,['wspd']]
sub['sales']= predictions
sub['hour']=test.loc[:,['hour']]
sub['day']=test.loc[:,['day']]
sub['month']=test.loc[:,['month']]
sub['year']=test.loc[:,['year']]
sub['Accuracy']= 100 - (abs(sub['sales']-sub['wspd'])/sub['wspd'])*100
print(sub.describe())
df3 = pd.read_excel('1.xls')
print(df3.head())
df3 = sub
df3.to_excel('1.xls')