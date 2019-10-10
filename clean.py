import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
import matplotlib.pylab as plt
import seaborn as sns

warnings.filterwarnings('ignore')


df= pd.read_csv('IM_group.csv')

#df[['hour']] = df[['hour']].apply(pd.to_numeric)
#df['hour'] = pd.to_numeric(df['hour'])

print(df.dtypes)
# print('\n\nLength: ', len(df), '\n\n')
# print('\n\nLength: ', len(df), '\n\n')

lvl12_1 = []
lvl12_2 = []

lvl12 = df['lvl12'].tolist()

for i in lvl12:
        lvl12_2.append(int(i%10))
        lvl12_1.append(int(i/10))


df = df.drop(['lvl12'], axis=1)
df['lvl12_1'] = pd.DataFrame(lvl12_1)

values = np.array(df['lvl12_1'])
df['lvl12_2'] = pd.DataFrame(lvl12_2)

# One Hot Encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

df1 = pd.DataFrame(onehot_encoded)
print(df1.head())
df1.columns = ['lvl12_1_0', 'lvl12_1_1']
# print(df1.head())

values = np.array(df['lvl12_2'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

df2 = pd.DataFrame(onehot_encoded)
print(df2.head())
df2.columns = ['lvl12_2_0']
# print(df2.head())

df = df.drop(['lvl12_1', 'lvl12_2'], axis=1)
df = df.join(df1)
df = df.join(df2)
print('\n\n')
for i in df.columns:
        print(i, ': ', df[i].unique())

df= df[['wdir','pressure','wspd','gph','npv','day','hour','month','year','lvl12_1_0','lvl12_1_1']]
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

# is_not_2016 = df['year'] != 2017
# train = df[is_not_2016]
#
# is_2016 = df['year'] == 2017
# test = df[is_2016]

#is_not_2017 = df.head(18490)
train = df[:18490]

#is_2017 = df.tail (10)
test = df[18490:]

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
              'num_boost_round':2500, 'early_stopping_rounds':500,
           'nthread':-1}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

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

scaled_wind_avg = np.array(df3['sales'])
previous_values_of_windspd=np.array(df3['wspd'])

plt.plot(scaled_wind_avg, "b")
plt.plot(previous_values_of_windspd,"g")
plt.show()