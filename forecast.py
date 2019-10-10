from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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
df['lvl12_2'] = pd.DataFrame(lvl12_2)

# One Hot Encoding

values = np.array(df['lvl12_1'])
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

values = df.values


#is_not_2017 = df.head(18490)
train = df[:18000]

#is_2017 = df.tail (10)
test = df[18000:]


X_train = pd.DataFrame(train.drop(['wspd'], axis=1))
#X_train= np.array(X_train)
Y_train = pd.DataFrame(train['wspd'])
#Y_train = np

X_test = pd.DataFrame(test.drop(['wspd'], axis=1))
Y_test = pd.DataFrame(test['wspd'])
print(df.shape)
print(train.shape)
print(test.shape)

# print(train.head())
# print(test.head())
#
# print(train['Year'].unique())
# print(test['Year'].unique())


# model = tf.keras.Sequential()



X_train=X_train.values
X_test= X_test.values
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(X_train.shape)
print(X_test.shape)
print('train:\n', train.head())
print('test:\n', test.head())
#print('X_train:\n', X_train.head())
print('Y_train:\n', Y_train.head())
#print('X_test:\n', X_test.head())
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

print(X_train.shape)
print(Y_train.shape)
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

input_variables = ['wdir','gph','before_6hr' ,'before_6hr_wdir','before_12hr','before_12hr_wdir',
     'before_18hr','before_24hr','before_30hr','before_36hr' ,'before_42hr','before_48hr']
output_variables = ['wspd']


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))


model.summary()
#X_train= np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=20, batch_size=20, verbose=2)

#merged_df_length = len(merged_df)
#training_length = int(ratio * merged_df_length)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test, batch_size=300)

train_predict_plot = train_predict[:,0]
train_predict_plot = np.reshape(train_predict_plot, ( 18000, 1))

test_predict_plot = np.empty(500)
test_predict_plot[:] = np.nan
test_predict_plot[len(train_predict_plot)+1:500] = test_predict[:500]

scaled_wind_avg = np.array(df['wspd'])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(Y_train[:,0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_test[:,0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# calculate mean absolute error
trainScore = mean_absolute_error(Y_train[:,0], train_predict[:,0])
print('Train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(Y_test[:,0], test_predict[:,0])
print('Test Score: %.2f MAE' % (testScore))




plt.plot(scaled_wind_avg, "b")
plt.plot(train_predict_plot)
plt.plot(test_predict_plot, "g--")
plt.show()