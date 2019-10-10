from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import read_csv
from datetime import datetime
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

ld = df['wspd'].tolist()
ld_11 = []

for i in range(len(ld)):
    if i >= 4:
        ld_11.append(ld[i-4])
    else:
        ld_11.append(np.mean(ld[:4]))


df['before_24hr'] = pd.DataFrame(ld_11)

ld = df['wspd'].tolist()
ld_17 = []

for i in range(len(ld)):
    if i >= 5 :
        ld_17.append(ld[i-5])
    else:
        ld_17.append(np.mean(ld[:5]))


df['before_30hr'] = pd.DataFrame(ld_17)

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





#dataset = read_csv('pollution.csv', header=0, index_col=0)
#values = dataset.values
# specify columns to plot

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = df
dataset =dataset[['wdir','gph','month','wspd','before_6hr' ,'before_6hr_wdir','before_12hr','before_12hr_wdir',
     'before_18hr','before_24hr','before_30hr','before_36hr' ,'before_42hr','before_48hr']]
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:, 7] = encoder.fit_transform(values[:, 7])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[8,9, 10, 11, 12, 13, 14]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = 18000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

df3 = pd.DataFrame()
df3['inv_y']=inv_y
df3['test_y']=test_y
#df3['yhat']= yhat
#df3['inv_yhat']=inv_yhat
df3.to_excel('1.xls')