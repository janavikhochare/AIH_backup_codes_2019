import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
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



#code

#df = df[np.isfinite(df['wspd'])]

# is_not_2016 = df['year'] != 2016
# train = df[is_not_2016]
#
# is_2016 = df['year'] == 2016
# test = df[is_2016]
#
# is_not_2017 = df['year'] != 2017
train = df[:18000]
test=df[18000:]
#
# is_2017 = df['year'] == 2017
# test = df[is_2017]
# print(df.shape)


print(train.shape)
print(test.shape)

# print(train.head())
# print(test.head())
#
# print(train['Year'].unique())
# print(test['Year'].unique())


# model = tf.keras.Sequential()


training_x = pd.DataFrame(train.drop(['wspd'], axis=1))
training_y = pd.DataFrame(train['wspd'])

test_x = pd.DataFrame(test.drop(['wspd'], axis=1))
test_y = pd.DataFrame(test['wspd'])

print('train:\n', train.head())
print('test:\n', test.head())
print('X_train:\n', training_x.head())
print('Y_train:\n', training_y.head())
print('X_test:\n', test_x.head())
print('Y_test:\n', test_y.head())

numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.dtypes)

corr =numeric_features.corr()
print(corr['wspd'].sort_values(ascending=False))
#print(corr)
#correlation matrix
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=1, square=True)
plt.show()

#training_x, training_y, test_x, test_y = functions_single.split_dataset(merged_df_scaled, input_variables=input_variables, output_variables=output_variables, ratio=ratio)


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1,  )))
model.add(LSTM(50 ))
model.add(Dense(len(output_variables), activation='sigmoid'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(training_x, training_y, epochs=epochs, batch_size=20, verbose=2)

#merged_df_length = len(merged_df)
#training_length = int(ratio * merged_df_length)

train_predict = model.predict(training_x)
test_predict = model.predict(test_x, batch_size=300)

train_predict_plot = train_predict[:,0]
train_predict_plot = numpy.reshape(train_predict_plot, ( , 1))

test_predict_plot = numpy.empty(merged_df_length)
test_predict_plot[:] = numpy.nan
test_predict_plot[len(train_predict_plot)+1:merged_df_length] = test_predict[:,0]



scaled_wind_avg = numpy.array(merged_df_scaled['wind_avg_in24h'])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(training_y[:,0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_y[:,0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# calculate mean absolute error
trainScore = mean_absolute_error(training_y[:,0], train_predict[:,0])
print('Train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(test_y[:,0], test_predict[:,0])
print('Test Score: %.2f MAE' % (testScore))




plt.plot(scaled_wind_avg, "b")
plt.plot(train_predict_plot)
plt.plot(test_predict_plot, "g--")
plt.show()
