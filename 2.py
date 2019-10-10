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




from sklearn import preprocessing

pd.options.mode.chained_assignment = None
np.random.seed(1234)


def drop_duplicates(df):
        print("Number of duplicates: {}".format(len(df.index.get_duplicates())))
        return df[~df.index.duplicated(keep='first')]


def impute_missing(df):
        # todo test with moving average / mean or something smarter than forward fill
        print("Number of rows with nan: {}".format(np.count_nonzero(df.isnull())))
        df.fillna(method='ffill', inplace=True)
        return df


def first_order_difference(data, columns):
        for column in columns:
                data[column + '_d'] = data[column].diff(periods=1)

        return data.dropna()


def derive_prediction_columns(data, column, horizons):
        for look_ahead in horizons:
                data['prediction_' + str(look_ahead)] = data[column].diff(periods=look_ahead).shift(-look_ahead)

        return data.dropna()


def scale_features(scaler, features):
        scaler.fit(features)

        scaled = scaler.transform(features)
        scaled = pd.DataFrame(scaled, columns=features.columns)

        return scaled


def inverse_prediction_scale(scaler, predictions, original_columns, column):
        loc = original_columns.get_loc(column)

        inverted = np.zeros((len(predictions), len(original_columns)))
        inverted[:, loc] = np.reshape(predictions, (predictions.shape[0],))

        inverted = scaler.inverse_transform(inverted)[:, loc]

        return inverted


def invert_all_prediction_scaled(scaler, predictions, original_columns, horizons):
        inverted = np.zeros(predictions.shape)

        for col_idx, horizon in enumerate(horizons):
                inverted[:, col_idx] = inverse_prediction_scale(
                        scaler, predictions[:, col_idx],
                        original_columns,
                        "prediction_" + str(horizon))

        return inverted


def inverse_prediction_difference(predictions, original):
        return predictions + original


def invert_all_prediction_differences(predictions, original):
        inverted = predictions

        for col_idx, horizon in enumerate(horizons):
                inverted[:, col_idx] = inverse_prediction_difference(predictions[:, col_idx], original)

        return inverted

dataset = drop_duplicates(df)
dataset = impute_missing(df)

#select features we're going to use
features = dataset[['wdir','gph','month','wspd']]
    #,'before_6hr' ,'before_6hr_wdir','before_12hr','before_12hr_wdir',
     #               'before_18hr','before_24hr','before_30hr','before_36hr' ,'before_42hr','before_48hr']]

# the time horizons we're going to predict (in hours)
horizons = [1, 6, 12, 24]

features = first_order_difference(features, features.columns)
features = derive_prediction_columns(features, 'wspd', horizons)

scaler = preprocessing.StandardScaler()
scaled = scale_features(scaler, features)

scaled.describe()


def prepare_test_train(data, features, predictions, sequence_length, split_percent=0.9):
    num_features = len(features)
    num_predictions = len(predictions)

    # make sure prediction cols are at end
    columns = features + predictions

    data = data[columns].values

    print("Using {} features to predict {} horizons".format(num_features, num_predictions))

    result = []
    for index in range(len(data) - sequence_length + 1):
        result.append(data[index:index + sequence_length])

    result = np.array(result)
    # shape (n_samples, sequence_length, num_features + num_predictions)
    print("Shape of data: {}".format(np.shape(result)))

    row = round(split_percent * result.shape[0])
    train = result[:row, :]

    X_train = train[:, :, :-num_predictions]
    y_train = train[:, -1, -num_predictions:]
    X_test = result[row:, :, :-num_predictions]
    y_test = result[row:, -1, -num_predictions:]

    print("Shape of X train: {}".format(np.shape(X_train)))
    print("Shape of y train: {}".format(np.shape(y_train)))
    print("Shape of X test: {}".format(np.shape(X_test)))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], num_features))

    y_train = np.reshape(y_train, (y_train.shape[0], num_predictions))
    y_test = np.reshape(y_test, (y_test.shape[0], num_predictions))

    return X_train, y_train, X_test, y_test, row
sequence_length = 48

prediction_cols = ['prediction_' + str(h) for h in horizons]
feature_cols = ['wdir','gph','month','wspd']
                #'before_6hr' ,'before_6hr_wdir','before_12hr','before_12hr_wdir',
                #'before_18hr','before_24hr','before_30hr','before_36hr' ,'before_42hr','before_48hr']

X_train, y_train, X_test, y_test, row_split = prepare_test_train(
    scaled,
    feature_cols,
    prediction_cols,
    sequence_length,
    split_percent = 0.9)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# (-1 is because we only take the last y row in each sequence)
sequence_offset = sequence_length - 1

# validate train
inverse_scale = invert_all_prediction_scaled(scaler, y_train, scaled.columns, horizons)

assert (mean_squared_error(
    features[prediction_cols][sequence_offset:row_split + sequence_offset],
    inverse_scale) < 1e-10)

undiff_prediction = invert_all_prediction_differences(
    inverse_scale,
    features['wspd'][sequence_offset:row_split + sequence_offset])

for i, horizon in enumerate(horizons):
    assert (mean_squared_error(
        features['wspd'][sequence_offset + horizon:row_split + sequence_offset + horizon],
        undiff_prediction[:, i]) < 1e-10)

# validate test
inverse_scale = invert_all_prediction_scaled(scaler, y_test, scaled.columns, horizons)

assert (mean_squared_error(
    features[prediction_cols][sequence_offset + row_split:],
    inverse_scale) < 1e-10)

undiff_prediction = invert_all_prediction_differences(
    inverse_scale,
    features['wspd'][sequence_offset + row_split:])

for i, horizon in enumerate(horizons):
    assert (mean_squared_error(
        features['wspd'][sequence_offset + row_split + horizon:],
        undiff_prediction[:-horizon, i]) < 1e-10)
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[3], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(layers[4], activation="linear"))

    model.compile(loss="mse", optimizer='rmsprop')

    print(model.summary())

    return model


from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau


def run_network(X_train, y_train, X_test, layers, epochs, batch_size=512):
    model = build_model(layers)
    history = None

    try:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[
                TensorBoard(log_dir='/tmp/tensorboard', write_graph=True),
                # EarlyStopping(monitor='val_loss', patience=5, mode='auto')
            ])
    except KeyboardInterrupt:
        print("\nTraining interrupted")

    predicted = model.predict(X_test)

    return model, predicted, history

model, predicted, history = run_network(
    X_train,
    y_train,
    X_test,
    layers=[X_train.shape[2], 20, 15, 20, y_train.shape[1]],
    epochs=20)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()

from sklearn.metrics import mean_squared_error, mean_absolute_error

print("MAE {:.3}, MSE {:.3}".format(
    mean_absolute_error(y_test, predicted),
    mean_squared_error(y_test, predicted)))

for i, horizon in enumerate(horizons):
    print("MAE {:.3f}, MSE {:.3f} for horizon {}".format(
        mean_absolute_error(y_test[:,i], predicted[:,i]),
        mean_squared_error(y_test[:,i], predicted[:,i]),
        horizon))

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

inverse_scale = invert_all_prediction_scaled(scaler, predicted, scaled.columns, horizons)

predicted_signal = invert_all_prediction_differences(
    inverse_scale,
    features['wspd'][sequence_offset + row_split:])

for i, horizon in enumerate(horizons):
    a = features['wspd'][sequence_offset + row_split + horizon:]
    p = predicted_signal[:-horizon, i]

    print("Real scale predictions at horizon {:>2} has MAE {:.3f}, MSE {:.3f}, RMSE {:.3f}".format(
        horizon,
        mean_absolute_error(a, p),
        mean_squared_error(a, p),
        sqrt(mean_squared_error(a, p))))


def evaluate_persistence_forecast(test, horizons):
    for i, horizon in enumerate(horizons):
        a = test[horizon:]
        p = test[:-horizon]

        print("Persistence Method prediction at horizon {:>2} has MAE {:.3f}, MSE {:.3f}, RMSE {:.3f}".format(
            horizon,
            mean_absolute_error(a, p),
            mean_squared_error(a, p),
            sqrt(mean_squared_error(a, p))))


evaluate_persistence_forecast(
    features['wspd'][sequence_offset + row_split:].values,  # ensure we have same test set
    horizons)

import matplotlib.pyplot as plt

plot_samples = 800
max_horizon = horizons[-1]
plots = len(horizons)

fig = plt.figure(figsize=(14, 5 * plots))
fig.suptitle("Model Prediction at each Horizon")

for i, horizon in enumerate(horizons):
    plt.subplot(plots, 1, i + 1)

    len_adjust = max_horizon - horizon  # ensure all have same lenght

    real = features['wspd'][sequence_offset + row_split + horizon + len_adjust:].values
    pred = predicted_signal[len_adjust:-horizon, i]

    plt.plot(real[:plot_samples], label='observed')
    plt.plot(pred[:plot_samples], label='predicted')
    plt.title("Prediction for {} Hour Horizon".format(horizon))
    plt.xlabel("Hour")
    plt.ylabel("Wind Speed (m/sec)")
    plt.legend()
    plt.tight_layout()

fig.tight_layout()
plt.subplots_adjust(top=0.95)