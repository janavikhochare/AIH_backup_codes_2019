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
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import shape
import matplotlib.ticker as ticker



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
df12=df[['wspd','wdir','gph','month','before_6hr' ,'before_6hr_wdir','before_12hr','before_12hr_wdir',
     'before_18hr','before_24hr','before_30hr','before_36hr' ,'before_42hr','before_48hr']]

rnn_unit = 10  # hidden layer units
input_size = 13
output_size = 1
lr = 0.0006  # learning rate

batch_size = 14
time_step = 6

train_begin = 0
train_end = 18000
test_begin = 18000
test_len = 180
iter_time = 50

# RNN output node weights and biases
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}

"""
Preparing the training data   
"""


# Get train data function: load training data for LSTM
# Input: batch_size, time_step, train_begin, train_end
# Output: batch_index, train_x, train_y

def get_train_data(batch_size, time_step, train_begin, train_end):
    batch_index = []

    # normalize the data
    scaler_for_x = MinMaxScaler(feature_range=(0, 1))

    scaled_x_data = scaler_for_x.fit_transform(df12)

    scaler_for_y = MinMaxScaler(feature_range=(0, 1))
    scaled_y_data = scaler_for_y.fit_transform(df12["wspd"])
    scaled_y_data= scaled_y_data.reshape(-1,1)
    # get train data
    normalized_train_data = scaled_x_data[train_begin:train_end]
    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :5]
        y = normalized_train_data[i:i + time_step, 0, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


"""
Preparing the testing data   
"""


# Get test data function: load testing data for LSTM
# Input: time_step, test_begin, test_len
# Output: test_x, test_y, scaler_for_x, scaler_for_y

def get_test_data(time_step, test_begin, test_len):
    # normalize the data
    scaler_for_x = MinMaxScaler(feature_range=(0, 1))
    scaler_for_y = MinMaxScaler(feature_range=(0, 1))
    scaled_x_data = scaler_for_x.fit_transform(df)
    scaled_y_data = scaler_for_y.fit_transform(df[:, 0].reshape(-1, 1))

    # get test data
    size = test_len // time_step
    normalized_test_data = scaled_x_data[test_begin: (test_begin + test_len)]
    test_y = normalized_test_data[:, 0]
    test_x = []

    for i in range(size):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :5]
        test_x.append(x.tolist())
    return test_x, test_y, scaler_for_x, scaler_for_y


"""
Creating the LSTM network   
"""


# LSTM function: definition of recurrent neural network
# Input: X
# Output: pred, final_states

def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']

    # reshape to (batch_size * time_step, input_size)
    input = tf.reshape(X, [-1, input_size])  # turn tensor to 3D-Array as the input of hidden layer
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])

    # create an LSTM cell to be unrolled
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    # cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
    # cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)

    # At each time step, reinitialising the hidden state
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # generate prediction
    # create a dynamic RNN object in TensorFlow.
    # This object will dynamically perform the unrolling of the LSTM cell over each time step
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)

    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    ## Get the last output
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


"""
Training the LSTM Model and Making Predictions   
"""


# train_lstm function: train the LSTM model, make predictions, and calculate the error of predication
# Input: batch_size, time_step, train_begin, train_end, test_begin, iter_time, test_len
# Output: test_y, test_predict, loss_list, rmse, mae

def train_lstm(batch_size, time_step, train_begin, train_end, test_begin, iter_time, test_len):
    # set up the state storage / extraction
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)

    print("Training parameters:***************************************************")
    print("batch size: ", batch_size)
    print("Number of batches: ", len(batch_index))
    print("Shape of training samples:", shape(train_x))
    print("Shape of training labels:", shape(train_y))

    pred, _ = lstm(X)

    ## Loss and optimizer
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    loss_list = []
    print("Training begins: *****************************************************")

    ## Training step optimization
    """
    The loss are accumulated to monitor the progress of the training. 
    20 iteration is generally enough to achieve an acceptable accuracy.
    """

    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        # repeat training 50 times
        for epoch in range(iter_time):
            for step in range(len(batch_index) - 2):
                ## Calculate batch loss
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: np.array(
                                                                     train_y[batch_index[step]:batch_index[step + 1]],
                                                                     dtype=float).reshape(batch_size, time_step, 1)})
                loss_list.append(loss_)

            # Show loss every 5 iterations
            if epoch % 5 == 0:
                print("Epoch:", epoch, " loss:", loss_)

                # if step%100==0:
                # print('Epoch:', epoch, 'steps: ', step,  'loss:', loss_)
        print("Training Optimization Finished! ***************************************")

        """Testing the model"""
        print("Prediction Begins: ****************************************************")
        test_x, test_y, scaler_for_x, scaler_for_y = get_test_data(time_step, test_begin, test_len)
        print("Shape of testing samples:", shape(test_x))

        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        # test_predict = scaler_for_y.inverse_transform(np.array(test_predict).reshape(-1,1))
        # test_y = scaler_for_y.inverse_transform(np.array(test_y).reshape(-1,1))

        test_y = np.array(test_y).reshape(-1, 1)
        test_predict = np.array(test_predict).reshape(-1, 1)
        print("Shape of testing lables:", shape(test_predict))
        test_predict = scaler_for_y.inverse_transform(test_predict).reshape(-1, 1)
        test_y = scaler_for_y.inverse_transform(test_y).reshape(-1, 1)

        # calculate the error of predication
        rmse = np.sqrt(mean_squared_error(test_predict, test_y))
        mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
        print("Mean absolute error:", "{:.3f}".format(mae),
              "Root mean squared error:", "{:.3f}".format(rmse))
    return test_y, test_predict, loss_list, rmse, mae


test_y, test_predict, loss_list, rmse, mae = train_lstm(batch_size, time_step, train_begin, train_end, test_begin,
                                                        iter_time, test_len)