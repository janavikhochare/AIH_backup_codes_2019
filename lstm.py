import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import *
from sklearn.metrics import mean_squared_error
import statistics 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pylab as plt
import seaborn as sns
from shutil import copyfile
from sklearn.externals import joblib
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, pearsonr

file_name= "INM00042103"

def deploy(file_name):
    
    file_name = file_name + '.csv'
    df = pd.read_csv(file_name)
    
    df = df.replace(np.nan, 0)
    df=df.tail(30000)
    df.reset_index(inplace=True)

    # print(df.info())

    a = ['pressure', 'gph', 'temp']

    for i in a:
        l = df[i].tolist()
        m = []
        if type(l[0]) == str:
                df = df.drop(i, axis=1)
                for j in l:
                    try:
                        if j[-1] == 'A':
                            j = j[:-1]
                        m.append(int(j))
                    except:
                        m.append(int(0))
                df[i] = pd.DataFrame(m)


    m = df['wspd'].tolist()
    n = []
    df = df.drop('wspd', axis=1)
    c = 0
    for i in m:
        try:
            n.append(int(i))
        except:
            c = c+1
            n.append(0)

    #print("\n\nCount: ", c, "\n\n")
    df['wspd'] = pd.DataFrame(n)

    # print(df.info())


    df = df.replace(-9999, np.nan)

    #df['gph'] = df['gph'].map(lambda x: df.gph.mean() if x == 0 else x)
    #df['pressure'] = df['pressure'].map(
     #   lambda x: df.pressure.mean() if x == 0 else x)
    df['wdir'] = df['wdir'].map(lambda x: df.wdir.mean() if x == np.nan else x)
    # df['wspd'] = df['wspd'].map( lambda x : 0 if type(x) == str and len(x) > 10 else int(x))
    df['wspd'] = df['wspd'].map(lambda x: df.wspd.mean() if x == np.nan else x)
    #df['temp'] = df['temp'].map(lambda x: df.temp.mean() if x == 0 else x)
    df['rh'] = df['rh'].map(lambda x: df.rh.mean() if x == np.nan else x)
    df['dpdp'] = df['dpdp'].map(lambda x: df.dpdp.mean() if x == np.nan else x)
    df['reltime'] = df['reltime'].map(lambda x: df.reltime.mean() if x == np.nan else x)
    df['npv'] = df['npv'].map(lambda x: df.npv.mean() if x == np.nan else x)


    i=0
    while (i<30):
        i=i+1
        df['pressure'].fillna(method='backfill', inplace=True)
        df['gph'].fillna(method='backfill', inplace=True)
        df['npv'].fillna(method='backfill', inplace=True)

    df = df.replace(np.nan, 0)

    lvl12_1 = []
    lvl12_2 = []

    l1 = [1, 2, 3]
    l2 = [0, 1, 2]

    lvl12 = df['lvl12'].tolist()

    for i in lvl12:
        j = int(i % 10)
        k = int(i / 10)
        if j in l2:
            lvl12_2.append(int(j))
        else:
            lvl12_2.append(0)

        if k in l1:
            lvl12_1.append(int(k))
        else:
            lvl12_1.append(2)

    df = df.drop(['lvl12'], axis=1)
    df['lvl12_1'] = pd.DataFrame(lvl12_1)
    df['lvl12_2'] = pd.DataFrame(lvl12_2)

    l1_set = list(set(lvl12_1))
    l2_set = list(set(lvl12_2))

    l1_set.sort()
    l2_set.sort()

    l1_col = []
    l2_col = []

    s1 = 'lvl12_1_'
    s2 = 'lvl12_2_'

    for i in l1_set:
        l1_col.append(s1 + str(i))

    for i in l2_set:
        l2_col.append(s2 + str(i))

    # print('l1_col and l2_col:')
    # print(l1_col)
    # print(l2_col)

    # print('df:\n', df.head())
    # print('lvl12_1: ', df['lvl12_1'].unique())
    # print('lvl12_2: ', df['lvl12_2'].unique())

    # print("LENGTHS:\n\n")
    # print('df:\n', df.shape)
    # print('lvl12_1: ', len(df['lvl12_1']))
    # print('lvl12_2: ', len(df['lvl12_2']))

    # One Hot Encoding

    values = np.array(df['lvl12_1'])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    df1 = pd.DataFrame(onehot_encoded)
    # print('df1:\n', df1.head())
    df1.columns = l1_col  # ['lvl12_1_0', 'lvl12_1_1', 'lvl12_1_2', 'lvl12_1_3']
    # print(df1.head())

    values = np.array(df['lvl12_2'])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    df2 = pd.DataFrame(onehot_encoded)
    df2.columns = l2_col  # ['lvl12_2_0', 'lvl12_2_1']
    # print(df2.head())

    df = df.drop(['lvl12_1', 'lvl12_2'], axis=1)
    df = df.join(df1)
    df = df.join(df2)

    r = df['reltime'].tolist()
    rh = []
    rm = []
    for i in r:
        rm.append(int(i%100))
        rh.append(int(i/100))
    df['reltime_h'] = pd.DataFrame(rh)
    df['reltime_m'] = pd.DataFrame(rm)
    df = df.drop('reltime', axis=1)
    
    p = ['station-code'] + l1_col + l2_col
    q = list(set(df.columns) - set(p))
    df_new = pd.DataFrame(columns=df.columns)
    start = 0
    c = 0

    def mode(array):
        most = max(list(map(array.count, array)))
        l = list(set(filter(lambda x: array.count(x) == most, array)))
        return l[0]

    def fn_cols(s, i):
        a = []
        for j in df.columns:
            z = []
            for x in range(s, i+1):
                z.append(df.loc[x, j])
            if j in p:
                a.append(mode(z))
            else:
                a.append(np.mean(z))
        df_new.loc[c] = a


   

    for i in range( len(df)-1):
        if df.loc[i+1, 'day'] != df.loc[i, 'day'] or df.loc[i+1, 'hour'] != df.loc[i, 'hour']:
            try:
                fn_cols(start, i)
                
                start = i+1
                c = c+1
            except:
                
                pass


    
    df = df_new
    

    cor = df.corr()
    

    cor_target = abs(cor['wspd'])
    #df= df[['wdir', 'pressure', 'wspd', 'gph', 'npv', 'day', 'hour', 'month', 'year']]+ l1_col+l2_col
    x = l1_col + l2_col + ['temp', 'rh', 'dpdp', 'wdir', 'pressure', 'wspd', 'gph', 'npv', 'day', 'month', 'year']
    # x.append(t)
    df = df[x]

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



   
    total_rows = df.shape[0]

    print(total_rows)
    m= total_rows-10

    train = df[:m]

    
    test = df[m:]

    print(df.shape)
    print(train.shape)
    print(test.shape)

   


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
    

    corr =numeric_features.corr()
    print(corr['wspd'].sort_values(ascending=False))

    print('\n\n')
    print("Corelation ")

    wspd = np.array(df['wspd'])


    for i in df.columns:
        # print(i, ': ', df[i].unique())
        j = i
        k = np.array(df[i])
        coef_s, p = spearmanr(k, wspd)
        coef_p, p = pearsonr(k, wspd)
        print("\n\n {}: S: {} P: {} \n".format(j, coef_s, coef_p))

    print("\n\n")

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0, 1))

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)

    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    X_test=np.array(X_test)
    Y_test=np.array(Y_test)

    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1] , 1))
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))

    
    f, ax = plt.subplots(figsize=(12, 9))
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=( X_train.shape[1],1 )))
    #model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(LSTM(units= 64, return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(LSTM(units=128))
    #model.add(Dropout(0.2))
    model.add(Dense(units = 1, activation="linear"))
    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
    model.fit(X_train, Y_train, epochs = 150, batch_size = 32)
    model.save('lstm.h5')

    # layers = [X_train.shape[2], 20, 15, 20, Y_train.shape[1]],
    #
    # model = Sequential()
    #
    # model.add(LSTM( input_shape=(X_train.shape[1],1 ),return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(layers[2], return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(layers[3], return_sequences=False))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(units=1, activation="linear"))
    #
    # model.compile(loss="mse", optimizer='rmsprop')
    # #model.fit(X_train, Y_train, epochs = 20, batch_size = 32)
    #
    # print(model.summary())
    # model.fit(X_train, Y_train, epochs = 40, batch_size = 32)
    # model.save('lstm.h5')


    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)


    
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    print('RMSE: %.3f' % rmse)




    sub = test.loc[:,['wspd']]
    sub['sales']= predictions
    #sub['hour']=test.loc[:,['hour']]
    sub['day']=test.loc[:,['day']]
    sub['month']=test.loc[:,['month']]
    sub['year']=test.loc[:,['year']]
    sub['Accuracy']= 100 - (abs(sub['sales']-sub['wspd'])/sub['wspd'])*100
    #print(sub.describe())
    sub = sub[sub.Accuracy >= 50]
    acc = sub['Accuracy'].mean()
    print("The accuracy of Model is : ", acc)
    #
    #df3 = pd.read_excel('1.xls')
    #print(df3.head())
    df3 = sub
    df3.to_excel('1.xls')

    scaled_wind_avg = np.array(df3['sales'])
    previous_values_of_windspd=np.array(df3['wspd'])

    plt.plot(scaled_wind_avg, label='predicted')
    plt.plot(previous_values_of_windspd,label='observed')

    plt.title("Prediction and observation of wind speed for every 6 hours")

    plt.xlabel("Hour on the scale of 6")
    plt.ylabel("Wind Speed (m/sec)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
m,n=deploy(file_name)
print(m,"  ",n)