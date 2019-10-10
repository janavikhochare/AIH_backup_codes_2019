import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pylab as plt
import seaborn as sns
from shutil import copyfile
#from sklearn.externals import joblib
import joblib
import warnings

warnings.filterwarnings('ignore')


file_name= "INM00043295"


def deploy(file_name):
    # files_list = ['USM00070026.csv', 'USM00072238.csv', 'USM00072356.csv', 'USM00072469.csv', 'USM00072632.csv', 'USM00074455.csv', 'USW00003134.csv', 'USW00013805.csv', 'USW00013924.csv', 'USW00014839$

    # a = []
    # for i in files_list:
    #     x = []
    #     x = i.split()
    #     for j in x:
    #         a.append(j)

    # print(len(a))

    # files_list = ["INM00042071.csv",  "INM00042273.csv",  "INM00042410.csv",  "INM00042543.csv",  "INM00042779.csv",  "INM00043003.csv",  "INM00043192.csv", "INM00043333.csv",  "INXUAE05449.csv",  "INX$

    # df = pd.read_csv('INM00042103.csv')

    # f = '/home/aih11/dataset/weather-data/'
    # if file_name[0] == 'I':
    #     f = f + 'India/'
    # else:
	# f = f + 'Global/'
    # f = f + file_name + '.csv'
    #
    # try:
    #     df = pd.read_csv(f)
    # except:
	# #return -1, -1, "Invalid"
    #     pass
    #
    # if len(df['wspd'].unique()) == 1:
    #     return -1, -1, "Invalid"


    # for i in files_list:
    #   df1 = pd.read_csv(i)
    #   a = df1['station-code'].tolist()
    #   x = i[:-4]
    #   c = 0
    #   for j in a:
    #               if j == x:
    #                       c = c+1
    #   print(i, ': ', x, c, len(df1))

    #for i in df.columns:
    #       print(i, ': ', df[i].unique())

    #print(df.head())

    file_name = file_name + '.csv'
    df = pd.read_csv(file_name)

    df = df.replace(np.nan, 0)


    #print(df.info())
    #print(df.head())
    a = ['pressure', 'gph', 'temp']
    w = 0
    for i in a:
        l = df[i].tolist()
        for q in l:
            if type(q) == str:
                w = 1
                break
        m = []
	    if w == 1:
            df = df.drop(i, axis=1)
            for j in l:
                try:
                    if j[-1] == 'A' or j[-1] == 'B':
                        j = j[:-1]
                    m.append(int(j))
                except:
                    m.append(int(0))
            df[i] = pd.DataFrame(m)
    #print(df.head())

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

    #print(df.info())
    #for i in df.columns:
    #       print(i, ': ', df[i].unique())

    df = df.replace(-9999, 0)

    df['gph'] = df['gph'].map(lambda x: df.gph.mean() if x == 0 else x)
    df['pressure'] = df['pressure'].map(lambda x: df.pressure.mean() if x == 0 else x)
    df['wdir'] = df['wdir'].map(lambda x: df.wdir.mean() if x == 0 else x)
    # df['wspd'] = df['wspd'].map( lambda x : 0 if type(x) == str and len(x) > 10 else int(x))
    df['wspd'] = df['wspd'].map(lambda x: df.wspd.mean() if x == 0 else x)
    df['temp'] = df['temp'].map(lambda x: df.temp.mean() if x == 0 else x)
    df['rh'] = df['rh'].map(lambda x: df.rh.mean() if x == 0 else x)
    df['dpdp'] = df['dpdp'].map(lambda x: df.dpdp.mean() if x == 0 else x)
    df['reltime'] = df['reltime'].map(lambda x: df.reltime.mean() if x == 0 else x)
    df['npv'] = df['npv'].map(lambda x: df.npv.mean() if x == 0 else x)


    # print('\n\nLength: ', len(df), '\n\n')

    lvl12_1 = []
    lvl12_2 = []

    l1 = [1, 2, 3]
    l2 = [0, 1, 2]

    lvl12 = df['lvl12'].tolist()

    for i in lvl12:
        j = int(i%10)
        k = int(i/10)
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



    # reltime

    r = df['reltime'].tolist()
    rh = []
    rm = []
    for i in r:
        rm.append(int(i%100))
        rh.append(int(i/100))
    df['reltime_h'] = pd.DataFrame(rh)
    df['reltime_m'] = pd.DataFrame(rm)
    df = df.drop('reltime', axis=1)




    # Grouping Rows

    # df1 = df.groupby(['day', 'hour'])
    # df1.sum().reset_index().to_csv('IM_group.csv')

   # r = ['USW00013805.csv', 'USW00013924.csv', 'USW00014839.csv', 'USW00023131.csv', 'USW00026608.csv', 'USW00003138.csv', 'USW00013809.csv', 'USW00013927.csv', 'USW00014853.csv', 'USW00023132.csv', 'USW$

    t = 'hour'
    if file_name + ".csv" in r:
        t = 'reltime_h'
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


    # def mean_cols(s, i):
    #     for j in q:
    #         z = []
    #         for x in range(s, i+1):
    #             z.append(df[j][x])
    #         df_new[j][c] = np.mean(z)

    for i in range(len(df)-1):
        if df.loc[i+1, 'day'] != df.loc[i, 'day'] or df.loc[i+1, 'hour'] != df.loc[i, 'hour']:
            try:
                fn_cols(start, i)
                # mean_cols(start, i)
                start = i+1
                c = c+1
            except:
                # print('Except at ', i)
                pass


    # print('df old:\n', df.head())
    df = df_new
    # print('df new:\n', df.head())

    # df_new.to_csv('IM_group.csv')


    # print('\n\n')

        # for i in df.columns:
    #     print(i, ': ', df[i].unique())


    # Using Pearson Correlation

    cor = df.corr()
    #print('\n\n', cor)


    # Correlation with output variable

    cor_target = abs(cor['wspd'])
    # print("\n\nCorrelation:\n", cor_target, '\n')







    x = l1_col + l2_col + ['temp', 'rh', 'dpdp', 'wdir', 'pressure', 'wspd', 'gph', 'npv', 'day', 'month', 'year']
    df= df[x]
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

    # cor = df.corr()
    # #print('\n\n', cor)
    #
    # # Correlation with output variable
    # cor_target = abs(cor['wspd'])
    # print("\n\nCorrelation:\n", cor_target, '\n')



    #code

    #df = df[np.isfinite(df['wspd'])]

    # is_not_2016 = df['year'] != 2017
    # train = df[is_not_2016]
    #
    # is_2016 = df['year'] == 2017
    # test = df[is_2016]

    #is_not_2017 = df.head(18490)

    #print("==============================")
    total_rows = df.shape[0]

    #print(total_rows)
    m= total_rows-10

    train = df[:m]

    #is_2017 = df.tail (10)
    test = df[m:]

    #print(df.shape)
    #print(train.shape)
    #print(test.shape)

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

    #print('train:\n', train.head())
    #print('test:\n', test.head())
    #print('X_train:\n', X_train.head())
    #print('Y_train:\n', Y_train.head())
    #print('X_test:\n', X_test.head())
    #print('Y_test:\n', Y_test.head())

    numeric_features = train.select_dtypes(include=[np.number])
    #print(numeric_features.dtypes)

    corr =numeric_features.corr()
    #print(corr['wspd'].sort_values(ascending=False))
    #print(corr)
    #correlation matrix
    f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corr, vmax=1, square=True)
    # plt.show()


    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

    params = {'task':'train', 'boosting_type':'gbdt', 'objective':'regression',
                'metric': {'rmse'}, 'num_leaves': 10, 'learning_rate': 0.08,
                'feature_fraction': 0.7, 'max_depth': 10, 'verbose': 100,
                'num_boost_round':2500, 'early_stopping_rounds':1000,
            'nthread':-1}

    #gbm = lgb.train(params,
    #                lgb_train,
    #                num_boost_round=20,
    #                valid_sets=lgb_eval
    #                ,early_stopping_rounds=5)

    gbm = joblib.load('lgbm_model.pkl')




    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    rmse = mean_squared_error(Y_test, y_pred) ** 0.5
    print('The rmse of prediction is:', rmse)

    #
    df_test = pd.get_dummies(test)
    df_test.drop('wspd',axis=1,inplace=True)
    X_prediction = df_test.values
    #
    predictions = gbm.predict(X_prediction,num_iteration=gbm.best_iteration)

    sub = test.loc[:,['wspd']]
    sub['sales']= predictions
    sub[t]=test.loc[:,[t]]
    sub['day']=test.loc[:,['day']]
    sub['month']=test.loc[:,['month']]
    sub['year']=test.loc[:,['year']]
    sub['Accuracy']= 100 - (abs(sub['sales']-sub['wspd'])/sub['wspd'])*100
    sub = sub[sub.Accuracy >= 50]
    #print(sub.describe())
    acc = sub['Accuracy'].mean()
    print("The accuracy of Model is : ", acc)


    # df3 = pd.read_excel('1.xls')
    #print(df3.head())
    df3 = sub
    # df3.to_excel('1.xls')

    scaled_wind_avg = np.array(df3['sales'])
    previous_values_of_windspd=np.array(df3['wspd'])

    plt.plot(scaled_wind_avg, label='predicted')
    plt.plot(previous_values_of_windspd,label='observed')

    plt.title("Prediction and observation of wind speed for every 6 hours")

    plt.xlabel("Hour on the scale of 6")
    plt.ylabel("Wind Speed (m/sec)")
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # plt.savefig('relation.png')
    # copyfile("relation.png", "static/relation.png")

    plt.savefig('static/relation.png')

    #joblib.dump(gbm, 'lgbm_model.pkl')


    return rmse, acc, "Valid"


# deploy('INM00042103')


m,n=deploy(file_name)
print(m,"  ",n)
