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
from missingpy import MissForest

warnings.filterwarnings('ignore')

file_name = "INM00042103"


def deploy(file_name):
    file_name = file_name + '.csv'
    df = pd.read_csv(file_name)
    df=df.tail(30000)
    df = df.replace(to_replace = -9999, value =np.nan)
    #
    # i=0
    # while (i<30):
    #     i=i+1
    #     df['pressure'].fillna(method='backfill', inplace=True)
    #     df['gph'].fillna(method='backfill', inplace=True)
    # #
    #
    # df= df[['pressure','temp','gph']]
    # print(df.head(10))
    # df.replace(np.nan,0)


    # df1 = pd.read_excel('/Users/jashrathod/Desktop/')
    df_new= pd.DataFrame()
    df_new['wdir_new'] = df['wdir']
    df_new['gph']=df['gph']
    df_new.reset_index(inplace=True)
    print(df_new.head())
    #df_new = df.replace(-9999, np.nan)
    imputer = MissForest()
    df_new = imputer.fit_transform(df_new)
    #print(df_new.head())
    df_new = pd.DataFrame(df_new)
    df_new.rename(columns={0:'a',1:'b',2:'c'})
    print(df_new.columns)
    print(df_new.head())
    df= df.join(df_new)

    df_new.to_excel("1filmiss.xls")
    # for i in df.columns:
    #     try:
    #         ab = df[i].loc[1000:]
    #         ab.plot(style='.')
    #         plt.title(i)
    #         #print("=============")
    #         plt.show()
    #     except:
    #         #print("==============")
    #         pass
    #
    # return 0
m= deploy(file_name)