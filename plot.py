import pandas as pd
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import numpy as np
import lightgbm as lgb
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop

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



file_name= "INM00042103"

file_name = file_name + '.csv'
df = pd.read_csv(file_name)
#df = df.tail(300)
#df.reset_index(inplace=True)


#df=df['wspd']
# scaled_wind_avg = np.array(df['wspd'])
# previous_values_of_windspd = np.array(df[''])
#
# plt.plot(scaled_wind_avg, label='predicted')
# plt.plot(previous_values_of_windspd, label='observed')
#
# plt.title("Prediction and observation of wind speed for every 6 hours")
#
# plt.xlabel("Hour on the scale of 6")
# plt.ylabel("Wind Speed (m/sec)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# df.plot(figsize=(20,10), linewidth=5, fontsize=20)
# plt.xlabel('hour', fontsize=20)
# plt.show()

print("std withod filling")
print(df.std())

cor = df.corr(method= 'kendall')
cor_target = abs(cor['wspd'])
print(cor_target)

# print('Mean:', np.mean(df))
# print('Standard Deviation:', np.std(df))
#
#
# df['wspd'] = df['wspd'].map(lambda x: df.wspd.mean() if x == 0 else x)
#
# print('Mean after filling values:', np.mean(df))
# print('Standard Deviation after filling values:', np.std(df))
print("===============================================================================================")
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

print(df.std())

cor = df.corr(method="kendall")
cor_target = abs(cor['wspd'])
print(cor_target)

