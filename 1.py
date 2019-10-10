import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
import seaborn as sns


df =pd.read_csv("INM00042111.csv")
df= df.replace(-9999,0)
df = df.replace(np.nan, 0)
df['gph'] = df['gph'].map( lambda x : df.gph.mean() if x == 0 else x)
df['pressure'] = df['pressure'].map( lambda x : df.pressure.mean() if x == 0 else x)
df['wdir'] = df['wdir'].map( lambda x : df.wdir.mean() if x == 0 else x)
df['wspd'] = df['wspd'].map( lambda x : df.wspd.mean() if x == 0 else x)
df['temp'] = df['temp'].map( lambda x : df.temp.mean() if x == 0 else x)
df['rh'] = df['rh'].map( lambda x : df.rh.mean() if x == 0 else x)
df['dpdp'] = df['dpdp'].map( lambda x : df.dpdp.mean() if x == 0 else x)
df['reltime'] = df['reltime'].map( lambda x : df.reltime.mean() if x == 0 else x)
df['npv'] = df['npv'].map( lambda x : df.npv.mean() if x == 0 else x)

# print('\n\nLength: ', len(df), '\n\n')

lvl12_1 = []
lvl12_2 = []

lvl12 = df['lvl12'].tolist()

for i in lvl12:
        lvl12_2.append(int(i%10))
        lvl12_1.append(int(i/10))


# print(lvl12_1[:10])
# print(lvl12_2[:10])

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
df1.columns = ['lvl12_1_0', 'lvl12_1_1', 'lvl12_1_2', 'lvl12_1_3']
# print(df1.head())

values = np.array(df['lvl12_2'])
label_encoder = LabelEncoder()


values = np.array(df['lvl12_1'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

df1 = pd.DataFrame(onehot_encoded)
df1.columns = ['lvl12_1_0', 'lvl12_1_1', 'lvl12_1_2', 'lvl12_1_3']
# print(df1.head())

values = np.array(df['lvl12_2'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

df2 = pd.DataFrame(onehot_encoded)
df2.columns = ['lvl12_2_0', 'lvl12_2_1']
# print(df2.head())

df = df.drop(['lvl12_1', 'lvl12_2'], axis=1)
df = df.join(df1)
df = df.join(df2)

for i in df.columns:
        print(i, ': ', df[i].unique())

# Using Pearson Correlation
cor = df.corr()
# Correlation with output variable
cor_target = abs(cor['wspd'])
print(cor_target)

