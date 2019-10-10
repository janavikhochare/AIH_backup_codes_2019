import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("INM00042103.csv")
df= df[4082:]
print(df.head())

df= df[['wdir','wspd','gph','npv','day','month','year','hour']]
print(df.head())

df_new = df['hour']

df_new_onehot = df_new.copy()

c = ['Address', 'CommandResponse', 'ControlMode', 'ControlScheme', 'FunctionCode',
     'InvalidDataLength', 'InvalidFunctionCode']
c=[ '0','6','12','18','11','17','5','7', '9',  '1' , '8', '15',  '2', '13' ,'10']
# indicator cols
for i in range(len(c)):
    a1 = pd.get_dummies(df_new_onehot, c[i])
    #,'lvl12_1_0','lvl12_1_1','lvl12_1_2','lvl12_1_3','lvl12_2_0','lvl12_2_1']]
print(a1.head())


df= df.join(a1)
df=df.drop('hour',axis=1)
cor = df.corr()
#print('\n\n', cor)

# Correlation with output variable
cor_target = abs(cor['wspd'])
print(cor_target)

print("\n\nCorrelation:\n", cor_target, '\n')

ld = df['wspd'].tolist()

ld_7 = []

for i in range(len(ld)):
    if i >= 1 :
        ld_7.append(ld[i-1])
    else:
        ld_7.append(np.mean(ld[:1]))



# cmnts = {}
# for i, row in df.iterrows():
#     while True:
#         try:
#             if row['hour']:
#                 cmnts[row['Name']].append(row['Use_Case'])
#
#             else:
#                 cmnts[row['Name']].append('n/a')
#
#             break
#
#         except KeyError:
#             cmnts[row['Name']] = []
#
# df.drop_duplicates('Name', inplace=True)
# df['Use_Case'] = ['; '.join(v) for v in cmnts.values()]