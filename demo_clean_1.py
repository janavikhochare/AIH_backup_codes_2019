import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
import matplotlib.pylab as plt
import seaborn as sns

warnings.filterwarnings('ignore')


#accuracy is 82.25%

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

values = np.array(df['lvl12_1'])
df['lvl12_2'] = pd.DataFrame(lvl12_2)

# One Hot Encoding
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



#code

#df = df[np.isfinite(df['wspd'])]

# is_not_2016 = df['year'] != 2017
# train = df[is_not_2016]
#
# is_2016 = df['year'] == 2017
# test = df[is_2016]

#is_not_2017 = df.head(18490)
train = df[:18490]

#is_2017 = df.tail (10)
test = df[18490:]

print(df.shape)
print(train.shape)
print(test.shape)

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

print('train:\n', train.head())
print('test:\n', test.head())
print('X_train:\n', X_train.head())
print('Y_train:\n', Y_train.head())
print('X_test:\n', X_test.head())
print('Y_test:\n', Y_test.head())

numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.dtypes)

corr =numeric_features.corr()
print(corr['wspd'].sort_values(ascending=False))
#print(corr)
#correlation matrix
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=1, square=True)
plt.show()


def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


import lightgbm as lgb
fit_params={"early_stopping_rounds":30,
            "eval_metric" : 'auc',
            "eval_set" : [(X_test,Y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50),
             'min_child_samples': sp_randint(100, 500),
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8),
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

#This parameter defines the number of HP points to be tested
n_HP_points_to_test = 100

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test,
    n_iter=n_HP_points_to_test,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)
opt_parameters = {'colsample_bytree': 0.9234, 'min_child_samples': 399, 'min_child_weight': 0.1, 'num_leaves': 13, 'reg_alpha': 2, 'reg_lambda': 5, 'subsample': 0.855}


clf_sw = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_sw.set_params(**opt_parameters)

gs_sample_weight = GridSearchCV(estimator=clf_sw,
                                param_grid={'scale_pos_weight':[1,2,6,12]},
                                scoring='roc_auc',
                                cv=5,
                                refit=True,
                                verbose=True)

gs_sample_weight.fit(X_train, Y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, gs_sample_weight.best_params_))


print("Valid+-Std     Train  :   Parameters")
for i in np.argsort(gs_sample_weight.cv_results_['mean_test_score'])[-5:]:
    print('{1:.3f}+-{3:.3f}     {2:.3f}   :  {0}'.format(gs_sample_weight.cv_results_['params'][i],
                                    gs_sample_weight.cv_results_['mean_test_score'][i],
                                    gs_sample_weight.cv_results_['mean_train_score'][i],
                                    gs_sample_weight.cv_results_['std_test_score'][i]))

#Configure from the HP optimisation
#clf_final = lgb.LGBMClassifier(**gs.best_estimator_.get_params())

#Configure locally from hardcoded values
clf_final = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_final.set_params(**opt_parameters)

#Train the final model with learning rate decay
clf_final.fit(X_train, Y_train, **fit_params, callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])

#
#
# probabilities = clf_final.predict_proba()
# submission = pd.DataFrame({
#     'SK_ID_CURR': df['hour'],
#     'TARGET':     [ row[1] for row in probabilities]
# })
# submission.to_csv("submission.csv", index=False)