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
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

file_name= "INM00042103"

def deploy(file_name):
    # files_list = ['USM00070026.csv', 'USM00072238.csv', 'USM00072356.csv', 'USM00072469.csv', 'USM00072632.csv', 'USM00074455.csv', 'USW00003134.csv', 'USW00013805.csv', 'USW00013924.csv', 'USW00014839.csv', 'USW00023131.csv', 'USW00026608.csv', 'USM00070027.csv', 'USM00072239.csv', 'USM00072357.csv', 'USM00072475.csv', 'USM00072634.csv', 'USM00074479.csv', 'USW00003138.csv', 'USW00013809.csv', 'USW00013927.csv', 'USW00014853.csv', 'USW00023132.csv', 'USW00026618.csv', 'USM00070086.csv', 'USM00072240.csv', 'USM00072363.csv', 'USM00072476.csv', 'USM00072636.csv', 'USM00074482.csv', 'USW00003143.csv', 'USW00013810.csv', 'USW00013928.csv', 'USW00014884.csv', 'USW00023136.csv', 'USW00026635.csv', 'USM00070133.csv', 'USM00072243.csv', 'USM00072364.csv', 'USM00072480.csv', 'USM00072637.csv', 'USM00074486.csv', 'USW00003835.csv', 'USW00013811.csv', 'USW00013930.csv', 'USW00014892.csv', 'USW00023139.csv', 'USW00026636.csv', 'USM00070194.csv', 'USM00072244.csv', 'USM00072365.csv', 'USM00072481.csv', 'USM00072639.csv', 'USM00074494.csv', 'USW00003955.csv', 'USW00013812.csv', 'USW00013933.csv', 'USW00014896.csv', 'USW00023140.csv', 'USW00045703.csv', 'USM00070204.csv', 'USM00072247.csv', 'USM00072374.csv', 'USM00072483.csv', 'USM00072641.csv', 'USM00074505.csv', 'USW00012802.csv', 'USW00013814.csv', 'USW00013935.csv', 'USW00014903.csv', 'USW00023151.csv', 'USW00073807.csv', 'USM00070219.csv', 'USM00072248.csv', 'USM00072376.csv', 'USM00072485.csv', 'USM00072645.csv', 'USM00074531.csv', 'USW00012803.csv', 'USW00013815.csv', 'USW00013936.csv', 'USW00014920.csv', 'USW00023152.csv', 'USW00073808.csv', 'USM00070231.csv', 'USM00072249.csv', 'USM00072381.csv', 'USM00072486.csv', 'USM00072649.csv', 'USM00074544.csv', 'USW00012805.csv', 'USW00013817.csv', 'USW00013937.csv', 'USW00014923.csv', 'USW00023155.csv', 'USW00093062.csv', 'USM00070261.csv', 'USM00072250.csv', 'USM00072383.csv', 'USM00072489.csv', 'USM00072654.csv','USM00074560.csv', 'USW00012810.csv', 'USW00013819.csv', 'USW00013942.csv', 'USW00014924.csv', 'USW00023177.csv', 'USW00093130.csv', 'USM00070267.csv', 'USM00072253.csv', 'USM00072385.csv', 'USM00072493.csv', 'USM00072655.csv', 'USM00074570.csv', 'USW00012811.csv', 'USW00013820.csv', 'USW00013943.csv', 'USW00014927.csv', 'USW00023178.csv', 'USW00093170.csv', 'USM00070270.csv', 'USM00072255.csv', 'USM00072387.csv', 'USM00072497.csv', 'USM00072659.csv', 'USM00074574.csv', 'USW00012814.csv', 'USW00013821.csv', 'USW00013944.csv', 'USW00021502.csv', 'USW00023183.csv', 'USW00093204.csv', 'USM00070273.csv', 'USM00072256.csv', 'USM00072388.csv', 'USM00072501.csv', 'USM00072662.csv', 'USM00074606.csv', 'USW00012815.csv', 'USW00013822.csv', 'USW00013946.csv', 'USW00022002.csv', 'USW00023188.csv', 'USW00093701.csv', 'USM00070291.csv', 'USM00072257.csv', 'USM00072389.csv', 'USM00072503.csv', 'USM00072666.csv', 'USM00074611.csv', 'USW00012819.csv', 'USW00013824.csv', 'USW00013947.csv', 'USW00022003.csv', 'USW00023206.csv', 'USW00093756.csv', 'USM00070308.csv', 'USM00072260.csv', 'USM00072391.csv', 'USM00072506.csv', 'USM00072672.csv', 'USM00074612.csv', 'USW00012824.csv', 'USW00013825.csv', 'USW00013960.csv', 'USW00022004.csv', 'USW00023207.csv', 'USW00093804.csv', 'USM00070316.csv', 'USM00072261.csv', 'USM00072393.csv', 'USM00072509.csv', 'USM00072677.csv', 'USM00074614.csv', 'USW00012825.csv', 'USW00013826.csv', 'USW00013977.csv', 'USW00022502.csv', 'USW00023211.csv', 'USW00093812.csv', 'USM00070326.csv', 'USM00072263.csv', 'USM00072394.csv', 'USM00072514.csv', 'USM00072681.csv', 'USM00074618.csv', 'USW00012863.csv', 'USW00013827.csv', 'USW00013988.csv', 'USW00022504.csv', 'USW00023213.csv', 'USW00093821.csv', 'USM00070350.csv', 'USM00072264.csv', 'USM00072401.csv', 'USM00072515.csv', 'USM00072683.csv', 'USM00074619.csv', 'USW00012880.csv', 'USW00013828.csv', 'USW00014702.csv', 'USW00022505.csv', 'USW00023214.csv', 'USW00093822.csv', 'USM00070381.csv', 'USM00072265.csv', 'USM00072402.csv', 'USM00072518.csv', 'USM00072688.csv', 'USM00074626.csv', 'USW00012881.csv', 'USW00013829.csv', 'USW00014703.csv', 'USW00022514.csv', 'USW00023234.csv', 'USW00093871.csv', 'USM00070398.csv', 'USM00072266.csv', 'USM00072403.csv', 'USM00072520.csv', 'USM00072693.csv', 'USM00074630.csv', 'USW00012901.csv', 'USW00013830.csv', 'USW00014704.csv', 'USW00023001.csv', 'USW00024001.csv', 'USW00093877.csv', 'USM00070409.csv', 'USM00072267.csv', 'USM00072405.csv', 'USM00072524.csv', 'USM00072694.csv', 'USM00074631.csv', 'USW00012902.csv', 'USW00013831.csv', 'USW00014708.csv', 'USW00023006.csv', 'USW00024002.csv', 'USW00093917.csv', 'USM00070414.csv', 'USM00072268.csv', 'USM00072407.csv', 'USM00072528.csv', 'USM00072698.csv', 'USM00074638.csv', 'USW00012904.csv', 'USW00013833.csv', 'USW00014709.csv', 'USW00023008.csv', 'USW00024014.csv', 'USW00093963.csv', 'USM00070439.csv', 'USM00072269.csv', 'USM00072408.csv', 'USM00072531.csv', 'USM00072712.csv', 'USM00074641.csv', 'USW00012906.csv', 'USW00013835.csv', 'USW00014710.csv', 'USW00023010.csv', 'USW00024016.csv', 'USW00094176.csv', 'USM00070454.csv', 'USM00072273.csv', 'USM00072409.csv', 'USM00072532.csv', 'USM00072734.csv', 'USM00074646.csv', 'USW00012908.csv', 'USW00013836.csv', 'USW00014711.csv', 'USW00023011.csv', 'USW00024027.csv', 'USW00094730.csv', 'USM00070485.csv', 'USM00072274.csv', 'USM00072412.csv', 'USM00072533.csv', 'USM00072745.csv', 'USM00074671.csv', 'USW00012909.csv', 'USW00013837.csv', 'USW00014712.csv', 'USW00023012.csv', 'USW00024035.csv', 'USW00094750.csv', 'USM00070489.csv', 'USM00072278.csv', 'USM00072414.csv', 'USM00072534.csv', 'USM00072747.csv','USM00074693.csv', 'USW00012911.csv', 'USW00013838.csv', 'USW00014716.csv', 'USW00023019.csv', 'USW00024101.csv', 'USW00094761.csv', 'USM00072201.csv', 'USM00072280.csv', 'USM00072421.csv', 'USM00072535.csv', 'USM00072753.csv', 'USM00074695.csv', 'USW00012914.csv', 'USW00013839.csv', 'USW00014719.csv', 'USW00023021.csv', 'USW00024110.csv', 'USW00094763.csv', 'USM00072202.csv', 'USM00072281.csv', 'USM00072424.csv', 'USM00072536.csv', 'USM00072764.csv', 'USM00074724.csv', 'USW00012915.csv', 'USW00013846.csv', 'USW00014721.csv', 'USW00023027.csv', 'USW00024134.csv', 'USW00094775.csv', 'USM00072203.csv', 'USM00072293.csv', 'USM00072425.csv', 'USM00072537.csv', 'USM00072767.csv', 'USM00074732.csv', 'USW00012920.csv', 'USW00013850.csv', 'USW00014722.csv', 'USW00023030.csv', 'USW00024135.csv', 'USW00094881.csv', 'USM00072206.csv', 'USM00072295.csv', 'USM00072426.csv', 'USM00072545.csv', 'USM00072768.csv', 'USM00074754.csv', 'USW00013702.csv', 'USW00013851.csv', 'USW00014723.csv', 'USW00023031.csv', 'USW00024156.csv', 'USW00094922.csv', 'USM00072208.csv', 'USM00072297.csv', 'USM00072432.csv', 'USM00072546.csv', 'USM00072773.csv', 'USM00074755.csv', 'USW00013705.csv', 'USW00013855.csv', 'USW00014725.csv', 'USW00023038.csv', 'USW00024163.csv', 'USW00094925.csv', 'USM00072210.csv', 'USM00072303.csv', 'USM00072433.csv', 'USM00072548.csv', 'USM00072776.csv', 'USM00074768.csv', 'USW00013707.csv', 'USW00013856.csv', 'USW00014726.csv', 'USW00023041.csv', 'USW00024203.csv', 'USW00094937.csv', 'USM00072213.csv', 'USM00072304.csv', 'USM00072434.csv', 'USM00072557.csv', 'USM00072777.csv', 'USM00074777.csv', 'USW00013708.csv', 'USW00013857.csv', 'USW00014734.csv', 'USW00023043.csv', 'USW00024206.csv', 'USW00094941.csv', 'USM00072214.csv', 'USM00072305.csv', 'USM00072435.csv', 'USM00072558.csv', 'USM00072781.csv', 'USM00074778.csv', 'USW00013710.csv', 'USW00013859.csv', 'USW00014738.csv', 'USW00023052.csv', 'USW00024207.csv', 'USXUA724586.csv', 'USM00072216.csv', 'USM00072306.csv', 'USM00072436.csv', 'USM00072562.csv', 'USM00072783.csv', 'USM00074780.csv', 'USW00013711.csv', 'USW00013860.csv', 'USW00014742.csv', 'USW00023068.csv', 'USW00024220.csv', 'USXUA724755.csv', 'USM00072220.csv', 'USM00072308.csv', 'USM00072438.csv', 'USM00072563.csv', 'USM00072786.csv', 'USM00074790.csv', 'USW00013712.csv', 'USW00013874.csv', 'USW00014748.csv', 'USW00023102.csv', 'USW00024228.csv', 'USXUA725185.csv', 'USM00072221.csv', 'USM00072311.csv', 'USM00072440.csv', 'USM00072564.csv', 'USM00072792.csv', 'USM00074792.csv', 'USW00013715.csv', 'USW00013882.csv', 'USW00014751.csv', 'USW00023103.csv', 'USW00024234.csv', 'USXUAC03522.csv', 'USM00072222.csv', 'USM00072317.csv', 'USM00072445.csv', 'USM00072566.csv', 'USM00072793.csv', 'USM00074794.csv', 'USW00013717.csv', 'USW00013893.csv', 'USW00014752.csv', 'USW00023104.csv', 'USW00024264.csv', 'USXUAC03540.csv', 'USM00072224.csv', 'USM00072326.csv', 'USM00072450.csv', 'USM00072569.csv', 'USM00072797.csv', 'USM00074796.csv', 'USW00013718.csv', 'USW00013894.csv', 'USW00014755.csv', 'USW00023108.csv', 'USW00025325.csv', 'USXUAC03546.csv', 'USM00072225.csv', 'USM00072327.csv', 'USM00072451.csv', 'USM00072572.csv', 'USM00072798.csv', 'USM00091162.csv', 'USW00013719.csv', 'USW00013903.csv', 'USW00014761.csv', 'USW00023109.csv', 'USW00026401.csv', 'USXUAC03671.csv', 'USM00072226.csv', 'USM00072340.csv', 'USM00072455.csv', 'USM00072575.csv', 'USM00074001.csv', 'USM00091165.csv', 'USW00013729.csv', 'USW00013904.csv', 'USW00014771.csv', 'USW00023110.csv', 'USW00026410.csv', 'USM00072228.csv', 'USM00072344.csv', 'USM00072456.csv', 'USM00072576.csv', 'USM00074002.csv', 'USM00091170.csv','USW00013733.csv', 'USW00013905.csv', 'USW00014804.csv', 'USW00023111.csv', 'USW00026414.csv', 'USM00072229.csv', 'USM00072349.csv', 'USM00072457.csv', 'USM00072581.csv', 'USM00074004.csv', 'USM00091176.csv', 'USW00013780.csv', 'USW00013907.csv', 'USW00014814.csv', 'USW00023117.csv', 'USW00026440.csv', 'USM00072230.csv', 'USM00072351.csv', 'USM00072462.csv', 'USM00072582.csv', 'USM00074005.csv', 'USM00091182.csv', 'USW00013792.csv', 'USW00013909.csv', 'USW00014821.csv', 'USW00023119.csv', 'USW00026508.csv', 'USM00072231.csv', 'USM00072352.csv', 'USM00072464.csv', 'USM00072583.csv', 'USM00074207.csv', 'USM00091190.csv', 'USW00013801.csv', 'USW00013916.csv', 'USW00014822.csv', 'USW00023122.csv', 'USW00026509.csv', 'USM00072232.csv', 'USM00072354.csv', 'USM00072465.csv', 'USM00072591.csv', 'USM00074389.csv', 'USM00091285.csv', 'USW00013802.csv', 'USW00013920.csv', 'USW00014834.csv', 'USW00023125.csv', 'USW00026601.csv', 'USM00072235.csv', 'USM00072355.csv', 'USM00072468.csv', 'USM00072597.csv', 'USM00074392.csv', 'USW00003123.csv', 'USW00013804.csv', 'USW00013923.csv', 'USW00014838.csv', 'USW00023126.csv', 'USW00026605.csv']

    # a = []
    # for i in files_list:
    #     x = []
    #     x = i.split()
    #     for j in x:
    #         a.append(j)

    # print(len(a))

    # files_list = ["INM00042071.csv",  "INM00042273.csv",  "INM00042410.csv",  "INM00042543.csv",  "INM00042779.csv",  "INM00043003.csv",  "INM00043192.csv", "INM00043333.csv",  "INXUAE05449.csv",  "INXUAE05784.csv",  "INXUAE05834.csv",  "INM00042101.csv",  "INM00042314.csv",  "INM00042416.csv",  "INM00042591.csv",  "INM00042798.csv",  "INM00043041.csv",  "INM00043194.csv",  "INM00043344.csv",  "INXUAE05454.csv",  "INXUAE05786.csv",  "INXUAE05840.csv",  "INM00042103.csv",  "INM00042339.csv",  "INM00042475.csv",  "INM00042623.csv",  "INM00042840.csv",  "INM00043049.csv",  "INM00043201.csv",  "INM00043346.csv",  "INXUAE05455.csv",  "INXUAE05794.csv",  "INM00042111.csv",  "INM00042348.csv",  "INM00042492.csv",  "INM00042634.csv",  "INM00042867.csv",  "INM00043063.csv",  "INM00043237.csv",  "INM00043353.csv",  "INXUAE05457.csv",  "INXUAE05796.csv",  "INM00042165.csv",  "INM00042369.csv",  "INM00042498.csv",  "INM00042647.csv",  "INM00042874.csv",  "INM00043110.csv",  "INM00043284.csv",  "INM00043369.csv",  "INXUAE05462.csv",  "INXUAE05798.csv",  "INM00042182.csv",  "INM00042379.csv",  "INM00042516.csv",  "INM00042675.csv",  "INM00042895.csv",  "INM00043128.csv",  "INM00043285.csv",  "INM00043371.csv",  "INXUAE05466.csv",  "INXUAE05800.csv",  "INM00042189.csv", "INM00042382.csv",  "INM00042539.csv",  "INM00042701.csv",  "INM00042909.csv",  "INM00043181.csv",  "INM00043295.csv",  "INXUAC03369.csv",  "INXUAE05469.csv",  "INXUAE05822.csv",  "INM00042260.csv",  "INM00042397.csv",  "INM00042542.csv",  "INM00042734.csv",  "INM00042970.csv",  "INM00043185.csv",  "INM00043311.csv",  "INXUAE05432.csv",  "INXUAE05473.csv",  "INXUAE05832.csv"]

    # df = pd.read_csv('INM00042103.csv')
    file_name = file_name + '.csv'
    df = pd.read_csv(file_name)
    df= df.tail(30000)
    df.reset_index(inplace=True)
    # for i in files_list:
    #	df1 = pd.read_csv(i)
    #	a = df1['station-code'].tolist()
    #	x = i[:-4]
    #	c = 0
    #	for j in a:
    #               if j == x:
    #                       c = c+1
    #	print(i, ': ', x, c, len(df1))

    # for i in df.columns:
    #        print(i, ': ', df[i].unique())


    df = df.replace(np.nan, 0)


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


    df = df.replace(np.nan, 0)


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







    #df= df[['wdir', 'pressure','temp','rh','dpdp' 'wspd', 'gph', 'npv', 'day', 'hour', 'month', 'year', 'lvl12_1_1', 'lvl12_1_2', 'lvl12_2_0']]
    x = l1_col + l2_col + ['temp', 'rh', 'dpdp', 'wdir', 'pressure', 'wspd', 'gph', 'npv', 'day', 'month', 'year']
    #x.append(t)
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

    print(total_rows)
    m= total_rows-10

    train = df[:m]

    #is_2017 = df.tail (10)
    test = df[m:]

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

    # numeric_features = train.select_dtypes(include=[np.number])
    # #print(numeric_features.dtypes)
    #
    # corr =numeric_features.corr()
    # print(corr['wspd'].sort_values(ascending=False))
    # #print(corr)
    # #correlation matrix
    # f, ax = plt.subplots(figsize=(12, 9))
    cor =df.corr(method ='spearman')
    cor_target= abs(cor['wspd'])
    print(cor_target)

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
    # layers = [X_train.shape[1], 20, 15, 20, Y_train.shape[1]]
    #
    # model = Sequential()
    #
    # model.add(LSTM(
    #     layers[1],
    #     input_shape=(None, layers[0]),
    #     return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(layers[2], return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(layers[3], return_sequences=False))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(layers[4], activation="linear"))
    #
    # model.compile(loss="mse", optimizer='rmsprop')
    #
    # print(model.summary())
    #
    # history = None
    #
    # try:
    #     history = model.fit(
    #         X_train, Y_train,
    #         batch_size=512,
    #         epochs=50,
    #         validation_split=0.1,
    #         callbacks=[
    #             TensorBoard(log_dir='/tmp/tensorboard', write_graph=True),
    #             # EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    #         ])
    # except KeyboardInterrupt:
    #     print("\nTraining interrupted")
    #
    # predicted = model.predict(X_test)
    #
    # from sklearn.metrics import mean_squared_error, mean_absolute_error
    #
    # print("MAE {:.3}, MSE {:.3}".format(
    #     mean_absolute_error(Y_test, predicted),
    #     mean_squared_error(Y_test, predicted)))
    #


    # sns.heatmap(corr, vmax=1, square=True)
    # plt.show()

    #
    #
    #
    #
    # from keras.models import Sequential
    # from keras.layers.core import Dense, Activation, Dropout
    # from keras.layers.recurrent import LSTM
    # from keras.optimizers import RMSprop
    #
    # def build_model(layers):
    #     model = Sequential()
    #
    #     model.add(LSTM(
    #         layers[1],
    #         input_shape=(None, layers[0]),
    #         return_sequences=True))
    #     model.add(Dropout(0.2))
    #
    #     model.add(LSTM(layers[2], return_sequences=True))
    #     model.add(Dropout(0.2))
    #
    #     model.add(LSTM(layers[3], return_sequences=False))
    #     model.add(Dropout(0.2))
    #
    #     model.add(Dense(layers[4], activation="linear"))
    #
    #     model.compile(loss="mse", optimizer='rmsprop')
    #
    #     print(model.summary())
    #
    #     return model
    #
    #
    # def run_network(X_train, y_train, X_test, layers, epochs, batch_size=512):
    #     model = build_model(layers)
    #     history = None
    #
    #     try:
    #         history = model.fit(
    #             X_train, y_train,
    #             batch_size=batch_size,
    #             epochs=epochs,
    #             validation_split=0.1,
    #             callbacks=[
    #                 TensorBoard(log_dir='/tmp/tensorboard', write_graph=True),
    #                 # EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    #             ])
    #     except KeyboardInterrupt:
    #         print("\nTraining interrupted")
    #
    #     predicted = model.predict(X_test)
    #
    #     return model, predicted, history
    #
    # model, predicted, history = run_network(
    #     X_train,
    #     Y_train,
    #     X_test,
    #     layers=[X_train.shape[2], 20, 15, 20, Y_train.shape[1]],
    #     epochs=50)

    #
    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

    params = {'task':'train', 'boosting_type':'gbdt', 'objective':'regression',
                'metric': {'rmse'}, 'num_leaves': 10, 'learning_rate': 0.08,
                'feature_fraction': 0.7, 'max_depth': 8, 'verbose': 100,
                'num_boost_round':2000, 'early_stopping_rounds':500,
            'nthread':-1}

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval
                    ,early_stopping_rounds=5)
    #
    # # gbm = joblib.load('lgbm_model.pkl')
    #
    #
    #
    #
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    rmse = mean_squared_error(Y_test, y_pred) ** 0.5
    print('The rmse of prediction is:', rmse)

    # #
    df_test = pd.get_dummies(test)
    df_test.drop('wspd',axis=1,inplace=True)
    X_prediction = df_test.values
    # #
    predictions = gbm.predict(X_prediction,num_iteration=gbm.best_iteration)

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
    plt.show()
    #
    # # plt.savefig('relation.png')
    # # copyfile("relation.png", "static/relation.png")
    # # plt.savefig('static/relation.png')
    # # joblib.dump(gbm, 'lgbm_model.pkl')
    #
    return rmse, acc

m,n=deploy(file_name)
print(m,"  ",n)
