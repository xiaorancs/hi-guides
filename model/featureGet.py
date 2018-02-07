
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[ ]:

df_train_1 = pd.read_csv('../data/dataSet/df_train_1.csv')
df_test_1 = pd.read_csv('../data/dataSet/df_test_1.csv')


df_train_2 = pd.read_csv('../data/dataSet/df_train_2.csv')
df_test_2 = pd.read_csv('../data/dataSet/df_test_2.csv')


df_train_3 = pd.read_csv('../data/dataSet/df_train_3.csv')
df_test_3 = pd.read_csv('../data/dataSet/df_test_3.csv')


df_train_4 = pd.read_csv('../data/dataSet/df_train_4.csv')
df_test_4 = pd.read_csv('../data/dataSet/df_test_4.csv')

df_train_5 = pd.read_csv('../data/dataSet/df_train_5.csv')
df_test_5 = pd.read_csv('../data/dataSet/df_test_5.csv')

df_train_6 = pd.read_csv('../data/dataSet/df_train_6.csv')
df_test_6 = pd.read_csv('../data/dataSet/df_test_6.csv')

df_train_7 = pd.read_csv('../data/dataSet/df_train_7.csv')
df_test_7 = pd.read_csv('../data/dataSet/df_test_7.csv')

df_train_8 = pd.read_csv('../data/dataSet/df_train_8.csv')
df_test_8 = pd.read_csv('../data/dataSet/df_test_8.csv')

df_train_9 = pd.read_csv('../data/dataSet/df_train_9.csv')
df_test_9 = pd.read_csv('../data/dataSet/df_test_9.csv')

df_train_10 = pd.read_csv('../data/dataSet/df_train_10.csv')
df_test_10 = pd.read_csv('../data/dataSet/df_test_10.csv')

df_train_11 = pd.read_csv('../data/dataSet/df_train_11.csv')
df_test_11 = pd.read_csv('../data/dataSet/df_test_11.csv')

df_train_12 = pd.read_csv('../data/dataSet/df_train_12.csv')
df_test_12 = pd.read_csv('../data/dataSet/df_test_12.csv')

df_train_12_top110 = pd.read_csv('../data/dataSet/df_train_12_top110.csv')
df_test_12_top110 = pd.read_csv('../data/dataSet/df_test_12_top110.csv')

df_train_13 = pd.read_csv('../data/dataSet/df_train_13.csv')
df_test_13 = pd.read_csv('../data/dataSet/df_test_13.csv')



# In[ ]:

stopfeatures = ['hasOrder','actionTypeIn24Avg','actionTypeIn14Avg',
                'actionTime24CurLastDiff','actionTime59CurLastDiff','actionTimeLast0.042Days8Cnt',
                'actionTimeLast0.042Days9Cnt','actionTimeLast0.125Days8Cnt','actionTimeLast0.125Days9Cnt',
                'actionTimeLast0.25Days8Cnt','actionTimeLast0.25Days9Cnt','actionTimeLast0.5Days24Cnt',
                'actionTimeLast0.5Days59Cnt','actionTimeLast0.5Days9Cnt','actionTimeLast15Days24Cnt',
                'actionTimeLast15Days59Cnt','actionTimeLast1Days24Cnt','actionTimeLast1Days59Cnt',
                'actionTimeLast1Days9Cnt','actionTimeLast3Days24Cnt','actionTimeLast3Days59Cnt',
                'actionTimeLast3Days9Cnt','actionTimeLast5Days24Cnt','actionTimeLast5Days59Cnt',
                'actionTimeLast7Days24Cnt','actionTimeLast7Days59Cnt','actionTimeLast5Days9Cnt',
                 # df_xtrain_6
                'actionTime24LastFirstDiffRate','actionType8Cnt','actionTypeIn14Cnt','actionTime24CurFirstDiff',
                'actionType3Cnt_actionCnt_rate','actionTypeIn24Cnt','actionTime24LastFirstDiff',
                'actionTypeIn29Cnt','actionTypeCnt','actionType4Cnt','actionType2Cnt',
                'actionType3Cnt','actionTypeIn29Cnt_actionCnt_rate','actionTypeIn14Cnt_actionCnt_rate',
                'actionType9Cnt','actionTypeIn14Avg','actionTypeIn59Avg','actionTypeIn29Avg','actionTypeIn24Avg',
                # df_xtrain_7
                'type4DiffTimeAvg','typeDiff8TimeMaxSubAvg','type4Last3DiffTime','type2Last2DiffTime',
                'type9DiffTimeAvg','typeDiff2TimeMaxSubAvg','type7Last2DiffTime','typeDiff4TimeAvgSubMin',
                'type8Last2DiffTime','type4Last2DiffTime','typeDiff7TimeMaxSubMin','typeDiff8TimeAvgSubMin',
                'type3DiffTimeMax','type2DiffTimeAvg','type3Last2DiffTime','type2DiffTimeMin',
                'typeDiff2TimeAvgSubMin','type8Last1DiffTime','type8DiffTimeAvg','type8DiffTimeMin',
                'type3DiffTimeAvg','type2DiffTimeMax','type4DiffTimeMax','typeDiff3TimeMaxSubMin',
                'typeDiff8TimeMaxSubMin','type8Last3DiffTime','typeDiff2TimeMaxSubMin','type7Last3DiffTime',
                'typeDiff4TimeMaxSubMin','typeDiff7TimeAvgSubMin','type2Last3DiffTime','typeDiff3TimeAvgSubMin',
                'type9DiffTimeMax','type2Last1DiffTime','type4Last1DiffTime','typeDiff9TimeAvgSubMin',
                'type3DiffTimeMin','type3Last1DiffTime','type9Last2DiffTime','type3Last3DiffTime',
                'type4DiffTimeMin','type9Last3DiffTime','type9DiffTimeMin','typeDiff9TimeMaxSubAvg',
                'type9Last1DiffTime','typeDiff9TimeMaxSubMin',
                # df_xtrain_8
                'actionTimeLast0.042Days1Cnt','actionTimeLast15Days2Cnt','actionTimeLast15Days8Cnt',
                'actionTimeLast0.5Days6Cnt','actionTimeLast0.25Days5Cnt','actionTimeLast0.25Days6Cnt',
                 'actionTimeLast0.125Days1Cnt','actionTimeLast0.5Days1Cnt','actionTimeLast0.125Days59Cnt',
                 'actionTimeLast15Days9Cnt','actionTimeLast5Days2Cnt','actionTimeLast7Days8Cnt',
                 'actionTimeLast7Days7Cnt','actionTimeLast7Days3Cnt','actionTimeLast7Days4Cnt',
                 'actionTimeLast5Days4Cnt','actionTimeLast3Days4Cnt','actionTimeLast7Days2Cnt',
                 'actionTimeLast1Days8Cnt','actionTimeLast0.5Days7Cnt','actionTimeLast3Days3Cnt',
                 'actionTimeLast3Days2Cnt','actionTimeLast1Days3Cnt','actionTimeLast0.125Days4Cnt',
                 'actionTimeLast0.125Days7Cnt','actionTimeLast5Days8Cnt','actionTimeLast1Days4Cnt',
                 'actionTimeLast0.042Days4Cnt','actionTimeLast0.25Days7Cnt','actionTimeLast3Days8Cnt',
                 'actionTimeLast0.042Days3Cnt','actionTimeLast3Days7Cnt','actionTimeLast0.5Days8Cnt',
                 'actionTimeLast5Days3Cnt','actionTimeLast0.125Days3Cnt','actionTimeLast1Days7Cnt',
                 'actionTimeLast0.5Days3Cnt','actionTimeLast0.125Days24Cnt','actionTimeLast1Days2Cnt',
                 'actionTimeLast0.125Days2Cnt','actionTimeLast0.25Days3Cnt','actionTimeLast5Days7Cnt',
                 'actionTimeLast0.042Days2Cnt','actionTimeLast0.5Days4Cnt','actionTimeLast7Days9Cnt',
                 'actionTimeLast0.5Days2Cnt','actionTimeLast5Days9Cnt','actionTimeLast0.25Days4Cnt',
                 'actionTimeLast0.25Days2Cnt','actionTimeLast3Days9Cnt','actionTimeLast1Days9Cnt',
                 'actionTimeLast0.25Days24Cnt','actionTimeLast0.125Days8Cnt','actionTimeLast0.125Days9Cnt',
                 'actionTimeLast0.25Days59Cnt','actionTimeLast0.5Days59Cnt','actionTimeLast0.042Days9Cnt',
                 'actionTimeLast0.042Days8Cnt','actionTimeLast0.25Days8Cnt','actionTimeLast1Days59Cnt',
                # df_xtrain_10
                'type7Last2Time','type2Last2Time','type9Last1Time','type8Last2Time','type2Last3Time',
                'type7Last3Time','type3Last3Time','type4Last2Time','type4Last3Time','type3Last2Time',
                'type8Last3Time','type9Last2Time','type9Last3Time']

               


# 1. [df_train_1 ---> df_train_12]: 使用所有的12个文件， submission1_12.csv
# 
# 3. [df_train_1 ---> df_train_12_top110.csv] - stopfeature : submisson1_120top_stopf.csv
# 
# 4. [df_train_1 ---> df_train_12_top110.csv] + df_train_13(stack) - stopfeature : submission1_120top_stopf.csv
# 
# 5. [df_train_1 ---> df_train_6,] + df_train_12.csv - stopfeature submission1_7_12.csv
# 
# 6. [df_train_1 ---> df_train_6,] + df_train_12.csv + df_strain_13 - stopfeature  submission1_7_12_13.csv
# 

# In[ ]:

def getDataSet1():
    df_train = pd.read_csv('../data/train/orderFuture_train.csv')
    df_test = pd.read_csv('../data/test/orderFuture_test.csv')

    df_train = pd.merge(df_train,df_train_1,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_1,how='left',on='userid')
   
    df_train = pd.merge(df_train,df_train_2,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_2,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_3,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_3,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_4,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_4,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_5,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_5,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_6,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_6,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_7,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_7,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_8,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_8,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_9,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_9,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_10,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_10,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_11,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_11,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_12,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_12,how='left',on='userid')
    
    goodfeature = list(df_test.columns[1:])

    for f in stopfeatures:
        if f in goodfeature:
            goodfeature.remove(f)
    
    # 设置特征数据，去除id数据，不能进行预测
    features = goodfeature

    label = 'orderType'
    
    return df_train, df_test, features, label


# In[ ]:

def getDataSet2():
    df_train = pd.read_csv('../data/train/orderFuture_train.csv')
    df_test = pd.read_csv('../data/test/orderFuture_test.csv')

    df_train = pd.merge(df_train,df_train_1,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_1,how='left',on='userid')
   
    df_train = pd.merge(df_train,df_train_2,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_2,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_3,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_3,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_4,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_4,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_5,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_5,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_6,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_6,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_7,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_7,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_8,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_8,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_9,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_9,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_10,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_10,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_11,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_11,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_12_top110,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_12_top110,how='left',on='userid')
    
    goodfeature = list(df_test.columns[1:])

    for f in stopfeatures:
        if f in goodfeature:
            goodfeature.remove(f)
    # 设置特征数据，去除id数据，不能进行预测
    features = goodfeature

    label = 'orderType'
    
    return df_train, df_test, features, label


# In[ ]:

def getDataSet3():
    df_train = pd.read_csv('../data/train/orderFuture_train.csv')
    df_test = pd.read_csv('../data/test/orderFuture_test.csv')

    df_train = pd.merge(df_train,df_train_1,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_1,how='left',on='userid')
   
    df_train = pd.merge(df_train,df_train_2,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_2,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_3,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_3,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_4,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_4,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_5,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_5,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_6,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_6,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_7,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_7,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_8,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_8,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_9,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_9,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_10,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_10,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_11,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_11,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_12_top110,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_12_top110,how='left',on='userid')

    df_train = pd.merge(df_train,df_train_13,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_13,how='left',on='userid')

    
    goodfeature = list(df_test.columns[1:])

    for f in stopfeatures:
        if f in goodfeature:
            goodfeature.remove(f)
    # 设置特征数据，去除id数据，不能进行预测
    features = goodfeature

    label = 'orderType'
    
    return df_train, df_test, features, label


# In[ ]:

def getDataSet4():
    df_train = pd.read_csv('../data/train/orderFuture_train.csv')
    df_test = pd.read_csv('../data/test/orderFuture_test.csv')

    df_train = pd.merge(df_train,df_train_1,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_1,how='left',on='userid')
   
    df_train = pd.merge(df_train,df_train_2,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_2,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_3,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_3,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_4,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_4,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_5,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_5,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_6,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_6,how='left',on='userid')
      
    df_train = pd.merge(df_train,df_train_12_top110,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_12_top110,how='left',on='userid')
    
    goodfeature = list(df_test.columns[1:])

    for f in stopfeatures:
        if f in goodfeature:
            goodfeature.remove(f)
    # 设置特征数据，去除id数据，不能进行预测
    features = goodfeature

    label = 'orderType'
    
    return df_train, df_test, features, label


# In[ ]:

def getDataSet5():
    df_train = pd.read_csv('../data/train/orderFuture_train.csv')
    df_test = pd.read_csv('../data/test/orderFuture_test.csv')

    df_train = pd.merge(df_train,df_train_1,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_1,how='left',on='userid')
   
    df_train = pd.merge(df_train,df_train_2,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_2,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_3,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_3,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_4,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_4,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_5,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_5,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_6,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_6,how='left',on='userid')
      
    df_train = pd.merge(df_train,df_train_12_top110,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_12_top110,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_13,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_13,how='left',on='userid')
    
    
    goodfeature = list(df_test.columns[1:])

    for f in stopfeatures:
        if f in goodfeature:
            goodfeature.remove(f)
    # 设置特征数据，去除id数据，不能进行预测
    features = goodfeature

    label = 'orderType'
    
    return df_train, df_test, features, label


# In[ ]:

def getDataSet6():
    df_train = pd.read_csv('../data/train/orderFuture_train.csv')
    df_test = pd.read_csv('../data/test/orderFuture_test.csv')

    df_train = pd.merge(df_train,df_train_1,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_1,how='left',on='userid')
   
    df_train = pd.merge(df_train,df_train_2,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_2,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_3,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_3,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_4,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_4,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_5,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_5,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_6,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_6,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_7,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_7,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_11,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_11,how='left',on='userid')
        
    
    df_train = pd.merge(df_train,df_train_12_top110,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_12_top110,how='left',on='userid')
    
    df_train = pd.merge(df_train,df_train_13,how='left',on='userid')
    df_test = pd.merge(df_test,df_test_13,how='left',on='userid')
    
    
    goodfeature = list(df_test.columns[1:])

    for f in stopfeatures:
        if f in goodfeature:
            goodfeature.remove(f)
    # 设置特征数据，去除id数据，不能进行预测
    features = goodfeature

    label = 'orderType'
    
    return df_train, df_test, features, label

