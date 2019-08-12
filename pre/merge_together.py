import pandas as pd
import matplotlib.pyplot as plt


#data_txt=np.loadtxt('../Data/YDN1_TARGET.txt')
#data_txtDF=pd.DataFrame(data_txt)
#print(data_txtDF.head())

train_A=pd.read_csv('E:\Bank\TRY_MERGE/YDN1_CUST_INFO.csv')
train_B=pd.read_csv('E:\Bank\TRY_MERGE/YDN1_IDV_TD.csv')

mm=pd.merge(train_A,train_B,how='outer',left_on='CUST_NO',right_on='CUST_NO',suffixes=['_CUST_INFO','_IDV_TD'])

print(len(set(train_A.CUST_NO)))


train_A=pd.read_csv('E:\Bank\DATA_ALL/YDN1_TARGET.csv')
train_B=pd.read_csv('E:\Bank\DATA_TEST/YDN1_TARGET.csv')

print(len(train_A))
print(len(train_B))


mm=pd.merge(train_A,train_B,how='outer',left_on='CUST_NO',right_on='CUST_NO',suffixes=['_TRAIN','_TEST'])
print(len(mm))

mm=pd.merge(train_A,train_B,how='inner',left_on='CUST_NO',right_on='CUST_NO',suffixes=['_TRAIN','_TEST'])
print(len(mm))