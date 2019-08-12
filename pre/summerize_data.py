import pandas as pd
import matplotlib.pyplot as plt


#data_txt=np.loadtxt('../Data/YDN1_TARGET.txt')
#data_txtDF=pd.DataFrame(data_txt)
#print(data_txtDF.head())
LISTname='FNCG'
train_target=pd.read_csv('E:/BANK/DATA_ALL/YDN1_TARGET.csv')
train_A=pd.read_csv('E:\Bank\DATA_ALL\YDN1_'+LISTname+'.csv', nrows=140000)





train_A.describe()
pd.set_option('display.max_rows', 5000) #最大行数
pd.set_option('display.max_columns', 5000) #最大列数
pd.set_option('display.width', 40000) #页面宽度
print(train_A.describe())
print('\n')
print(train_A.head(10))
print('\n')


df['CHANLCD_CDOM'].value_counts()
'''
print('\n')
print(train_A.head(10))
print('\n')
print(sum(train_A.FLAG))
print('\n')
print(train_A.info())
'''


def givemehist(i,train_A,key_name):
    print('\n')
    plt.figure(i)
    print([key_name+':'])
    SETV=set(train_A.MTH_ACT_DAYS_TOT)
    print(set(list(map(str,SETV))))
    print(len((set(list(map(str,SETV))))))
    LIST=list((set(list(map(str,SETV)))))
    print(LIST)
   # print(min(LIST))

    #print(len(set(train_A.key_name)))
    newcode = 'n, bins, patches = plt.hist(train_A.'
    newcode += key_name
    newcode += ', bins=100, facecolor="green")'
    # n, bins, patches = plt.hist(train_A.DATA_DAT, bins=50, facecolor='green', alpha=0.75)
    exec(newcode)
    print('\n')
    plt.title(key_name)
    plt.show()
givemehist(1,train_A,'MTH_ACT_DAYS_TOT')



#test_A=pd.read_csv('../DATA_TEST/YDN1_IDV_TD.csv')
#df_train1 = train_A.merge(test_A,on='CUST_NO',how='left')


#givemehist(2,df_train1,'DATA_DAT')

'''
keylist=list(train_A.keys())

for i in range(len(keylist)):
    print('\n')
    plt.figure(i)
    print(keylist[i])
    exec('print(len(set(train_A.'+keylist[i]+')))')
    newcode='n, bins, patches = plt.hist(train_A.'
    newcode+=keylist[i]
    newcode+=', bins=50, facecolor="green", alpha=0.75)'
    #n, bins, patches = plt.hist(train_A.DATA_DAT, bins=50, facecolor='green', alpha=0.75)
    exec(newcode)
    print('\n')
    plt.show()

print(len(set(train_A.DATA_DAT)))
print(len(set(train_A.CUST_NO)))
print(len(set(train_A.CUST_NO)))
print(len(set(train_A.CUST_NO)))



n, bins, patches = plt.hist(train_A.DATA_DAT, bins=50, facecolor='green', alpha=0.75)
plt.show()

'''
import os
import pandas as pd
direct='E:\Bank\DATA_TEST'
txtlist=os.listdir(direct)
for i in range(len(txtlist)):
    train_A = pd.read_csv(direct + '\\'+txtlist[i])
    print(txtlist[i])
    print(train_A.head(2))
















