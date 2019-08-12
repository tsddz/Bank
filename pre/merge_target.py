import pandas as pd
import sys
sys.setrecursionlimit(1000000)
LISTname='FNCG'
train_A=pd.read_csv('E:\Bank\DATA_ALL/YDN1_TARGET.csv')
train_B=pd.read_csv('E:\Bank\DATA_ALL\YDN1_'+LISTname+'.csv')

deal = list(train_B.MATU_DAT)
for i in range(len(deal)):
    try:
        deal[i] = float(deal[i])
    except:
        deal[i] = float('NaN')
train_B.MATU_DAT = pd.Series(deal)

deal = list(train_B.DATA_DT)
for i in range(len(deal)):
    try:
        deal[i] = float(deal[i])
    except:
        deal[i] = float('NaN')
train_B.DATA_DT = pd.Series(deal)

deal = list(train_B.CLS_ACCT_DAT)
for i in range(len(deal)):
    try:
        deal[i] = float(deal[i])
    except:
        deal[i] = float('NaN')
train_B.CLS_ACCT_DAT = pd.Series(deal)


newone=pd.merge(train_A,train_B,how='inner',left_on='CUST_NO',right_on='CUST_NO',suffixes=['_TARGET','_'+LISTname])
newone.eval('cha31 = MATU_DAT - DATA_DT', inplace=True)
newone.eval('cha32 = ARG_CRT_DAT - DATA_DT', inplace=True)
newone.eval('cha33 = CLS_ACCT_DAT - DATA_DT', inplace=True)

keylist=list(newone.keys())

for i in range(len(keylist)):
    #if keylist[i] not in ['FLAG','CUST_NO','end_dat','DATA_DAT','PROD_CLS_CD','CCY_CD']:
    if keylist[i][0]=='c':
        try:
            print('relationship between FLAG and ' + keylist[i] + ' : ')
            code = 'print(newone.FLAG.corr(pd.Series(list(newone.' + keylist[i] + '))))'
            exec(code)
        except:
            print('relationship between FLAG and ' + keylist[i] + ' : not avaliable')



def make_bond():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,bond))
        df.eval('cha31 = MATU_DAT - ARG_CRT_DAT',inplace=True)
        gp = df.groupby(['CUST_NO']).agg({'ARG_CRT_DAT':['sum','mean','max','min','std','count'],
                                          'MATU_DAT':['sum','mean','max','min','std','count'],
                                          'ARG_LIF_CYC_STA_CD':['sum','mean','max','min','std','count'],
                                          'PTPN_AMT':['sum','mean','max','min','std','count'],
                                          'NET_VAL_TOT_AMT':['sum','mean','max','min','std'],
                                          'cha31':['sum','mean','max','min','std']})
        gp.columns.droplevel(0)
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
    return result[0],result[1]


#train_B[train_B.CUST_NO==7910779762613928].MATU_DAT.max()
#train_B[train_B.CUST_NO==7910779762613928][train_B[train_B.CUST_NO==7910779762613928].MATU_DAT==44230]
import numpy as np
pd.Series(newone.LN_TERM).corr(pd.Series(np.array(newone.MATU_DAT)-np.array(newone.ARG_CRT_DAT)))



deal=list(train_B.CAN_RETURN_AMT)
for i in range(len(deal)):
    try:
        deal[i]=float(deal[i])
    except:
        deal[i]=0
train_B.CAN_RETURN_AMT=pd.Series(deal)

deal=list(train_B.TOT_AC_QUOT)
for i in range(len(deal)):
    try:
        deal[i]=float(deal[i])
    except:
        deal[i]=float(deal[i][:6])
train_B.TOT_AC_QUOT=pd.Series(deal)

newone=pd.merge(train_A,train_B,how='inner',left_on='CUST_NO',right_on='CUST_NO',suffixes=['_TARGET','_'+LISTname])
labeln=newone[newone.FLAG=='1']
