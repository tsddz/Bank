import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
train_dir = '/root/work/mytest/problems/1/train/'
test_dir = '/root/work/mytest/problems/1/A/'
target = 'YDN1_TARGET'
#金融性交易信息
tr = 'YDN1_TR'
tr2 = 'YDN1_TR'
#信用卡账户余额信息
cc_bal = 'YDN1_CC_ACCT_BAL'
#金融资产信息
asset = 'YDN1_ASSET'
asset2 = 'YDN1_ASSET'
#个贷账户信息
loan = 'YDN1_LOAN'
#账户信息
cust_info = 'YDN1_CUST_INFO'
cust_info2 = 'YDN1_CUST_INFO'
#基金账户信息
fund = 'YDN1_FUND'
#定期存款账户信息
idv_td = 'YDN1_IDV_TD'
#理财账户信息
fncg = 'YDN1_FNCG'
#国债账户信息
bond = 'YDN1_BOND'
#信用卡客户状态信息
cc_cust_sts = 'YDN1_CC_CUST_STS'

#计算CV的KS值
def compute_ks(label,predict_proba):
    fpr,tpr,thresholds = roc_curve(label,predict_proba)
    return max(tpr-fpr)

#lgb ks评价函数
def lgb_ks(preds,real):
    y_true = real.get_label()
    fpr,tpr,thresholds = roc_curve(y_true,preds)
    ks=max(tpr-fpr)
    return 'ks',ks,True

#生成训练及测试的用户集
def make_cust():
    df_train = pd.read_csv(os.path.join(train_dir,target))
    df_test = pd.read_csv(os.path.join(test_dir,target),header=None,names=['CUST_NO','end_dat'])
    return df_train,df_test

#生成训练及测试的信用卡账户余额信息集
def make_cc_bal():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,cc_bal))
        gp = df.groupby('CUST_NO').agg({'GENR_AC_QUOT':['sum','mean','max','count','min','std'],
                                        'TOT_AC_QUOT':['sum','mean','max','min','std'],
                                        'CAN_AMT':['sum','mean','max','min','std']})
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
        return result[0], result[1]

#生成训练及测试的金融资产信息集
def make_asset():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,asset))
        df['YAVER_AUM_BAL'][(df['YAVER_AUM_BAL']<0)]=None
        df['YAVER_AUM_BAL'].fillna(df['YAVER_AUM_BAL'].median())
        df['AUM_BAL_MAX'][(df['AUM_BAL_MAX']<0)]=None
        df['AUM_BAL_MAX'].fillna(df['AUM_BAL_MAX'].median())
        df['AUM_BAL_MAX_DATE'][(df['AUM_BAL_MAX_DATE']<0)]=None
        df['AUM_BAL_MAX_DATE'].fillna(df['AUM_BAL_MAX_DATE'].median())
        gp = df.groupby('CUST_NO').agg({'DAY_FA_BAL':['sum','mean','max','count','min','std'],
                                        'YAVER_FA_BAL':['sum','mean','max','min','std'],
                                        'DAY_AUM_BAL':['sum','mean','max','min','std'],
                                        'YAVER_AUM_BAL':['sum','mean','max','min','std'],
                                        'TOT_IVST_BAL':['sum','mean','max','min','std'],
                                        'MAVER_TOT_IVST_BAL':['sum','mean','max','min','std'],
                                        'SAVER_TOT_IVST_BAL':['sum','mean','max','min','std'],
                                        'YAVER_TOT_IVST_BAL':['sum','mean','max','min','std'],
                                        'FA_BAL_MAX':['sum','mean','max','min','std'],
                                        'FA_BAL_MAX_DATE':['sum','mean','max','min','std'],
                                        'AUM_BAL_MAX':['sum','mean','max','min','std'],
                                        'AUM_BAL_MAX_DATE':['sum','mean','max','min','std']})
        gp.columns.droplevel(0)
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
        return result[0], result[1]

#生成训练及测试的金融资产信息集
def make_asset2():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,asset2))
        df['YAVER_AUM_BAL'][(df['YAVER_AUM_BAL']<0)]=None
        df['YAVER_AUM_BAL'].fillna(df['YAVER_AUM_BAL'].median())
        df['AUM_BAL_MAX'][(df['AUM_BAL_MAX']<0)]=None
        df['AUM_BAL_MAX'].fillna(df['AUM_BAL_MAX'].median())
        df['AUM_BAL_MAX_DATE'][(df['AUM_BAL_MAX_DATE']<0)]=None
        df['AUM_BAL_MAX_DATE'].fillna(df['AUM_BAL_MAX_DATE'].median())
        gp = df.groupby('CUST_NO').agg({'DAY_AUM_BAL':['sum','mean','max','min','std'],
                                        'YAVER_AUM_BAL':['sum','mean','max','min','std'],
                                        'FA_BAL_MAX_DATE':['sum','mean','max','min','std'],
                                        'AUM_BAL_MAX_DATE':['sum','mean','max','min','std']})
        gp.columns.droplevel(0)
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
    return result[0], result[1]

#生成训练及测试的个贷账户信息
def make_loan():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,loan))
        df.eval('cha2_loan = MATU_DAT - ARG_CRT_DAT',inplace=True)
        gp = df.groupby(['CUST_NO','FORM_STS_CD']).agg({'LN_TERM':['sum','mean','max','count'],
                                                        'NML_CAP_BAL':['sum','count','mean','max','min','std'],
                                                        'TOT_PRVD_AMT':['sum','mean','max','min','std'],
                                                        'TOT_REVK_AMT':['sum','mean'],
                                                        'MTH_NML_CAP_ACCM':['sum','mean'],
                                                        'MTH_ACT_DAYS_TOT':['sum','max','min','std'],
                                                        'cha2_loan':['sum','max','min','std','count']})
        gp.columns.droplevel(0)
        gp.columns = ['LN_TERM_sum','LN_TERM_count','LN_TERM_mean','LN_TERM_max',
                      'NML_CAP_BAL_sum','NML_CAP_BAL_count','NML_CAP_BAL_mean','NML_CAP_BAL_max','NML_CAP_BAL_min','NML_CAP_BAL_std',
                      'TOT_PRVD_AMT_sum','TOT_PRVD_AMT_mean','TOT_PRVD_AMT_max','TOT_PRVD_AMT_min','TOT_PRVD_AMT_std',
                      'TOT_REVK_AMT_sum','TOT_REVK_AMT_mean',
                      'MTH_NML_CAP_ACCM_sum','MTH_NML_CAP_ACCM_mean',
                      'MTH_ACT_DAYS_TOT_sum','MTH_ACT_DAYS_TOT_max','MTH_ACT_DAYS_TOT_min','MTH_ACT_DAYS_TOT_std',
                      'cha2_loan_sum','cha2_loan_max','cha2_loan_min','cha2_loan_std','cha2_loan_count']
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.reset_index(inplace=True)
        gp['FORM_STS_CD'] = gp['FORM_STS_CD'].astype('str')
        gp_pivot = gp.pivot(index='CUST_NO',columns='FORM_STS_CD')
        gp_pivot.columns = ['_'.join(col).strip() for col in gp_pivot.columns.values]
        gp_pivot.reset_index(inplace=True)
        result.append(gp_pivot)
    return result[0], result[1]

#生成训练及测试的客户基本信息
def make_cust_info():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,cust_info))
        df['GC_BRTH'][(df['GC_BRTH']<0)]=None
        df['FST_ARG_CRT_DAT'][(df['FST_ARG_CRT_DAT']<0)]=None
        df['FMLY_YEAR_INCM'][(df['FMLY_YEAR_INCM']<0)]=None
        df['GC_BRTH'].fillna(df['GC_BRTH'].median())
        df['FST_ARG_CRT_DAT'].fillna(df['FST_ARG_CRT_DAT'].median())
        df['FMLY_YEAR_INCM'].fillna(df['FMLY_YEAR_INCM'].median())
        df['DGR_CD'].fillna(df['DGR_CD'].mode())
        df['CUST_SEX_CD'].fillna(df['CUST_SEX_CD'].mode())
        df.eval('cha_f_crt = FST_ARG_CRT_DAT - GC_BRTH',inplace=True)
        gp = df.groupby(['CUST_NO']).agg({'DGR_CD':['count','max','min'],
                                          'MRGE_STS_CD':['max','min','count'],
                                          'GC_BRTH':['max','min'],'cha_f_crt':['max','min'],
                                          'FST_ARG_CRT_DAT':['max','min'],
                                          'YEAR_INCM':['sum','mean','max','min','std'],
                                          'FMLY_YEAR_INCM':['sum','mean','max','min','std']})
        gp.columns.droplevel(0)
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
    return result[0],result[1]

#生成训练及测试的客户基本信息
def make_cust_info2():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,cust_info2))
        df['GC_BRTH'][(df['GC_BRTH']<0)]=None
        df['FST_ARG_CRT_DAT'][(df['FST_ARG_CRT_DAT']<0)]=None
        df['FMLY_YEAR_INCM'][(df['FMLY_YEAR_INCM']<0)]=None
        df['GC_BRTH'].fillna(df['GC_BRTH'].median())
        df['FST_ARG_CRT_DAT'].fillna(df['FST_ARG_CRT_DAT'].median())
        df['FMLY_YEAR_INCM'].fillna(df['FMLY_YEAR_INCM'].median())
        df['DGR_CD'].fillna(df['DGR_CD'].mode())
        df['CUST_SEX_CD'].fillna(df['CUST_SEX_CD'].mode())
        df.eval('cha_f_crt = FST_ARG_CRT_DAT - GC_BRTH',inplace=True)
        gp = df.groupby(['CUST_NO']).agg({'DGR_CD':['count','max','min'],
                                          'MRGE_STS_CD':['max','min','count'],
                                          'GC_BRTH':['max'],'cha_f_crt':['max'],
                                          'FST_ARG_CRT_DAT':['max'],
                                          'YEAR_INCM':['sum','mean'],
                                          'FMLY_YEAR_INCM':['sum','mean','max','min']})
        gp.columns.droplevel(0)
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
    return result[0],result[1]

#生成训练及测试的基金账户信息
def make_fund():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,fund))
        gp = df.groupby(['CUST_NO']).agg({'FUD_PROD_TYP_CD':['mean','max','min','count'],
                                          'RSK_RANK_CD':['mean','max','min'],
                                          'FUND_BAL':['sum','mean','max','min','std']})
        gp.columns.droplevel(0)
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
    return result[0],result[1]

#生成训练及测试的定期存款账户信息
def make_idv_td():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,idv_td))
        df['ARG_CRT_DAT'][(df['ARG_CRT_DAT']<0)]=None
        df['ARG_CRT_DAT'].fillna(df['ARG_CRT_DAT'].median())
        df['MATU_DAT'][(df['MATU_DAT']<0)]=None
        df['MATU_DAT'].fillna(df['MATU_DAT'].median())
        df.eval('cha1 = CLS_ACCT_DAT - ARG_CRT_DAT',inplace=True)
        df.eval('cha2 = MATU_DAT - ARG_CRT_DAT',inplace=True)
        gp = df.groupby(['CUST_NO']).agg({'ARG_CRT_DAT':['sum','mean','max','min','std','count'],
                                          'CLS_ACCT_DAT':['sum','mean','max','min','std','count'],
                                          'MATU_DAT':['sum','mean','max','min','std','count'],
                                          'REG_CAP':['sum','mean','max','min','std',],
                                          'EXEC_RATE':['sum','mean','max','min','std','count'],
                                          'CRBAL':['sum','mean','max','min','std'],
                                          'MOTH_CR_ACCM':['sum','mean','max','min','std'],
                                          'MTH_ACT_DAYS_TOT':['sum','mean','max','min','std'],
                                          'cha1':['sum','mean','max','min','std'],
                                          'cha2':['sum','mean','max','min','std']})
        gp.columns.droplevel(0)
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
    return result[0],result[1]

#生成训练及测试的理财账户信息
def make_fncg():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,fncg))
        df['MATU_DAT'].fillna(0,inplace = True)
        df['EXIT_SHR'].fillna(df['EXIT_SHR'].mean(),inplace = True)
        df.eval('cha11 = CLS_ACCT_DAT - ARG_CRT_DAT',inplace=True)
        df.eval('cha21 = MATU_DAT - ARG_CRT_DAT',inplace=True)
        gp = df.groupby(['CUST_NO']).agg({'ARG_CRT_DAT':['sum','mean','max','min','std','count'],
        'CLS_ACCT_DAT':['sum','mean','max','min','std','count'],
        'MATU_DAT':['sum','mean','max','min','std','count'],
        'PROD_RSK_RANK_CD':['sum','mean','max','min','std',],
        'PROD_PFT_TYP_CD':['sum','mean','max','min','std','count'],
        'EXIT_SHR':['sum','mean','max','min','std'],
        'CUST_IVST_CST':['sum','mean','max','min','std'],
        'cha11':['sum','mean','max','min','std'],
        'cha21':['sum','mean','max','min','std'],
        'CHANL_CD':['count'],'PROD_CLS_CD':['count']})
        gp.columns.droplevel(0)
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
    return result[0],result[1]

#生成训练及测试的国债账户信息
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

#生成训练及测试的信用卡客户状态信息
def make_cc_cust_sts():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,cc_cust_sts))
        gp = df.groupby(['CUST_NO']).agg({'CUST_CYC_CR_IND':['sum','mean','max','min','std','count'],
                                          'CUST_CD_VLU':['sum','mean','max','min','std','count']})
        gp.columns.droplevel(0)
        for column in list(gp.columns[gp.isnull().sum()>0]):
            mean_val = gp[column].median()
            gp[column].fillna(mean_val,inplace = True)
        gp.columns = ['_'.join(col).strip() for col in gp.columns.values]
        gp.reset_index(inplace=True)
        result.append(gp)
    return result[0],result[1]

#生成训练及测试的交易信息集
def make_tr():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,tr))
        gp = df.groupby(['CUST_NO','TR_CD']).agg({'TR_AMT':['sum','count','mean','max','min','std'],
                                                  'TR_DAT':['sum','count','mean','max','min','std']})
        gp.columns.droplevel(0)
        gp.columns = ['TR_AMT_sum','TR_AMT_count','TR_AMT_mean','TR_AMT_max','TR_AMT_min','TR_AMT_std',
                      'TR_DAT_sum','TR_DAT_count','TR_DAT_mean','TR_DAT_max','TR_DAT_min','TR_DAT_std',]
        gp.eval('cha_tr_d = TR_DAT_max - TR_DAT_min',inplace=True)
        gp.eval('cha_tr_d = (TR_DAT_max - TR_DAT_min)/TR_DAT_count',inplace=True)
        gp.eval('cha_tr_mq = TR_AMT_sum /(TR_DAT_max - TR_DAT_min+1)',inplace=True)
        gp.reset_index(inplace=True)
        gp['TR_CD'] = gp['TR_CD'].astype('str')
        gp_pivot = gp.pivot(index='CUST_NO',columns='TR_CD')
        for column in list(gp_pivot.columns[gp_pivot.isnull().sum()>0]):
            mean_val = gp_pivot[column].median()
            gp_pivot[column].fillna(mean_val,inplace = True)
        gp_pivot.columns = ['_'.join(col).strip() for col in gp_pivot.columns.values]
        gp_pivot.reset_index(inplace=True)
        result.append(gp_pivot)
    return result[0],result[1]

#生成训练及测试的交易信息集
def make_tr2():
    result = []
    for data_dir in [train_dir,test_dir]:
        df = pd.read_csv(os.path.join(data_dir,tr2))
        gp = df.groupby(['CUST_NO','TR_CD']).agg({'TR_AMT':['sum','count','mean','max','min','std'],
                                                  'TR_DAT':['sum','count','mean','max','min','std']})
        gp.columns.droplevel(0)
        gp.columns = ['TR_AMT_sum','TR_AMT_count','TR_AMT_mean','TR_AMT_max','TR_AMT_min','TR_AMT_std',
                      'TR_DAT_sum','TR_DAT_count','TR_DAT_mean','TR_DAT_max','TR_DAT_min','TR_DAT_std',]
        gp.eval('cha_tr_d = TR_DAT_max - TR_DAT_min',inplace=True)
        gp.eval('cha_tr_d = (TR_DAT_max - TR_DAT_min)/TR_DAT_count',inplace=True)
        gp.eval('cha_tr_mq = TR_AMT_sum /(TR_DAT_max - TR_DAT_min+1)',inplace=True)
        gp.reset_index(inplace=True)
        gp['TR_CD'] = gp['TR_CD'].astype('str')
        gp_pivot = gp.pivot(index='CUST_NO',columns='TR_CD')
        for column in list(gp_pivot.columns[gp_pivot.isnull().sum()>0]):
            mean_val = gp_pivot[column].median()
            gp_pivot[column].fillna(mean_val,inplace = True)
        gp_pivot.columns = ['_'.join(col).strip() for col in gp_pivot.columns.values]
        gp_pivot.reset_index(inplace=True)
        result.append(gp_pivot)
    return result[0],result[1]

df_train1 = df_cust_train.merge(df_cc_bal_train,on='CUST_NO',how='left')
df_test1 = df_cust_test.merge(df_cc_bal_test,on='CUST_NO',how='left')
df_train2 = df_train1.merge(df_tr_train,on='CUST_NO',how='left')
df_test2 = df_test1.merge(df_tr_test,on='CUST_NO',how='left')
df_train3 = df_train2.merge(df_asset_train,on='CUST_NO',how='left')
df_test3 = df_test2.merge(df_asset_test,on='CUST_NO',how='left')
df_train4 = df_train3.merge(df_loan_train,on='CUST_NO',how='left')
df_test4 = df_test3.merge(df_loan_test,on='CUST_NO',how='left')
df_train5 = df_train4.merge(df_cust_info_train,on='CUST_NO',how='left')
df_test5 = df_test4.merge(df_cust_info_test,on='CUST_NO',how='left')
df_train6 = df_train5.merge(df_fund_train,on='CUST_NO',how='left')
df_test6 = df_test5.merge(df_fund_test,on='CUST_NO',how='left')
df_train7 = df_train6.merge(df_idv_td_train,on='CUST_NO',how='left')
df_test7 = df_test6.merge(df_idv_td_test,on='CUST_NO',how='left')
df_train8 = df_train7.merge(df_fncg_train,on='CUST_NO',how='left')
df_test8 = df_test7.merge(df_fncg_test,on='CUST_NO',how='left')
df_train9 = df_train8.merge(df_bond_train,on='CUST_NO',how='left')
df_test9 = df_test8.merge(df_bond_test,on='CUST_NO',how='left')
df_train10 = df_train9.merge(df_cc_cust_sts_train,on='CUST_NO',how='left')
df_test10 = df_test9.merge(df_cc_cust_sts_test,on='CUST_NO',how='left')
df_train11 = df_train10.merge(df_tr_train2,on='CUST_NO',how='left')
df_test11 = df_test10.merge(df_tr_test2,on='CUST_NO',how='left')
df_train12 = df_train11.merge(df_asset_train2,on='CUST_NO',how='left')
df_test12 = df_test11.merge(df_asset_test2,on='CUST_NO',how='left')
df_train13 = df_train12.merge(df_cust_info_train2,on='CUST_NO',how='left')
df_test13 = df_test12.merge(df_cust_info_test2,on='CUST_NO',how='left')
df_train13.fillna(0,inplace = True)
df_test13.fillna(0,inplace = True)

#lgb默认参数
param={
    'boost_from_average':'false',
    'boost':'gbdt',
    'max_depth':-1,
    'tree_learner':'serial',
    'objective':'binary',
    'threads':-1,
    'learning_rate':0.005,
    'num_leaves':96
}

#特征
features = [x for x in df_train13.columns if x not in ['CUST_NO','FLAG','end_dat']]
X_train = df_train13[features]
y_train = df_train13['FLAG']
X_test = df_test13[features]

# 10折交叉训练
num_round=9999999
folds = StratifiedKFold(n_splits=6,shuffle=False,random_state=66)
oof = np.zeros(len(y_train))
df_feature_importance = pd.DataFrame()
predictions = np.zeros(len(X_test))
for fold_,(trn_idx,val_idx) in enumerate(folds.split(X_train.values ,y_train.values)):
    print('----------')
    print('fold {}'.format(fold_ + 1))
    x0,y0 = X_train.iloc[trn_idx],y_train[trn_idx]
    x1,y1 = X_train.iloc[val_idx],y_train[val_idx]
    trn_data = lgb.Dataset(x0,label=y0)
    val_data = lgb.Dataset(x1,label=y1)
    model = lgb.train(param,trn_data,valid_sets=[trn_data,val_data],valid_names=['train','valid'],num_boost_round=num_round, verbose_eval=1500,early_stopping_rounds=1500,feval=lgb_ks)
    oof[val_idx] = model.predict(X_train.iloc[val_idx],num_iteration=model.best_iteration)
    df_fold_importance = pd.DataFrame()
    df_fold_importance['feature'] = features
    df_fold_importance['importance'] = model.feature_importance()
    df_fold_importance['fold'] = fold_+1
    df_feature_importance = pd.concat([df_feature_importance,df_fold_importance],axis=0)
    predictions += model.predict(X_test,num_iteration=model.best_iteration) / folds.n_splits
print('CV score: {:<8.5f}'.format(compute_ks(y_train,oof)))

df_upload = pd.DataFrame({'cust_no':df_test13['CUST_NO'],'label':predictions})
df_upload.to_csv('/root/work/mytest/submit-0805-02.csv',index=False,header=None)

#显示特征重要性
columns = (df_feature_importance[['feature','importance']].groupby('feature').mean().sort_values(by='importance',ascending=False)[:400].index)
best_features = df_feature_importance.loc[df_feature_importance.feature.isin(columns)]
plt.figure(figsize=(25,100))
sns.barplot(x='importance',y='feature',data=best_features.sort_values(by='importance',ascending=False))
plt.title('Feature Importance')
plt.tight_layout()