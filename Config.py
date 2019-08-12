import os
import numpy as np
import pandas as pd
import FNCG_DEAL as FD
import scipy

train_dir='E:\Bank\DATA_ALL'
test_dir='E:\Bank\DATA_TEST'

biuld_train_dir='E:\Bank\DATA_READY\DATA_TRAIN'#把新的表格放在这里
biuld_test_dir='E..'#把新的表格放在这里

target = 'YDN1_TARGET'
#金融性交易信息
tr = 'YDN1_TR'
ratio=10
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

TRAIN_TARGET=pd.read_csv('E:\Bank\DATA_ALL\YDN1_TARGET.csv')