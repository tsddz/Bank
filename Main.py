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
#在不知道新生成的特征有没有用，或者有很多新特征不知道选择哪一个的时候可以先把小表和TARGET融合，对新特征和FLAG进行相关性比较，用corr函数

#由于不同表格可能有相同的英文字段名，在merge中需要添加后缀（可能会有歧义），因此在建立子表过程中需要对变量重命名

# 以国债账户（YDN1_BOND)为例，建立函数make_bond()生成新的表，
# 比如BOND表中有用信息是PTPN_AMT，NET_VAL_TOT_AMT
# 表头可以写成，CUST_NO(这个一定不能动，因为以后merge的时候需要用到这个)，BOND_PTPNAMT_SUM,BOND_PTPNAMT_SUM_WEIGHTED,BOND_PTPNAMT_LAST....
# 解释一下，新的特征第一个字段是现有的表格名去掉_和开头的YDN1比如YDN1_CC_ACCT_BAL改写成CCACCTBAL_....
# 然后用一个下划线_连接第二个字段，第二个字段为该字表中的英文字段名，删掉所有_
# 然后用一个下划线_连接第三个字段，第三个字段可以显示对于同一个CUST_NO的处理，比如对于所有CUST_NO==905360325828608的PTPN_AMT进行取SUM可以定义成BOND_PTPNAMT_SUM
# 只取时间上最后一个的数值BOND_PTPNAMT_LAST，如果有用到比如加权求和，可以再添加BOND_PTPNAMT_WEIGHTED

#当然也可以对一些特征进行整合比如BOND_MATUDAT_jian_ARGCRTDAT其中用jia，jian，cheng，chu连接吧

FD.make_FNCG()



def make_bond():
    #FEATURE1:MATU_DAT - ARG_CRT_DAT
    #
    #
    print('start_bond_data_transs')
    for data_dir in [train_dir]:
        df = pd.read_csv(os.path.join(data_dir,bond+'.csv'))
        custno_list = list(set(df.CUST_NO))
        columns = ['BOND_PTPNAMT_WEIGHTED','BOND_PTPNAMT_SUM', 'BOND_PTPNAMT_LAST','BOND_PTPNAMT_MAX','BOND_PTPNAMT_MIN','BOND_PTPNAMT_STD',
                   'BOND_NETVALTOTAMT_WEIGHTED','BOND_NETVALTOTAMT_SUM','BOND_NETVALTOTAMT_LAST','BOND_NETVALTOTAMT_MAX','BOND_NETVALTOTAMT_MIN','BOND_NETVALTOTAMT_STD',
                   'BOND_FEATURE1_WEIGHTED','BOND_FEATURE1_SUM','BOND_FEATURE1_LAST']#当然可以添加好多
        data = {'CUST_NO': custno_list}
        df2 = pd.DataFrame(data)#建立一个新表
        for i in range(len(columns)):
            df2[columns[i]]='Nan'
        df.eval('cha31 = MATU_DAT - ARG_CRT_DAT',inplace=True)
        df.eval('cha32 = (ARG_LIF_CYC_STA_CD-63)', inplace=True)
        for i,j in enumerate(custno_list):
            if float(i/100)==int(i/100):
                print(i/len(custno_list))#打印进度
            kk=df[df.CUST_NO == j]
            maxDATA_DT=kk.DATA_DT.max()
            idmax=kk.DATA_DT.idxmax()
            minDATA_DT = kk.DATA_DT.min()
            diff=maxDATA_DT-minDATA_DT
            sumup=sum((np.array(kk.DATA_DT)-np.array(minDATA_DT))/diff)
            weight=(np.array(kk.DATA_DT)-np.array(minDATA_DT))/(diff*sumup)

            df2['BOND_PTPNAMT_WEIGHTED'][i]=float(sum(np.array(kk.PTPN_AMT)*weight))
            df2['BOND_PTPNAMT_SUM'][i] = float(sum(np.array(kk.PTPN_AMT)))
            df2['BOND_PTPNAMT_LAST'][i] = float(sum(kk[kk.DATA_DT==maxDATA_DT].PTPN_AMT))
            df2['BOND_PTPNAMT_MAX'][i] = float(max(np.array(kk.PTPN_AMT)))
            df2['BOND_PTPNAMT_MIN'][i] = float(min(np.array(kk.PTPN_AMT)))
            df2['BOND_PTPNAMT_STD'][i] = float(np.std(np.array(kk.PTPN_AMT)))

            df2['BOND_NETVALTOTAMT_WEIGHTED'][i] = float(sum(np.array(kk.NET_VAL_TOT_AMT) * weight))
            df2['BOND_NETVALTOTAMT_SUM'][i] = float(sum(np.array(kk.NET_VAL_TOT_AMT)))
            df2['BOND_NETVALTOTAMT_LAST'][i] = float(sum(kk[kk.DATA_DT == maxDATA_DT].NET_VAL_TOT_AMT))
            df2['BOND_NETVALTOTAMT_MAX'][i] = float(max(np.array(kk.NET_VAL_TOT_AMT)))
            df2['BOND_NETVALTOTAMT_MIN'][i] = float(min(np.array(kk.NET_VAL_TOT_AMT)))
            df2['BOND_NETVALTOTAMT_STD'][i] = float(np.std(np.array(kk.NET_VAL_TOT_AMT)))

            df2['BOND_FEATURE1_WEIGHTED'][i]=float(sum(np.array(kk.cha31) * weight))
            df2['BOND_FEATURE1_SUM'][i] = float(sum(np.array(kk.cha31)))
            df2['BOND_FEATURE1_LAST'][i] = float(sum(kk[kk.DATA_DT==maxDATA_DT].cha31))
        df2.to_csv(biuld_train_dir+'\YDN1_BOND.csv', index=False)
    return

def make_fund():
    print('start_fund_data_trans')
    for data_dir in [train_dir]:
        df = pd.read_csv(os.path.join(data_dir,fund+'.csv'))
        df=pd.read_csv('E:\Bank\DATA_ALL\YDN1_'+'FUND'+'.csv', nrows=140000)
        custno_list = list(set(df.CUST_NO))
        columnoh=['CHANL_CD','FUD_PROD_TYP_CD','RSK_RANK_CD']
        ONEHOT=[]
        OHAPPEND=[]
        for i,j in enumerate(columnoh):
            kk=pd.get_dummies(df[j])
            for ii,jj in enumerate(kk.keys()):
                OHAPPEND.append('FUND_'+j.replace('_','') +"_"+ str(jj))
                df[j.replace('_','') +"_"+ str(jj)] = pd.Series(kk[jj][:])
            del kk
            df.pop(j)
        columns = []#当然可以添加好多
        data = {'CUST_NO': custno_list}
        df2 = pd.DataFrame(data)#建立一个新表
        #df.eval('cha31 = MATU_DAT - ARG_CRT_DAT', inplace=True)
        columns = ['FUND_FUNDBAL_WEIGHTED','FUND_FUNDBAL_SUM','FUND_FUNDBAL_LAST','FUND_FUNDBAL_MAX','FUND_FUNDBAL_MIN','FUND_FUNDBAL_STD',
                   ]  # 当然可以添加好多
        columns.extend(OHAPPEND)
        for i in range(len(columns)):
            df2[columns[i]]='Nan'
        for i,j in enumerate(custno_list):
            if float(i/100)==int(i/100):
                print(i/len(custno_list))#打印进度
            kk=df[df.CUST_NO == j]
            maxDATA_DT=kk.DATA_DT.max()
            minDATA_DT = kk.DATA_DT.min()
            diff=maxDATA_DT-minDATA_DT
            sumup=sum((np.array(kk.DATA_DT)-np.array(minDATA_DT))/diff)
            weight=(np.array(kk.DATA_DT)-np.array(minDATA_DT))/(diff*sumup)

            df2['FUND_FUNDBAL_WEIGHTED'][i]=float(sum(np.array(kk.FUND_BAL)*weight))
            df2['FUND_FUNDBAL_SUM'][i] = float(sum(np.array(kk.FUND_BAL)))
            df2['FUND_FUNDBAL_LAST'][i] = float(sum(kk[kk.DATA_DT==maxDATA_DT].FUND_BAL))
            df2['FUND_FUNDBAL_MAX'][i] = float(max(np.array(kk.FUND_BAL)))
            df2['FUND_FUNDBAL_MIN'][i] = float(min(np.array(kk.FUND_BAL)))
            df2['FUND_FUNDBAL_STD'][i] = float(np.std(np.array(kk.FUND_BAL)))


            for ii,jj in enumerate(OHAPPEND):
                #df2[OHAPPEND[ii]][i]=sum(np.array(kk.FUND_BAL))
                code='df2[OHAPPEND[ii]][i]=sum(np.array(kk.'+jj[5:]+'))'
                exec(code)
        df2.to_csv(biuld_train_dir+'\YDN1_FUND.csv', index=False)
    return

def make_TR():
    for data_dir in [train_dir]:
        df = pd.read_csv(os.path.join(data_dir,tr+'.csv'))
        df=pd.read_csv('E:\Bank\DATA_ALL\YDN1_'+'TR'+'.csv', nrows=140000)
        custno_list = list(set(df.CUST_NO))
        columnoh=['TR_CD']
        ONEHOT=[]
        OHAPPEND=[]
        for i,j in enumerate(columnoh):
            kk=pd.get_dummies(df[j])
            for ii,jj in enumerate(kk.keys()):
                OHAPPEND.append('FUND_'+j.replace('_','') +"_"+ str(jj))
                df[j.replace('_','') +"_"+ str(jj)] = pd.Series(kk[jj][:])
            del kk
            df.pop(j)
        columns = []#当然可以添加好多
        data = {'CUST_NO': custno_list}
        df2 = pd.DataFrame(data)#建立一个新表
        #df.eval('cha31 = MATU_DAT - ARG_CRT_DAT', inplace=True)
        CDlist=[0,1,3,7,8,75]
        columns = ['TR_AMT_0_SUM','TR_AMT_0_MAX','TR_AMT_0_MIN','TR_AMT_0_LAST','TR_AMT_0_COUNT',
                   'TR_AMT_1_SUM','TR_AMT_1_MAX','TR_AMT_1_MIN','TR_AMT_1_LAST','TR_AMT_1_COUNT',
                   'TR_AMT_3_SUM','TR_AMT_3_MAX','TR_AMT_3_MIN','TR_AMT_3_LAST','TR_AMT_3_COUNT',
                   'TR_AMT_7_SUM','TR_AMT_7_MAX','TR_AMT_7_MIN','TR_AMT_7_LAST','TR_AMT_7_COUNT',
                   'TR_AMT_8_SUM','TR_AMT_8_MAX','TR_AMT_8_MIN','TR_AMT_8_LAST','TR_AMT_7_COUNT',
                   'TR_AMT_75_SUM','TR_AMT_75_MAX','TR_AMT_75_MIN','TR_AMT_75_LAST','TR_AMT_75_COUNT'
                   ]  # 当然可以添加好多
        for i in range(len(columns)):
            df2[columns[i]]='Nan'
        for i,j in enumerate(custno_list):
            if float(i/100)==int(i/100):
                print(i/len(custno_list))#打印进度
            kk=df[df.CUST_NO == j]
            '''
            maxTR_DAT=kk.TR_DAT.max()
            minTR_DAT = kk.TR_DAT.min()
            diff=maxTR_DAT-minTR_DAT
            sumup=sum((np.array(kk.TR_DAT)-np.array(minTR_DAT))/diff)
            weight=(np.array(kk.TR_DAT)-np.array(minTR_DAT))/(diff*sumup)
            '''
            for ii,jj in enumerate(CDlist):
                #df2['TR_AMT_'+str(jj)+'_SUM'][i]=float(sum(np.array(kk[kk.TRCD_0 == jj].TR_AMT)))
                subcode="kkk=kk[kk.TRCD_"+str(jj)+"==1]"
                exec(subcode)
                if len(kkk)==0:
                    continue
                df2['TR_AMT_' + str(jj) + '_SUM'][i] = float(sum(np.array(kkk.TR_AMT)))
                df2['TR_AMT_' + str(jj) + '_MAX'][i] = float(max(np.array(kkk.TR_AMT)))
                df2['TR_AMT_' + str(jj) + '_MIN'][i] = float(min(np.array(kkk.TR_AMT)))
                df2['TR_AMT_' + str(jj) + '_LAST'][i] = np.mean(kkk[kkk.TR_DAT == kkk['TR_DAT'].max()].TR_AMT)
                df2['TR_AMT_' + str(jj) + '_COUNT'][i] = len(kkk)
        df2.to_csv(biuld_train_dir+'\YDN1_TR.csv', index=False)
    return

def make_LOAN():
    for data_dir in [train_dir]:
        df = pd.read_csv(os.path.join(data_dir, loan + '.csv'))
       # df = pd.read_csv('E:\Bank\DATA_ALL\YDN1_' + 'LOAN' + '.csv')
        dfmerge=pd.merge(df,TRAIN_TARGET,how='outer',left_on='CUST_NO',right_on='CUST_NO')

        df.eval('cha31 = DATA_DT - ARG_CRT_DAT', inplace=True)
        df.eval('cha32 = MATU_DAT- DATA_DT', inplace=True)

        df.eval('cha41 = NML_CAP_BAL - TOT_PRVD_AMT', inplace=True)
        df.eval('cha42 = NML_CAP_BAL - TOT_REVK_AMT', inplace=True)
        df.eval('cha43 = TOT_PRVD_AMT - TOT_REVK_AMT', inplace=True)
        custno_list = list(set(df.CUST_NO))
        columns = ['LOAN_cha31_SUM','LOAN_cha31_MEAN','LOAN_cha31_MIN','LOAN_cha31_MAX','LOAN_cha31_LAST',
                   'LOAN_cha32_SUM','LOAN_cha32_MEAN','LOAN_cha32_MIN','LOAN_cha32_MAX','LOAN_cha32_LAST',
                   'LOAN_cha41_SUM', 'LOAN_cha41_MEAN', 'LOAN_cha41_MIN', 'LOAN_cha41_MAX', 'LOAN_cha41_LAST',
                   'LOAN_cha42_SUM', 'LOAN_cha42_MEAN', 'LOAN_cha42_MIN', 'LOAN_cha42_MAX', 'LOAN_cha42_LAST',
                   'LOAN_cha43_SUM', 'LOAN_cha43_MEAN', 'LOAN_cha43_MIN', 'LOAN_cha43_MAX', 'LOAN_cha43_LAST',
                   'LOAN_NMLCAPBAL_SUM','LOAN_NMLCAPBAL_MEAN','LOAN_NMLCAPBAL_MIN','LOAN_NMLCAPBAL_MAX','LOAN_NMLCAPBAL_LAST',
                   'LOAN_TOTPRVDAMT_SUM', 'LOAN_TOTPRVDAMT_MEAN', 'LOAN_TOTPRVDAMT_MIN', 'LOAN_TOTPRVDAMT_MAX','LOAN_TOTPRVDAMT_LAST',
                   'LOAN_TOTREVKAMT_SUM', 'LOAN_TOTREVKAMT_MEAN', 'LOAN_TOTREVKAMT_MIN', 'LOAN_TOTREVKAMT_MAX', 'LOAN_TOTREVKAMT_LAST',
                   'LOAN_MTHNMLCAPACCM_SUM', 'LOAN_MTHNMLCAPACCM_MEAN', 'LOAN_MTHNMLCAPACCM_MIN', 'LOAN_MTHNMLCAPACCM_MAX', 'LOAN_MTHNMLCAPACCM_LAST'
                   ]
        COLUMN=['cha31','cha32','cha41','cha42','cha43','NML_CAP_BAL','TOT_PRVD_AMT','TOT_REVK_AMT','MTH_NML_CAP_ACCM']
        data = {'CUST_NO': custno_list}
        df2 = pd.DataFrame(data)  # 建立一个新表
        for i in range(len(columns)):
            df2[columns[i]] = 'NaN'
        for i, j in enumerate(custno_list):
            if float(i / 10) == int(i / 10):
                print(i / len(custno_list))  # 打印进度
            kk = df[df.CUST_NO == j]
            maxDATA_DT = kk.DATA_DT.max()
            for ii,jj in enumerate(COLUMN):
                #df2['LOAN_' + jj.replace('_', '') + '_SUM'][i] = float(sum(np.array(kk.cha31)))
                sumcode="df2['LOAN_' + jj.replace('_', '') + '_SUM'][i] = float(sum(np.array(kk."+str(jj)+')))'
                exec(sumcode)
                #df2['LOAN_' + jj.replace('_', '') + '_MEAN'][i] = float(np.mean(np.array(kk.cha31)))
                meancode="df2['LOAN_' + jj.replace('_', '') + '_MEAN'][i] = float(np.mean(np.array(kk."+str(jj)+')))'
                exec(meancode)
                #df2['LOAN_' + jj.replace('_', '') + '_MIN'][i] = float(np.min(np.array(kk.cha31)))
                mincode="df2['LOAN_' + jj.replace('_', '') + '_MIN'][i] = float(np.min(np.array(kk."+str(jj)+')))'
                exec(mincode)
                #df2['LOAN_' + jj.replace('_', '') + '_MAX'][i] = float(np.max(np.array(kk.cha31)))
                maxcode="df2['LOAN_' + jj.replace('_', '') + '_MAX'][i] = float(np.max(np.array(kk."+str(jj)+')))'
                exec(maxcode)

                #df2['LOAN_' + jj.replace('_', '') + '_LAST'][i] = float(sum(kk[kk.DATA_DT == maxDATA_DT].cha31))
                lastcode="df2['LOAN_' + jj.replace('_', '') + '_LAST'][i] = float(sum(kk[kk.DATA_DT == maxDATA_DT]."+str(jj)+'))'
                exec(lastcode)
        df2.to_csv(biuld_train_dir + '\YDN1_LOAN.csv', index=False)
    return

def make_CC_ACCT_BAL():
    for data_dir in [train_dir]:
        df = pd.read_csv(os.path.join(data_dir, cc_bal + '.csv'))

        deal = list(df.CAN_RETURN_AMT)
        for i in range(len(deal)):
            try:
                deal[i] = float(deal[i])
            except:
                deal[i] = 0
        df.CAN_RETURN_AMT = pd.Series(deal)

        deal = list(df.TOT_AC_QUOT)
        for i in range(len(deal)):
            try:
                deal[i] = float(deal[i])
            except:
                deal[i] = float(deal[i][:6])
        df.TOT_AC_QUOT = pd.Series(deal)
       # df = pd.read_csv('E:\Bank\DATA_ALL\YDN1_' + 'LOAN' + '.csv')
        custno_list = list(set(df.CUST_NO))
        columns = ['CCACCTBAL_GENRACQUOT_SUM','CCACCTBAL_GENRACQUOT_MEAN','CCACCTBAL_GENRACQUOT_MIN','CCACCTBAL_GENRACQUOT_MAX','CCACCTBAL_GENRACQUOT_LAST',
                   'CCACCTBAL_TOTACQUOT_SUM','CCACCTBAL_TOTACQUOT_MEAN','CCACCTBAL_TOTACQUOT_MIN','CCACCTBAL_TOTACQUOT_MAX','CCACCTBAL_TOTACQUOT_LAST',
                   'CCACCTBAL_CANAMT_SUM', 'CCACCTBAL_CANAMT_MEAN', 'CCACCTBAL_CANAMT_MIN', 'CCACCTBAL_CANAMT_MAX', 'CCACCTBAL_CANAMT_LAST',
                   'CCACCTBAL_CANRETURNAMT_SUM', 'CCACCTBAL_CANRETURNAMT_MEAN', 'CCACCTBAL_CANRETURNAMT_MIN', 'CCACCTBAL_CANRETURNAMT_MAX', 'CCACCTBAL_CANRETURNAMT_LAST'
                   ]
        COLUMN=['GENR_AC_QUOT','TOT_AC_QUOT','CAN_AMT','CAN_RETURN_AMT']
        data = {'CUST_NO': custno_list}
        df2 = pd.DataFrame(data)  # 建立一个新表
        for i in range(len(columns)):
            df2[columns[i]] = 'NaN'
        for i, j in enumerate(custno_list):
            if float(i / 50) == int(i / 50):
                print(i / len(custno_list))  # 打印进度
            kk = df[df.CUST_NO == j]
            maxDATA_DAT = kk.DATA_DAT.max()
            for ii,jj in enumerate(COLUMN):
                #df2['LOAN_' + jj.replace('_', '') + '_SUM'][i] = float(sum(np.array(kk.cha31)))
                sumcode="df2['CCACCTBAL_' + jj.replace('_', '') + '_SUM'][i] = float(sum(np.array(kk."+str(jj)+')))'
                exec(sumcode)
                #df2['LOAN_' + jj.replace('_', '') + '_MEAN'][i] = float(np.mean(np.array(kk.cha31)))
                meancode="df2['CCACCTBAL_' + jj.replace('_', '') + '_MEAN'][i] = float(np.mean(np.array(kk."+str(jj)+')))'
                exec(meancode)
                #df2['LOAN_' + jj.replace('_', '') + '_MIN'][i] = float(np.min(np.array(kk.cha31)))
                mincode="df2['CCACCTBAL_' + jj.replace('_', '') + '_MIN'][i] = float(np.min(np.array(kk."+str(jj)+')))'
                exec(mincode)
                #df2['LOAN_' + jj.replace('_', '') + '_MAX'][i] = float(np.max(np.array(kk.cha31)))
                maxcode="df2['CCACCTBAL_' + jj.replace('_', '') + '_MAX'][i] = float(np.max(np.array(kk."+str(jj)+')))'
                exec(maxcode)

                #df2['LOAN_' + jj.replace('_', '') + '_LAST'][i] = float(sum(kk[kk.DATA_DT == maxDATA_DT].cha31))
                lastcode="df2['CCACCTBAL_' + jj.replace('_', '') + '_LAST'][i] = float(sum(kk[kk.DATA_DAT == maxDATA_DAT]."+str(jj)+'))'
                exec(lastcode)
        df2.to_csv(biuld_train_dir + '\YDN1_CC_ACCT_BAL.csv', index=False)
    return


def make_CC_CUST_STS():
    for data_dir in [train_dir]:
        df = pd.read_csv('E:\Bank\DATA_ALL\YDN1_'+'CC_CUST_STS'+'.csv')
        custno_list = list(set(df.CUST_NO))
        columnoh=['CUST_CYC_CR_IND','CUST_CD_VLU']
        ONEHOT=[]
        OHAPPEND=[]
        for i,j in enumerate(columnoh):
            kk=pd.get_dummies(df[j])
            for ii,jj in enumerate(kk.keys()):
                OHAPPEND.append('CCCUSTSTS_'+j.replace('_','') +"_"+ str(jj))
                df[j.replace('_','') +"_"+ str(jj)] = pd.Series(kk[jj][:])
            del kk
            df.pop(j)
        columns = []#当然可以添加好多
        data = {'CUST_NO': custno_list}
        df2 = pd.DataFrame(data)#建立一个新表
        #df.eval('cha31 = MATU_DAT - ARG_CRT_DAT', inplace=True)
        columns = []  # 当然可以添加好多
        for i in range(len(OHAPPEND)):
            columns.append(OHAPPEND[i]+'_SUM')
            columns.append(OHAPPEND[i]+'_LAST')
        for i in range(len(columns)):
            df2[columns[i]]='Nan'
        for i,j in enumerate(custno_list):
            if float(i/10)==int(i/10):
                print(i/len(custno_list))#打印进度
            kk=df[df.CUST_NO == j]
            maxDATA_DAT=kk.DATA_DAT.max()
            for ii,jj in enumerate(OHAPPEND):
                #df2[OHAPPEND[ii]][i]=sum(np.array(kk.FUND_BAL))
                code='df2[OHAPPEND[ii]+"_SUM"][i]=sum(np.array(kk.'+jj[10:]+'))'
                exec(code)
                code='df2[OHAPPEND[ii]+"_LAST"][i]=np.mean(kk[kk.DATA_DAT==maxDATA_DAT].'+jj[10:]+')'
                exec(code)
        df2.to_csv(biuld_train_dir+'\YDN1_CC_CUST_STS.csv', index=False)
    return

make_CC_ACCT_BAL()

'''










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
'''












'''
pd.get_dummies(df['B'])


for i in range(len(kk.keys()))
    df2['la_'+kk.keys()[i]]=kk[kk.keys()[i]]
df2['date'] = date
'''