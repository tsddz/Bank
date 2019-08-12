import Config
import pandas as pd
import numpy as np
import os


def make_FNCG():
    print('start_FNCG_data_transs')
    subdirect='E:\Bank\TRY_MERGE'
    df = pd.read_csv('E:\BANK\DATA_ALL\YDN1_FNCG.csv')
    custno_list = list(set(df.CUST_NO))
    block=0
    ilocslice=[]
    num=0
    for i, j in enumerate(custno_list):
        if block==201:
            kk=df.iloc[ilocslice]
            kk.to_csv(subdirect+"\FNCG_"+str(num)+'.csv', index=False)
            num+=1
            print(str(num)+'*50 finished')
            block=0
            break
        mm=df[df.CUST_NO==j]
        ilocslice.extend(list(mm.index))
        block+=1
    kk = df.iloc[ilocslice]
    kk.to_csv(subdirect + "\FNCG_" + str(num) + '.csv', index=False)
    del kk
    print('last finished')
    filelist=os.listdir(subdirect)
    del df
    result = pd.DataFrame()
    for filei,filename in enumerate(filelist):
        df = pd.read_csv(subdirect+'\\'+filename)
        deal = list(df.MATU_DAT)
        for i in range(len(deal)):
            try:
                deal[i] = float(deal[i])
            except:
                deal[i] = float('NaN')
        df.MATU_DAT = pd.Series(deal)

        deal = list(df.ARG_CRT_DAT)
        for i in range(len(deal)):
            try:
                deal[i] = float(deal[i])
            except:
                deal[i] = float('NaN')
        df.ARG_CRT_DAT = pd.Series(deal)

        deal = list(df.DATA_DT)
        for i in range(len(deal)):
            try:
                deal[i] = float(deal[i])
            except:
                deal[i] = float('NaN')
        df.DATA_DT = pd.Series(deal)

        deal = list(df.CLS_ACCT_DAT)
        for i in range(len(deal)):
            try:
                deal[i] = float(deal[i])
            except:
                deal[i] = float('NaN')
        df.CLS_ACCT_DAT = pd.Series(deal)

        deal = list(df.EXIT_SHR)
        for i in range(len(deal)):
            try:
                deal[i] = float(deal[i])
            except:
                deal[i] = float('NaN')
        df.EXIT_SHR = pd.Series(deal)


        deal = list(df.CUST_IVST_CST)
        for i in range(len(deal)):
            try:
                deal[i] = float(deal[i])
            except:
                deal[i] = float('NaN')
        df.CUST_IVST_CST = pd.Series(deal)



        custno_list = list(set(df.CUST_NO))
        df.eval('cha31 = MATU_DAT - DATA_DT', inplace=True)
        df.eval('cha32 = ARG_CRT_DAT - DATA_DT', inplace=True)
        df.eval('cha33 = CLS_ACCT_DAT - DATA_DT', inplace=True)
        df.eval('cha34 = MATU_DAT - ARG_CRT_DAT', inplace=True)
        column1=['FNCG_cha31_MAX','FNCG_cha32_MAX','FNCG_cha33_MAX','FNCG_cha34_MAX',
                 'FNCG_cha31_MEAN','FNCG_cha32_MEAN','FNCG_cha33_MEAN','FNCG_cha34_MEAN']
        column2=['FNCG_EXITSHR_MAX','FNCG_EXITSHR_MIN','FNCG_EXITSHR_MEAN','FNCG_EXITSHR_SUM','FNCG_EXITSHR_LAST',
                 'FNCG_CUSTIVSTCST_MAX', 'FNCG_CUSTIVSTCST_MIN', 'FNCG_CUSTIVSTCST_MEAN', 'FNCG_CUSTIVSTCST_SUM', 'FNCG_CUSTIVSTCST_LAST']
        columnoh = ['PROD_CLS_CD', 'PROD_RSK_RANK_CD','PROD_PFT_TYP_CD']
        OHAPPEND = []
        for i, j in enumerate(columnoh):
            kk = pd.get_dummies(df[j])
            for ii, jj in enumerate(kk.keys()):
                OHAPPEND.append('FNCG_' + j.replace('_', '') + "_" + str(jj).replace('.', ''))
                df[j.replace('_', '') + "_" + str(jj).replace('.','')] = pd.Series(kk[jj][:])
            del kk
            df.pop(j)
        data = {'CUST_NO': custno_list}
        df2 = pd.DataFrame(data)  # 建立一个新表
        # df.eval('cha31 = MATU_DAT - ARG_CRT_DAT', inplace=True)
        columns = []  # 当然可以添加好多
        columns+=column1
        columns+=column2
        for i in range(len(OHAPPEND)):
            columns.append(OHAPPEND[i] + '_SUM')
            columns.append(OHAPPEND[i] + '_LAST')
        for i in range(len(columns)):
            df2[columns[i]]='Nan'
        for i,j in enumerate(custno_list):
            if float(i/10)==int(i/10):
                print(i/len(custno_list))#打印进度
            kk=df[df.CUST_NO == j]
            maxDATA_DT=kk.DATA_DT.max()

            df2['FNCG_cha31_MAX'][i]=  max(kk.cha31)
            df2['FNCG_cha32_MAX'][i] = max(kk.cha32)
            df2['FNCG_cha33_MAX'][i] = max(kk.cha33)
            df2['FNCG_cha34_MAX'][i] = max(kk.cha34)
            df2['FNCG_cha31_MEAN'][i] = np.mean(kk.cha31)
            df2['FNCG_cha32_MEAN'][i] = np.mean(kk.cha32)
            df2['FNCG_cha33_MEAN'][i] = np.mean(kk.cha33)
            df2['FNCG_cha34_MEAN'][i] = np.mean(kk.cha34)

            df2['FNCG_EXITSHR_MAX'][i]=max(kk.EXIT_SHR)
            df2['FNCG_EXITSHR_MIN'][i] = min(kk.EXIT_SHR)
            df2['FNCG_EXITSHR_MEAN'][i] = np.mean(kk.EXIT_SHR)
            df2['FNCG_EXITSHR_SUM'][i] = sum(kk.EXIT_SHR)
            df2['FNCG_EXITSHR_LAST'][i] = np.mean(kk[kk.DATA_DT==maxDATA_DT].EXIT_SHR)

            df2['FNCG_CUSTIVSTCST_MAX'][i] = max(kk.CUST_IVST_CST)
            df2['FNCG_CUSTIVSTCST_MIN'][i] = min(kk.CUST_IVST_CST)
            df2['FNCG_CUSTIVSTCST_MEAN'][i] = np.mean(kk.CUST_IVST_CST)
            df2['FNCG_CUSTIVSTCST_SUM'][i] = sum(kk.CUST_IVST_CST)
            df2['FNCG_CUSTIVSTCST_LAST'][i] = np.mean(kk[kk.DATA_DT == maxDATA_DT].CUST_IVST_CST)

            for ii,jj in enumerate(OHAPPEND):
                #df2[OHAPPEND[ii]][i]=sum(np.array(kk.FUND_BAL))
                code='df2[OHAPPEND[ii]+"_SUM"][i]=sum(np.array(kk.'+jj[5:]+'))'
                exec(code)
                code='df2[OHAPPEND[ii]+"_LAST"][i]=np.mean(kk[kk.DATA_DT==maxDATA_DT].'+jj[5:]+')'
                exec(code)
        result.append([df2], ignore_index=True)
    result.to_csv(biuld_train_dir+'\YDN1_CC_CUST_STS.csv', index=False)
    return









def pre_deal(df):
    deal = list(df.MATU_DAT)
    for i in range(len(deal)):
        try:
            deal[i] = float(deal[i])
        except:
            deal[i] = float('NaN')
    df.MATU_DAT = pd.Series(deal)

    deal = list(df.ARG_CRT_DAT)
    for i in range(len(deal)):
        try:
            deal[i] = float(deal[i])
        except:
            deal[i] = float('NaN')
    df.ARG_CRT_DAT = pd.Series(deal)

    deal = list(df.DATA_DT)
    for i in range(len(deal)):
        try:
            deal[i] = float(deal[i])
        except:
            deal[i] = float('NaN')
    df.DATA_DT = pd.Series(deal)

    deal = list(df.CLS_ACCT_DAT)
    for i in range(len(deal)):
        try:
            deal[i] = float(deal[i])
        except:
            deal[i] = float('NaN')
    df.CLS_ACCT_DAT = pd.Series(deal)
    return df
