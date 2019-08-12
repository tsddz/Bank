import numpy as np
import pandas as pd
import os
import csv

#data_txt=np.loadtxt('../Data/YDN1_TARGET.txt')
#data_txtDF=pd.DataFrame(data_txt)
#print(data_txtDF.head())

direct='../Test_Data/'
txtlist=os.listdir(direct)
direct_csv='../Data_CSV_SMALL/'
count = 0
for i in range(len(txtlist)):
    txtname=txtlist[i]
    name=txtname[:-4]
    print(count)
    print(txtname)
    print('\n')

    csv_name=direct_csv+name+'.csv'
    txt_name=direct+name+'.txt'

    with open(csv_name, 'w', newline='') as writecsv:
        f_csv = csv.writer(writecsv)
        with open(txt_name, 'rb') as filein:
            count = 0
            for line in filein:
                liststr = str(line)
                liststr = liststr[2:-5]
                line_list = liststr.split(',')
                if count == 20:
                    break
                count += 1
#                for i in range(len(line_list)):
#                    if line_list[i]=='':
#                        line_list[i]='NA'
                f_csv.writerow(line_list)
    writecsv.close()

'''
outscv=csv.writer(open('../Data_CSV/YDN1_TARGET.csv','wb'))
with open('../Data_CSV/YDN1_TARGET.csv','w',newline='') as writecsv:
    f_csv=csv.writer(writecsv)
    with open('../Data/YDN1_TARGET.txt', 'rb') as filein:
        count = 0
        for line in filein:
            liststr = str(line)
            print(liststr)
            liststr = liststr[2:-5]
            line_list = liststr.split(',')
            if count == 1000:
                break
            count += 1
            f_csv.writerow(line_list)


'''




