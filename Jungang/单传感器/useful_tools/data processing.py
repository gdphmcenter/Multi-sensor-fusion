import os
import numpy as np
import numpy
import sys
import scipy.io as scio
import pandas as pd
from scipy.io import savemat

def read_mat(dir):
    # ='../4分类/02/归一化/X02'
    # DONE
    data = np.array([], [])
    label = np.array([])
    count = 0
    # 遍历oneD文件夹中的各个mat文件
    for fn in os.listdir('./' + dir):
        if fn.endswith('.mat'):
            # 路径
            path = './' + dir + '/' + "".join(fn)  # join()是一个字符串方法，它返回被子字符串连接的字符串
            # path = './oneD/'+"".join(fn)
            read_data = scio.loadmat(path)
            # 获得标签
            now_data_label = fn.split('_')[0]
            # print(now_data_label)
            # 获得mat的字典列表
            var_dict = list(read_data.keys())
            # 寻找DE的变量
            # for var in range(len(var_dict)):
            #     check_DE = var_dict[var].split("_")
            #     for check in check_DE:
            #         if check == 'DE':
            #             # 记录DE的位置
            #             location = var
            #             # 记录带DE的变量名
            #             var_DE = var_dict[location]
            #             break
            # 读取数据并且转置
            now_data = read_data['data'].T
            # print('now_data',now_data.shape)
            # 剔除后面
            unwanted = now_data.shape[1] % 1024
            # print('unwanted',unwanted)
            now_data = now_data[..., :-unwanted]
            # print('now_data',now_data.shape)
            # 分割数据为8192
            now_data = now_data.reshape(-1, 1024)
            # print('now_data-1',now_data.shape)
            now_data_len = now_data.shape[0]
            # print('now_data_len',now_data_len)
            # 记录标签
            for layer in range(int(now_data_len)):
                label = np.append(label, int(now_data_label))
            # 第一次记录
            if count == 0:
                data = now_data
                count += 1
                continue
            # 两次以上的记录
            data = np.vstack((data, now_data))
            # print('data',data.shape)
            count += 1
    # 返回数据集的数据和标标签
    data = data.reshape(-1, 1024, 1)
    return data, label





if __name__ == '__main__':
    pass

    dataA, labelA = read_mat('../4分类/02/归一化00/X02')
    dataB, labelB = read_mat('../4分类/02/归一化00/Y02')
    dataC, labelC = read_mat('../4分类/02/归一化00/Z02')
    print('dataA', dataA.shape)
    print('dataB', dataB.shape)
    print('dataC', dataC.shape)
    print('labelA', labelA.shape)



    data_list=[]
    E = []
    for i in range(244):
        E = np.concatenate((dataA[i], dataB[i], dataC[i]), axis=0)
        data_list.append(E)
    data_list = np.array([data_list])
    data_list1 = data_list.reshape(-1, 1)
    print(data_list1.shape)
    # file_name = '0_normal'
    # savemat('0_normal.mat',{'data': data_list1})
    # savemat('0_normal.mat',{'data': data_list1})
    # savemat('2_B_fault.mat',{'data': data_list1})
    # savemat('3_C_fault.mat',{'data': data_list1})



    # print()
    # i = 0
    # j = 100
    # lst = np.array([])
    # while i < j:
    #     data = np.concatenate((dataA[i], dataB[i], dataC[i]), axis=0)
    #     # data.sort()
    #     lst=np.append(data)
    #     i += 1
    #
    # print(lst.shape)















