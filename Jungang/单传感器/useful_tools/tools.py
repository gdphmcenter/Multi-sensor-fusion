from useful_tools.model import network_encoder
from useful_tools.data_util import MatHandler

from  matplotlib.colors import  rgb2hex
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.metrics import silhouette_score
from collections import  Counter

import keras
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


def plot_with_labels(train, labels, clusters_number=4):
    """
    绘制聚类图
    DONE
    """
    # 设置字体
    from matplotlib import rcParams
    config = {
        "font.family":'Times New Roman',  # 设置字体类型
        # "font.size": 80,
    }
    rcParams.update(config)

    fig, ax = plt.subplots()
    np.random.seed(0)
    colors = tuple([(np.random.random(),np.random.random(), np.random.random()) for i in range(clusters_number)])
    colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex

    for i, color in enumerate(colors):
        need_idx = np.where(labels == i)[0]
        ax.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i)
    # legend = ax.legend(loc='BestOutside')
    legend = ax.legend(loc=[1, 0])
    # plt.legend(loc=[1, 0])
    plt.savefig('./picture/3#-1-1.png')
    plt.show()


def within_the_class_distance(data_twoD, labels,class_range_number):
    """
    计算类内距离和类间距离
    DONE
    """
    # 分类 + 计算各个类中的类内距离
    # 类内距离(intra-class)
    intra_class_distance = []
    # 类间距离(inter-class)
    inter_class_distance = []
    print("-" * 80)
    for class_lable in range(class_range_number):
        intra_class_distance.append(account_class_distance(data_twoD[np.where(labels == class_lable)], class_lable))
    print("-" * 80)
    # 分类 + 计算各个类中的类间距离
    print("-" * 80)
    for class_lable_1 in range(class_range_number):
        for class_lable_2 in range(class_lable_1, class_range_number):
            if class_lable_1 == class_lable_2:
                continue
            first_data_twoD = data_twoD[np.where(labels == class_lable_1)]
            second_data_twoD = data_twoD[np.where(labels == class_lable_2)]
            inter_class_distance.append(account_between_class_distance(first_data_twoD, second_data_twoD, class_lable_1, class_lable_2))
    print("-" * 80)
    # 计算类内距离的均值和标准差(无偏)
    intra_class_distance_mean = np.mean(intra_class_distance)
    intra_class_distance_std = np.std(intra_class_distance, ddof=1)
    print("类内距离的均值:" + str(intra_class_distance_mean))
    print("类内距离的标准差(无偏):" + str(intra_class_distance_std))
    # 计算类间距离的均值和标准差
    inter_class_distance_mean = np.mean(inter_class_distance)
    inter_class_distance_std = np.std(inter_class_distance, ddof=1)
    print("类间距离的均值:" + str(inter_class_distance_mean))
    print("类间距离的标准差(无偏):" + str(inter_class_distance_std))


def two_spot_distance(first_twoD, second_twoD):
    """
    两点之间的距离
    DONE
    """
    # DONE
    x1, y1 = first_twoD
    x2, y2 = second_twoD
    dist = (x1 - x2)**2 + (y1 - y2)**2
    return dist**0.5


def account_class_distance(data_twoD, label):
    """
    计算类内距离
    DONE
    """
    two_spot_distance_sum = 0
    for i in range(len(data_twoD)):
        for j in range(len(data_twoD)):
            if i == j:
                continue
            two_spot_distance_sum += two_spot_distance(data_twoD[i], data_twoD[j])
    ans = two_spot_distance_sum/(len(data_twoD)*len(data_twoD))
    print("类" + str(label) + "的类内距离为:" + str(ans))
    return ans


def account_between_class_distance(first_data_twoD, second_data_twoD, label_1, lable_2):
    """
    计算类间距离
    DONE
    """
    two_spot_distance_sum = 0
    for i in range(len(first_data_twoD)):
        for j in range(len(second_data_twoD)):
            two_spot_distance_sum += two_spot_distance(first_data_twoD[i], second_data_twoD[j])
    ans = two_spot_distance_sum/(len(first_data_twoD)*len(second_data_twoD))
    print("类" + str(label_1) + "和类" + str(lable_2) + "的类间距离为:" + str(ans))
    return ans


def check_label(label ,result_list, result_num):
    list_Counter = []
    confusion = np.zeros((4,4))

    for i in range(result_num):
        list_Counter.append(Counter(label[result_list == i]))

    for i in range(result_num):
        print("第" + str(i) + "类：")
        sum = 0
        class_num = 0.0
        for j in range(result_num):
            sum += list_Counter[i][class_num] 
            class_num += 1

        class_num = 0.0
        for j in range(result_num):
            if list_Counter[i][class_num] != 0:
                print("类" + str(class_num) + "的百分比：" + str(list_Counter[i][class_num]/sum))
                confusion[i][int(class_num)] = list_Counter[i][class_num]/sum
            class_num += 1
    confusion = np.around(confusion, 2)
    return confusion, list_Counter

