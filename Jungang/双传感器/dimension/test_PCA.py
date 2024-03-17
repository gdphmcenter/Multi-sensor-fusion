import numpy as np
import matplotlib.pyplot as plt
# data 输入数据  维度 [N，D]  N是样本数目 D是样本维度
# n_dim: 降维后的维度
# 返回 [N,n_dim]
def pca(data, n_dim):
    
    N,D = np.shape(data)
    
    data = data - np.mean(data, axis = 0, keepdims = True)  # (中心化)对输入的数据减取均值（沿着0的维度） 1*D的维度  X

    C = np.dot(data.T, data)/(N-1)  # [D,D]  C
    
    # 计算特征值和特征向量
    eig_values, eig_vector = np.linalg.eig(C)
    
    # 将特征值进行排序选取 n_dim 个较大的特征值
    indexs_ = np.argsort(-eig_values)[:n_dim]  # 加负号变为升序
    
    # 选取相应的特征向量组成降维矩阵
    picked_eig_vector = eig_vector[:, indexs_]  # [D,n_dim]
    
    # 对数据进行降维
    data_ndim = np.dot(data, picked_eig_vector)  # 对数据X与降维矩阵相乘
    return data_ndim, picked_eig_vector  # 返回降维的数据 用于降维的矩阵

def draw_pic(datas,labs):  # 绘图颜色
    plt.cla()
    unque_labs = np.unique(labs)
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1,len(unque_labs))]
    p=[]
    legends = []
    for i in range(len(unque_labs)):
        index = np.where(labs==unque_labs[i])
        pi = plt.scatter(datas[index, 0], datas[index, 1], c =[colors[i]] )
        p.append(pi)
        legends.append(unque_labs[i])
    
    plt.legend(p, legends)
    # plt.savefig('../picture/PCA/Fuse.png')
    # plt.show()
    
def within_the_class_distance(data_twoD, labels, class_range_number):
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
    # np.save(("类" + str(label) + "的类内距离为:" + str(ans)))
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
    
    


if __name__ == "__main__":
    
    # 加载数据
    # data = np.loadtxt("iris.data",dtype="str",delimiter=',')
    # feas = data[:,:-1]
    # feas = np.float32(feas)
    # labs = data[:,-1]
    # 加载数据
    dataX = np.load('../feature/Xfeature/X02data.npy')
    dataY = np.load('../feature/Yfeature/Y02data.npy')
    dataZ = np.load('../feature/Zfeature/Z02data.npy')
    data = np.hstack((dataX, dataY, dataZ))
    print(data.shape)
    # data = np.hstack((dataX, dataY))
    labs = np.load('../feature/Xfeature/X02label.npy')
    # labsY = np.load('.\\feature\\Yfeature\\Y02label.npy')
    # labsZ = np.load('.\\feature\\Zfeature\\Z02label.npy')
    # 进行降维
    data_2d, picked_eig_vector = pca(data, 10)
    dimA = data_2d
    # print(dimA.shape)
    # print(dimA)
    data_2d, picked_eig_vector = pca(data, 20)
    dimB = data_2d
    # print(dimB.shape)
    # print(dimB)
    data_2d, picked_eig_vector = pca(data, 30)
    dimC = data_2d
    print(dimC.shape)
    print(dimC)
    A = dimC[:,0]
    B = dimC[:,1]
    print(dimC[:,0])
    print(dimC[:,1])
    data_2d, picked_eig_vector = pca(data, 2)
    dimD = data_2d
    # print(dimD.shape)
    # print(dimD)
    # 绘图
    # plt.figure(figsize=(12,6))
    plt.subplot(2,2,1)
    # plt.subplot(2,2)
    plt.title("dimA=10")
    plt.scatter(dimA[:, 0], dimA[:, 9], c = labs, marker='+')  # dimA[:, 2],

    plt.subplot(2,2,2)
    plt.title("dimA=20")
    plt.scatter(dimB[:, 0], dimB[:, 19], c = labs, marker='v')

    plt.subplot(2,2,3)
    plt.title("dimA=30")
    plt.scatter(dimC[:, 0], dimC[:, 29], c = labs)
    # plt.show()

    plt.subplot(2,2,4)
    plt.title("dimA=2")
    plt.scatter(dimD[:, 0], dimD[:, 1], c = labs)
    plt.show()


    # plt.scatter(dimB[:, 0], dimB[:, 1], c = labs)
    # plt.scatter(dimC[:, 0], dimB[:, 1], c = labs)



    # np.save('../Fuse/PCA/PCAfuse1.npy', data_2d)
    # within_the_class_distance(data_2d, labs, class_range_number=4)
    
    #绘图
    # draw_pic(data_2d,labs)
    
    
   
    
    
    
    
    
    
    
    