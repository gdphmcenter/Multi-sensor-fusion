import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 定义核函数 计算两个矢量的相关性
# Sigmoid核
def sigmoid(x1, x2,a = 0.25,r=3):
    x = np.dot(x1, x2)
    return np.tanh(a*x+r)
# 多项式核
def linear(x1,x2,a=1,c=0,d=1):
    x = np.dot(x1, x2)
    x = np.power((a*x+c),d)
    return x
# 高斯核/RBF（径向基）
def rbf(x1,x2,gamma = 0.1):
    x = np.dot((x1-x2),(x1-x2))
    x = np.exp(-gamma*x)
    return x




    
def kpca(data, n_dims=6, kernel = linear):  # data为数据 n_dims为降维数 kernel为核函数
    
    N,D = np.shape(data)
    K = np.zeros([N,N])  # K为样本两两之间的相关性
    
    # 利用核函数计算K
    for i in range(N):
        for j in range(N):
            K[i,j]=kernel(data[i],data[j])
    
    # 对K进行中心化
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    #计算特征值和特征向量
    eig_values, eig_vector = np.linalg.eig(K)
    idx = np.argsort(-eig_values)[:n_dims] # 从大到小排序
   
    # 选取较大的特征值
    eigval = eig_values[idx]
    eigvector = eig_vector[:, idx]  #[N,d]
    
    # 进行正则
    eigval = eigval**(1/2)
    u = eigvector/eigval.reshape(-1,n_dims)  # u [N,d]
    
    # 进行降维
    data_n = np.dot(K, u)  # [N,N]*[N,d]=[N,d]
    # labels = labs
    # within_the_class_distance(data_n, labels, class_range_number=4)
    return data_n  # 得到降维之后的数据




def draw_pic(datas,labs):
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
    # plt.savefig('../picture/KPCA/linear.png')
    plt.savefig('../picture/KPCA/linear2.png')
    plt.show()


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
    
    # # 加载数据
    # data = np.loadtxt("iris.data",dtype="str",delimiter=',')
    # feas = data[:,:-1]
    # feas = np.float32(feas)
    # labs = data[:,-1]
    # enc = preprocessing.LabelEncoder()
    # enc = enc.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    # labs = enc.transform(labs)
    # kpca(feas)
    #
    # # 进行降维
    # data_2d = kpca(feas, n_dims=2, kernel=rbf)
    #
    # #绘图
    # draw_pic(data_2d,labs)

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
    kpca(data)
    # # 进行降维
    data_2d = kpca(data, n_dims=2, kernel=linear)
    within_the_class_distance(data_2d, labs, class_range_number=4)
    print(data_2d.shape)
    # 绘图
    draw_pic(data_2d, labs)

    
    
    
    # a1 = np.random.rand(100, 2)
    # a1 =a1+np.array([4,0])
    #
    # a2 = np.random.rand(100, 2)
    # a2 =a2+np.array([-4,0])
    #
    # a = np.concatenate((a1,a2),axis=0)
    #
    # b = np.random.rand(200, 2)
    #
    # data = np.concatenate((a,b),axis=0)
    # labs =np.concatenate((np.zeros(200),np.ones(200)),axis=0)
    # draw_pic(data,labs)
    #
    #
    # data_2d = kpca(data, n_dims=2,kernel=rbf)
    #
    # draw_pic(data_2d,labs)