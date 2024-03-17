import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap
from tqdm import tqdm

# x 维度 [N,D]
def cal_pairwise_dist(x):
    
    N,D = np.shape(x)
    
    dist = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            # dist[i,j] = np.dot((x[i]-x[j]),(x[i]-x[j]).T)         # 欧式距离 2范数平方
            dist[i,j] = np.sqrt(np.dot((x[i]-x[j]),(x[i]-x[j]).T))  # 2范数
            # dist[i,j] = np.sum(np.abs(x[i]-x[j]))                 # 1范数
    
    #返回任意两个点之间距离
    return dist
    
    
    
# 构建最短路径图  弗洛伊德算法
def floyd(D,n_neighbors=15):
    Max = np.max(D)*1000
    n1,n2 = D.shape
    k = n_neighbors  # k近邻
    D1 = np.ones((n1,n1))*Max
    D_arg = np.argsort(D,axis=1)
    for i in range(n1):
        D1[i,D_arg[i,0:k+1]] = D[i,D_arg[i,0:k+1]]
    for k in tqdm(range(n1)):
        
        for i in range(n1):
            for j in range(n1):
                if D1[i,k]+D1[k,j]<D1[i,j]:
                    D1[i,j] = D1[i,k]+D1[k,j]
    return D1


def my_mds(dist, n_dims):
    # dist (n_samples, n_samples)
    dist = dist**2
    n = dist.shape[0]
    T1 = np.ones((n,n))*np.sum(dist)/n**2
    T2 = np.sum(dist, axis = 1)/n
    T3 = np.sum(dist, axis = 0)/n

    B = -(T1 - T2 - T3 + dist)/2

    eig_val, eig_vector = np.linalg.eig(B)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]

    return picked_eig_vector*picked_eig_val**(0.5)

    
 
# dist N*N 距离矩阵样本点两两之间的距离 
# n_dims 降维
# 返回 降维后的数据
# def my_mds(dist, n_dims):
    
    # n,n = np.shape(dist)
    
    # dist[dist < 0 ] = 0
    
    # T1 = np.ones((n,n))*np.sum(dist)/n**2
    # T2 = np.sum(dist, axis = 1, keepdims=True)/n
    # T3 = np.sum(dist, axis = 0, keepdims=True)/n

    # B = -(T1 - T2 - T3 + dist)/2

    # eig_val, eig_vector = np.linalg.eig(B)
    # index_ = np.argsort(-eig_val)[:n_dims]
    # picked_eig_val = eig_val[index_].real
    # picked_eig_vector = eig_vector[:, index_]
    
    # return picked_eig_vector*picked_eig_val**(0.5)

def my_Isomap(D,n=10,n_neighbors=30):

    D_floyd=floyd(D, n_neighbors)
    data_n = my_mds(D_floyd, n_dims=n)
    return data_n



def scatter_3d(X, y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    # plt.show(block=False)
    plt.savefig('./picture/ISOMAP/三维fuse.png')
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
    
    
    # X, Y = make_s_curve(n_samples = 500,
    #                        noise = 0.1,
    #                        random_state = 42)
    # scatter_3d(X,Y)
    
    # 计算距离
    # dist = cal_pairwise_dist(X)
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
    # scatter_3d(data,labs) # 三维图
    # 计算距离
    dist = cal_pairwise_dist(data)
    print(dist.shape)

    # # MDS 降维
    # data_MDS = my_mds(dist, 2)
    #
    # plt.figure()
    # plt.title("my_MSD")
    # plt.scatter(data_MDS[:, 0], data_MDS[:, 1], c = labs)
    # plt.show(block=False)
    
    
    # ISOMAP 降维
    data_ISOMAP = my_Isomap(dist, 10, 10)
   
    plt.figure()
    plt.title("my_Isomap")
    plt.scatter(data_ISOMAP[:, 0], data_ISOMAP[:, 1], c = labs)
    # plt.show(block=False)
    plt.savefig('../picture/ISOMAP/Fuse.png')
    plt.show()

    # data_ISOMAP2 = Isomap(n_neighbors = 10, n_components = 2).fit_transform(X)
    #
    # plt.figure()
    # plt.title("sk_Isomap")
    # plt.scatter(data_ISOMAP2[:, 0], data_ISOMAP2[:, 1], c = Y)
    # plt.show(block=False)
    #
    # plt.show()
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # 加载数据
    # data = np.loadtxt("iris.data",dtype="str",delimiter=',')
    # feas = data[:,:-1]
    # feas = np.float32(feas)
    # labs = data[:,-1]
    
    # # 计算距离
    # dist = cal_pairwise_dist(feas)
    
    # # 进行降维
    # data_2d = my_mds(dist, 2)
    
    # #绘图
    # draw_pic(data_2d,labs)
    
    
    