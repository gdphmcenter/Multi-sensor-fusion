# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, load_iris
from sklearn.datasets import make_swiss_roll


# x 维度 [N,D]
def cal_pairwise_dist(X):  # n样本点的两两欧氏距离
    
    N,D = np.shape(X)

    tile_xi = np.tile(np.expand_dims(X,1),[1,N,1])
    tile_xj = np.tile(np.expand_dims(X,axis=0),[N,1,1])
    
    dist = np.sum((tile_xi-tile_xj)**2,axis=-1)

    #返回任意两个点之间距离
    return dist
    
    
# 径向基函数
def rbf(dist, t = 1.0):
    '''
    rbf kernel function
    '''
    return np.exp(-(dist/t))



def cal_rbf_dist(data, n_neighbors, t = 1):

    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    N = dist.shape[0]
    rbf_dist = rbf(dist, t)

    W = np.zeros([N, N])
    for i in range(N):
        index_ = np.argsort(dist[i])[1:1 + n_neighbors]  # 从小到大进行排序 去掉自己
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W

# X 输入高维数据 格式 [N，D]
# n_neighbors K近邻的数目
# t 权重计算的参数
def lpp(X,n_dims,n_neighbors, t = 1.0):  # X是数据 n_dims是降维 n_neighbors计算KNN邻近的的个数 t为权重计算时的参数
    
    N = X.shape[0]
    W = cal_rbf_dist(X, n_neighbors, t)
    D = np.zeros_like(W)

    for i in range(N):
        D[i,i] = np.sum(W[i])

    L = D - W
    XDXT = np.dot(np.dot(X.T, D), X)
    XLXT = np.dot(np.dot(X.T, L), X)

    eig_val, eig_vec = np.linalg.eig(np.dot(np.linalg.pinv(XDXT), XLXT))

    sort_index_ = np.argsort(np.abs(eig_val))
    eig_val = eig_val[sort_index_]
    print("eig_val[:10]", eig_val[:10])

    j = 0
    while eig_val[j] < 1e-6:
        j+=1

    print("j: ", j)

    sort_index_ = sort_index_[j:j+n_dims]
  
    eig_val_picked = eig_val[j:j+n_dims]
    print(eig_val_picked)
    A = eig_vec[:, sort_index_]

    Y = np.dot(X, A)

    return Y

def scatter_3d(X, y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    # plt.show(block=False)
    plt.savefig('./picture/LPP/三维.png')
    plt.show()


if __name__ == '__main__':
    # #1 测试瑞士卷数据
    # X, Y = make_swiss_roll(n_samples=1000)
    # scatter_3d(X, Y)
    # n_neighbors = 10

    # # 2 测试 load_digits 数据
    # X = load_digits().data
    # Y = load_digits().target
    # n_neighbors = 5
    
    
    # # #3 测试  load_iris 数据
    # X = load_iris().data
    # Y = load_iris().target
    # n_neighbors = 10

    # 加载数据
    # 加载数据
    dataX = np.load('../feature/Xfeature/X02data.npy')
    dataY = np.load('../feature/Yfeature/Y02data.npy')
    dataZ = np.load('../feature/Zfeature/Z02data.npy')
    data = np.hstack((dataX, dataY, dataZ))
    # data = np.hstack((dataX, dataY))
    # print(data.shape)
    # data = np.hstack((dataX, dataY))
    labs = np.load('../feature/Xfeature/X02label.npy')
    # labsY = np.load('.\\feature\\Yfeature\\Y02label.npy')
    # labsZ = np.load('.\\feature\\Zfeature\\Z02label.npy')
    n_neighbors = 5
    dist = cal_pairwise_dist(data)
    # print(dist.shape)
    max_dist = np.max(dist)

    # data_2d_LPP = lpp(data, n_neighbors = n_neighbors, t = 0.01*max_dist)
    # data_2d_PCA = PCA(n_components=2).fit_transform(data)


    # plt.figure(figsize=(12,6))
    # plt.subplot(121)
    # plt.title("LPP")
    # plt.scatter(data_2d_LPP[:, 0], data_2d_LPP[:, 1], c = labs)
    # # plt.savefig('../picture/LPP/LPPfuse.png')
    # # plt.show()
    #
    # plt.subplot(122)
    # plt.title("PCA")
    # plt.scatter(data_2d_PCA[:, 0], data_2d_PCA[:, 1], c = labs)
    # # plt.savefig('../picture/LPP/PCAfuse.png')
    # plt.show()
    
    # 进行降维
    data_2d_LPP = lpp(data, n_dims=10, n_neighbors = 10, t = 0.01*max_dist)
    dimA = data_2d_LPP
    # print(data_2d_LPP.shape)
    print(dimA.shape)
    data_2d_LPP = lpp(data, n_dims=20, n_neighbors = 10, t = 0.01*max_dist)
    dimB = data_2d_LPP
    # print(dimB.shape)
    data_2d_LPP = lpp(data, n_dims=30, n_neighbors = 10, t = 0.01*max_dist)
    dimC = data_2d_LPP
    # print(dimC.shape)
    data_2d_LPP = lpp(data, n_dims=40, n_neighbors = 10, t = 0.01*max_dist)
    dimD = data_2d_LPP
    # print(dimD.shape)
    # 绘图
    plt.subplot(2,2,1)
    plt.title("dimA=10")
    plt.scatter(dimA[:, 0], dimA[:, 1], c = labs)  # dimA[:, 2],, marker='+'

    plt.subplot(2,2,2)
    plt.title("dimB=20")
    plt.scatter(dimB[:, 0], dimB[:, 1], c = labs)  # , marker='v'

    plt.subplot(2,2,3)
    plt.title("dimC=30")
    plt.scatter(dimC[:, 0], dimC[:, 1], c = labs)
    # plt.show()

    plt.subplot(2,2,4)
    plt.title("dimD=40")
    plt.scatter(dimD[:, 0], dimD[:, 1], c = labs)
    plt.show()
    
    
