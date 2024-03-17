import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm
from sklearn.manifold import LocallyLinearEmbedding

# x 维度 [N,D]
# 求任意两点的距离（欧式距离）
def cal_pairwise_dist(x):
    
    N,D = np.shape(x)
    
    dist = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            dist[i,j] = np.sqrt(np.dot((x[i]-x[j]),(x[i]-x[j]).T))  # 欧式距离

    #返回任意两个点之间距离
    return dist


# 获取每个样本点的 n_neighbors个临近点的位置
def get_n_neighbors(data, n_neighbors = 10):
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    N = dist.shape[0]
    Index = np.argsort(dist,axis=1)[:,1:n_neighbors+1]  # 从第一个开始
    return Index  # 返回n个临近点的位置

# data : N,D 给定数据 数据形状为N*D N为样本的点数  D为特征的维度
def lle(data, n_dims, n_neighbors):
    N,D = np.shape(data)
    if n_neighbors > D:  # 阈值
        tol = 1e-3
    else:
        tol = 0
    # 获取 n_neighbors个临界点的位置
    Index_NN = get_n_neighbors(data,n_neighbors)
    
    # 计算重构权重
    w = np.zeros([N,n_neighbors])
    for i in range(N):
        
        X_k = data[Index_NN[i]]  #[k,D]  k个临近点
        X_i = [data[i]]       #[1,D]     i为中心点
        I = np.ones([n_neighbors,1])  # [K,1]的矩阵
        
        Si = np.dot((np.dot(I,X_i)-X_k), (np.dot(I,X_i)-X_k).T)
        
        # 为防止对角线元素过小
        Si = Si+np.eye(n_neighbors)*tol*np.trace(Si)  # eye为对角线元素  trace(Si)为Si对角线元素的和
        
        Si_inv = np.linalg.pinv(Si)  # 求Si的逆矩阵
        w[i] = np.dot(I.T,Si_inv)/(np.dot(np.dot(I.T,Si_inv),I))  # 求出w[i]
     
    # 计算 W  由w[i]求出W
    W = np.zeros([N,N])
    for i in range(N):
        W[i,Index_NN[i]] = w[i]  # 将w[i]填充为W
 
    I_N = np.eye(N)       
    C = np.dot((I_N-W).T,(I_N-W))

    # 进行特征值的分解
    eig_val, eig_vector = np.linalg.eig(C)
    
    index_ = np.argsort(eig_val)[1:n_dims+1]
    
    y = eig_vector[:,index_]
    return y

# 3D绘图的一个函数
def scatter_3d(X, y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    # plt.show(block=False)
    # plt.savefig('./picture/LLE/Fuse.png')
    plt.show()

if __name__ == "__main__":
    
    
    # X, Y = make_swiss_roll(n_samples=500)  # 样本点
    #
    # scatter_3d(X,Y)
    #
    # data_2d = lle(X, n_dims = 2, n_neighbors = 12)  # n_dims=2 降到2维  n_neighbors=12 临近点数量为12
    # print(data_2d.shape)

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
    # scatter_3d(data,labs)

    # 进行降维
    data_2d_LPP = lle(data, n_dims=10, n_neighbors = 10)
    dimA = data_2d_LPP
    print(dimA.shape)
    # 绘图
    plt.subplot(2,2,1)
    plt.title("dimA=10")
    plt.scatter(dimA[:, 0], dimA[:, 1], c = labs)  # dimA[:, 2],, marker='+'
    plt.show()

    # data_2d = lle(data, n_dims=20, n_neighbors=12)
    # print(data_2d.shape)



    # 降维后的数据打印
    # plt.figure()
    # plt.title("my_LLE")
    # plt.scatter(data_2d[:, 0], data_2d[:, 1], c = labs, cmap=plt.cm.hot)
    # # plt.show(block=False)
    # plt.show()


    # 用sklearn进行降维
    # data_2d_sk = LocallyLinearEmbedding(n_components=2, n_neighbors = 12).fit_transform(X)
    #
    # plt.figure()
    # plt.title("my_LLE_sk")
    # plt.scatter(data_2d[:, 0], data_2d[:, 1], c = Y,cmap=plt.cm.hot)
    # plt.show()
    
    
    
        
        
        
        
        
        
        
        
        
        
    

