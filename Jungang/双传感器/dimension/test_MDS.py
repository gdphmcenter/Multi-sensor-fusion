import numpy as np
import matplotlib.pyplot as plt

# x 维度 [N,D]
def cal_pairwise_dist(x):  # 计算矩阵X之间两两距离
    
    N,D = np.shape(x)
    
    dist = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            # dist[i,j] = np.dot((x[i]-x[j]),(x[i]-x[j]).T)          # 欧氏距离的平方  2范数平方
            # dist[i,j] = np.sqrt(np.dot((x[i]-x[j]),(x[i]-x[j]).T)) # 2范数
            dist[i,j] = np.sum(np.abs(x[i]-x[j]))                    # 1范数
    
    #返回任意两个点之间距离  距离矩阵
    return dist
    
 
    

# dist N*N 距离矩阵样本点两两之间的距离 
# n_dims 降维
# 返回 降维后的数据
def my_mds(dist, n_dims):
    
    n,n = np.shape(dist)
    
    dist[dist < 0 ] = 0
    dist = dist**2
    T1 = np.ones((n,n))*np.sum(dist)/n**2
    T2 = np.sum(dist, axis = 1, keepdims=True)/n
    T3 = np.sum(dist, axis = 0, keepdims=True)/n

    B = -(T1 - T2 - T3 + dist)/2

    eig_val, eig_vector = np.linalg.eig(B)  # 特征值 特征向量
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]
    
    return picked_eig_vector*picked_eig_val**(0.5)
    
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
    plt.savefig('./picture/MDS/特征1范数.png')
    plt.show() 


if __name__ == "__main__":
    
    # # 加载数据
    # data = np.loadtxt("iris.data",dtype="str",delimiter=',')
    # feas = data[:,:-1]
    # feas = np.float32(feas)
    # labs = data[:,-1]
    # 加载数据
    dataX = np.load('.\\feature\\Xfeature\\X02data.npy')
    dataY = np.load('.\\feature\\Yfeature\\Y02data.npy')
    dataZ = np.load('.\\feature\\Zfeature\\Z02data.npy')
    data = np.hstack((dataX, dataY, dataZ))
    print(data.shape)
    # data = np.hstack((dataX, dataY))
    labs = np.load('.\\feature\\Xfeature\\X02label.npy')
    # labsY = np.load('.\\feature\\Yfeature\\Y02label.npy')
    # labsZ = np.load('.\\feature\\Zfeature\\Z02label.npy')
    # 计算距离
    dist = cal_pairwise_dist(data)
    
    # 进行降维
    data_2d = my_mds(dist, 30)
    print(data_2d.shape)
    
    #绘图
    draw_pic(data_2d,labs)
    # draw_pic(data_2d,labs)

    
    