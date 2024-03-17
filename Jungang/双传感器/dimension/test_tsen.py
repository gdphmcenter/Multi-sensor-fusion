import numpy as np
import matplotlib.pyplot as plt
from test_PCA import pca
# 计算任意两点之前距离 ||x_i-x_j||^2
# X 维度 [N,D]
def cal_pairwise_dist(X):
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    #返回任意两个点之间距离
    return D



# 计算P_ij 以及 log松弛度
def calc_P_and_entropy(D,beta=1.0):
    P = np.exp(-D.copy() * beta)
    sumP = np.sum(P)
    # 计算熵
    log_entropy = np.log(sumP) + beta * np.sum(D * P) / sumP
    
    P = P/sumP
    return P,log_entropy
    
  
    
# 二值搜索寻找最优的 sigma
def binary_search(D, init_beta,logU, tol=1e-5, max_iter=50):
    
    beta_max = np.inf
    beta_min = -np.inf
    beta = init_beta
    
    P,log_entropy=calc_P_and_entropy(D,beta)
    diff_log_entropy = log_entropy - logU

    m_iter = 0
    while np.abs(diff_log_entropy)> tol and m_iter<max_iter:
        # 交叉熵比期望值大，增大beta
        if diff_log_entropy>0:
            beta_min = beta
            if beta_max == np.inf or beta_max == -np.inf:
                beta = beta*2
            else:
                beta = (beta+beta_max)/2.
        # 交叉熵比期望值小， 减少beta        
        else:
            beta_max = beta
            if beta_min == -np.inf or beta_min == -np.inf:
                beta = beta/2
            else:
                beta = (beta + beta_min)/2.
        
        # 重新计算
        P,log_entropy=calc_P_and_entropy(D,beta)
                
        diff_log_entropy = log_entropy - logU
        
        m_iter = m_iter+1
    
    # 返回最优的 beta 以及所对应的 P
    return P, beta
        

# 给定一组数据 datas ：[N,D] 
# 计算联合概率 P_ij : [N,N]
def p_joint(datas, target_perplexity):
    
    N,D = np.shape(datas)
    # 计算两两之间的距离
    distances = cal_pairwise_dist(datas)
    
    beta = np.ones([N,1])  # beta = 1/(2*sigma^2)
    logU = np.log(target_perplexity)
    p_conditional = np.zeros([N,N])
    # 对每个样本点搜索最优的sigma(beta) 并计算对应的P
    for i in range(N):
        if i %500 ==0:
            print("Compute joint P for %d points"%(i))
        # 删除 i -i 点
        Di = np.delete(distances[i,:],i)
        # 进行二值搜索，寻找 beta 
        # 使 log_entropy 最接近 logU
        P, beta[i] = binary_search(Di, beta[i],logU)
        
        # 在ii的位置插0
        p_conditional[i] = np.insert(P,i,0)

    # 计算联合概率
    P_join = p_conditional + p_conditional.T
    P_join = P_join/np.sum(P_join)
    
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P_join
    
# Y : 低维数据 [N,d]
# 根据Y，计算低维的联合概率 q_ij
def q_tsne(Y):
    N = np.shape(Y)[0]
    sum_Y = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)
    num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
    num[range(N), range(N)] = 0.
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    return Q,num
   

# datas 输入高维数据 [N,D]
# labs 高维数据的标签[N,1]
# dim 降维的维度 d  
# plot 绘图

def estimate_tsen(datas,labs,dim,target_perplexity,plot=False):
    
    N,D = np.shape(datas)
    
    # 随机初始化低维数据Y
    Y = np.random.randn(N, dim)
    
    # 计算高维数据的联合概率
    print("Compute P_joint")
    P = p_joint(datas, target_perplexity)
    
    # 开始若干轮对 P 进行放大
    P = P*4.
    P = np.maximum(P, 1e-12)
    
    # 开始进行迭代训练
    # 训练相关参数
    max_iter = 1500
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500  # 学习率
    min_gain = 0.01
    dY = np.zeros([N, dim]) # 梯度
    iY = np.zeros([N, dim]) # Y的变化
    gains = np.ones([N, dim])
    
    for m_iter in range(max_iter):
        
        # 计算 Q 
        Q,num= q_tsne(Y)
        
        # 计算梯度
        PQ = P - Q
        for i in range(N):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (dim, 1)).T * (Y[i, :] - Y), 0)
    
        
        # Perform the update
        if m_iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        
        # Y 取中心化
        Y = Y - np.tile(np.mean(Y, 0), (N, 1))

        # Compute current value of cost function
        if (m_iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: loss is %f" % (m_iter + 1, C))

        # 停止放大P
        if m_iter == 100:
            P = P / 4.
   
        if plot and m_iter % 100 == 0:
            print("Draw Map")
            draw_pic(Y,labs,name = "%d.jpg"%(m_iter))
    
    return Y
    

    
def draw_pic(datas,labs,name = '1.jpg'):
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
    # plt.savefig(name)
    plt.savefig('./picture/TSNE/Fuse.png')
    plt.show()

    




if __name__ == "__main__":
    
    # mnist_datas = np.loadtxt("mnist2500_X.txt")
    # mnist_labs = np.loadtxt("mnist2500_labels.txt")
    #
    # print("first reduce by PCA")
    # datas, _= pca(mnist_datas, 30)
    # X = datas.real
    #
    # Y = estimate_tsen(X,mnist_labs,2,30, plot=True)
    #
    # draw_pic(Y,mnist_labs,name = "final.jpg")

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
    datas, _= pca(data, 30)
    data = datas.real
    labs = estimate_tsen(data,labs,2,30,plot=True)

    draw_pic(labs,labs,name='final.png')
    
        
        
        
        
        
        
        
        
        
        
    

