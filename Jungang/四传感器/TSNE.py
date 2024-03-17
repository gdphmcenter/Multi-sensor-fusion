from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.colors import  rgb2hex
# 加载数据
from datatest import MatHandler

def plot_with_labels(train, labels, clusters_number=5):  # 聚类的总数为10类
    """
    绘制聚类图
    DONE
    """
    # 设置字体
    from matplotlib import rcParams
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": 10,
    }
    rcParams.update(config)

    fig, ax = plt.subplots()
    # fig, axes = plt.subplots(2, 2)  # 此处是一个2*2的图
    np.random.seed(0)
    colors = tuple(
        [(np.random.random(), np.random.random(), np.random.random()) for i in range(clusters_number)])  # 生成元组
    colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex

    for i, color in enumerate(colors):
        need_idx = np.where(labels == i)[0]
        ax.scatter(train[need_idx, 1], train[need_idx, 0], c=color, label=i)
    # legend = ax.legend(loc='BestOutside')
    legend = ax.legend(loc=[1, 0])
    # plt.legend(loc=[1, 0])
    plt.savefig('./picture/clustering/' + 'X-220'+'.png')  # 图片保存位置
    plt.show()

def TSNE_Mat(
        cluster_num=6 # 聚类数量为10
):
    """
    TSNE降维
    """
    data = np.load('.\\feature\\Xdataset-222.npy')
    label = np.load('.\\feature\\Xlabel-222.npy')
    # mathandler = MatHandler()
    # data1 = mathandler.X_train
    # label = mathandler.y_train
    # data=(data1.reshape(data1.shape[0], data1.shape[1] * data1.shape[2]))
    # data=(data1.reshape(data1.shape[0], data1.shape[1] * data1.shape[2]))
    print(data.shape)
    # print(label.shape)
    # tsne降维
    tsne = TSNE(perplexity=10, n_components=2, init='random', n_iter=5000)
    # 降维后的数据
    low_dim_embs = tsne.fit_transform(data)
    # labels = label

    # 画图
    plot_with_labels(low_dim_embs, label, clusters_number=cluster_num,)
    # plot_with_labels(low_dim_embs,  clusters_number=cluster_num,)


if __name__ == "__main__":
    TSNE_Mat(
        cluster_num=5 # 一共6类
    )



# from sklearn.manifold import TSNE
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# # 加载数据集
# data = np.load('.\\feature\\Xdataset.npy')
# label = np.load('.\\feature\\Xlabel1.npy')
# print(data.shape)
# print(label.shape)
#
# tsne = TSNE(perplexity=10, n_components=2, init='random', n_iter=5000).fit_transform(data)
# # 使用PCA 进行降维处理
# pca = PCA().fit_transform(data)
# # 设置画布的大小
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.scatter(tsne[:, 0], tsne[:, 1], c=label)
# plt.subplot(122)
# plt.scatter(pca[:, 0], pca[:, 1], c=label)
# plt.colorbar()
# plt.show()

