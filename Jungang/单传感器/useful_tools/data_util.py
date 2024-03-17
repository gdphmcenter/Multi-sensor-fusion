import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys
from matplotlib import pyplot as plt
# caixiaoman+
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import scipy.io as scio


class SortedGenerator_Mat(object):
    # USEFUL
    # TODO
    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, signal_size=1024, clusters_number=4):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset                            # 参数为train，test
        self.terms = terms
        self.clusters_number = clusters_number

        self.signal_size = signal_size

        # Initialize Mat dataset
        self.mathandler = MatHandler()
        self.n_samples = self.mathandler.get_n_samples(subset) # terms 获得图片个数
        self.n_batches = self.n_samples # batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Build sentences
        image_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')     # 标记句子是否为正样本
        positive_samples_n = self.positive_samples  # 正样本的数量
        # 采样
        for b in range(self.batch_size):

            # Set ordered predictions for positive samples
            seed = np.random.randint(0, self.clusters_number)
            sentence = np.mod(np.arange(seed, seed + self.terms + self.predict_terms), self.clusters_number)
            #
            if positive_samples_n <= 0:

                # Set random predictions for negative samples
                # Each predicted term draws a number from a distribution that excludes itself
                numbers = np.arange(0, self.clusters_number)
                predicted_terms = sentence[-self.predict_terms:] # 截取后4个
                for i, p in enumerate(predicted_terms):     # 随机改变
                    predicted_terms[i] = np.random.choice(numbers[numbers != p], 1)
                sentence[-self.predict_terms:] = np.mod(predicted_terms, self.clusters_number) # 重新新赋值
                sentence_labels[b, :] = 0  # 标记为负样本

            # Save sentence 保存矩阵
            image_labels[b, :] = sentence

            positive_samples_n -= 1

        # Retrieve actual images
        images, _ = self.mathandler.get_batch_by_labels(self.subset, image_labels.flatten(), self.signal_size)

        # Assemble batch
        images = images.reshape((self.batch_size, self.terms + self.predict_terms, images.shape[1], 1))
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]

        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)

        return [x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]


class SortedGenerator_Mat_8192(object):
    # UNUSEFUL
    # TODO
    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, signal_size=1024, clusters_number=4):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset                            # 参数为train，test
        self.terms = terms
        self.clusters_number = clusters_number

        self.signal_size = signal_size

        # Initialize Mat dataset
        self.mathandler_8192 = MatHandler_8192()
        self.mathandler = MatHandler()
        self.n_samples = self.mathandler.get_n_samples(subset) # terms 获得图片个数
        self.n_batches = self.n_samples # batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Build sentences
        image_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')     # 标记句子是否为正样本
        positive_samples_n = self.positive_samples  # 正样本的数量
        # 采样
        for b in range(self.batch_size):

            # Set ordered predictions for positive samples
            seed = np.random.randint(0, self.clusters_number)
            # sentence = np.mod(np.arange(seed, seed + self.terms + self.predict_terms), 10)
            sentence = (np.ones(self.terms + self.predict_terms) * seed).astype(np.int32)
            if positive_samples_n <= 0:

                # Set random predictions for negative samples
                # Each predicted term draws a number from a distribution that excludes itself
                numbers = np.arange(0, self.clusters_number)
                predicted_terms = sentence[-self.predict_terms:] # 截取后4个
                for i, p in enumerate(predicted_terms):     # 随机改变
                    # predicted_terms[i] = np.random.choice(numbers[numbers != p], 1)
                    predicted_terms[i] = np.random.choice(numbers, 1) # 是否包含原来的样本
                sentence[-self.predict_terms:] = np.mod(predicted_terms, self.clusters_number) # 重新新赋值
                sentence_labels[b, :] = 0  # 标记为负样本

            # Save sentence 保存矩阵
            image_labels[b, :] = sentence

            positive_samples_n -= 1

        # Retrieve actual images
        images, _ = self.mathandler.get_batch_by_labels(self.subset, image_labels.flatten(), self.signal_size, positive_samples_n=self.positive_samples, sum_term=self.terms + self.predict_terms)

        # Assemble batch
        images = images.reshape((self.batch_size, self.terms + self.predict_terms, images.shape[1], 1))
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]

        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)
        # return [x_images, y_images], sentence_labels
        return [x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]


class MatHandler_8192(object):
    # UNUSEFUL
    # TODO
    ''' Provides a convenient interface to manipulate MNIST data '''

    def __init__(self):

        # Download data if needed
        # 获得训练集，验证集，测试集
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()

    # def read_mat(self, dir='4分类/斩波器/1#'):
    # def read_mat(self, dir='4分类/斩波器/2#'):
    def read_mat(self, dir='4分类/斩波器/3#'):
    # def read_mat(self, dir='4分类/斩波器/4#'):
        # DONE
        data = np.array([],[])
        label = np.array([])
        count = 0
        # 遍历oneD文件夹中的各个mat文件
        for fn in os.listdir('./'+dir):
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

                now_data = read_data['data3'].T
                # now_data = read_data[data].T  # oneD数据
                # print(now_data.shape)
                # 剔除后面
                unwanted = now_data.shape[1] % 8192
                # print(unwanted)
                now_data = now_data[...,:-unwanted]
                # 分割数据为8192
                now_data = now_data.reshape(-1,8192)
                now_data_len = now_data.shape[0]
                # 记录标签
                for layer in range(int(now_data_len)):
                    label = np.append(label, int(now_data_label))
                # 第一次记录
                if count == 0:
                    data = now_data
                    count += 1
                    continue
                # 两次以上的记录
                data = np.vstack((data,now_data))
                count += 1
        # 返回数据集的数据和标标签
        data = data.reshape(-1, 8192, 1)
        return data, label

    def load_dataset(self):
        # DONE

        X, y = self.read_mat()
        # train:(913, 1024) test:(392, 1024)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

        ##################################
        # 打乱+合并
        X_train = np.squeeze(X_train)
        X_test = np.squeeze(X_test)

        X_train = np.vstack((X_train, X_test))
        
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        y_train = np.vstack((y_train, y_test))

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        
        X_train = X_train.reshape(-1, 8192, 1)
        X_test = X_test.reshape(-1, 8192, 1)
        ##################################

        # 250538
        X_train, X_val = X_train[:-38], X_train[-18:]
        y_train, y_val = y_train[:-38], y_train[-18:]
        # ################# 一维傅里叶变换
        X_train = oneD_Fourier_8192(X_train)
        X_test = oneD_Fourier_8192(X_test)
        X_val = oneD_Fourier_8192(X_val)
        #################

        return X_train, y_train, X_val, y_val, X_test, y_test



    def get_batch_by_labels(self, subset, labels, signal_size=1024):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Find samples matching labels
        idxs = []
        for i, label in enumerate(labels):

            idx = np.where(y == label)[0]
            # idx非空
            idx_sel = np.random.choice(idx, 1)[0]
            idxs.append(idx_sel)
        # Process batch
        batch = X[np.array(idxs), :].reshape((len(labels), 1024, 1))

        return batch.astype('float32'), labels.astype('int32')

    def get_n_samples(self, subset):

        if subset == 'train':
            y_len = self.y_train.shape[0]*4
        elif subset == 'valid':
            y_len = self.y_val.shape[0]*4
        elif subset == 'test':
            y_len = self.y_test.shape[0]*4

        return y_len


class MatHandler(object):
    # USEFUL
    # TODO
    ''' Provides a convenient interface to manipulate MNIST data '''

    def __init__(self):

        # Download data if needed
        # 获得训练集，验证集，测试集
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()
        self.mathandler_8192 = MatHandler_8192()

    # def read_mat(self, dir='4分类/斩波器/1#'):
    # def read_mat(self, dir='4分类/斩波器/2#'):
    def read_mat(self, dir='4分类/斩波器/3#'):
    # def read_mat(self, dir='4分类/斩波器/4#'):
    # def read_mat(self, dir='../4分类/02/拼接数据'):
        data = np.array([],[])
        label = np.array([])
        count = 0
        # 遍历oneD文件夹中的各个mat文件
        for fn in os.listdir('./'+dir):
            if fn.endswith('.mat'):
                # 路径
                # path = './oneD/'+"".join(fn)
                path = './' + dir + '/' + "".join(fn)  # join()是一个字符串方法，它返回被子字符串连接的字符串
                read_data = scio.loadmat(path)
                # print(read_data)
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
                now_data = read_data['data3'].T
                # now_data = read_data['data'].T
                # print(now_data.shape)
                # now_data.sort()
                # 剔除后面
                unwanted = now_data.shape[1] % 1024
                # print(unwanted)
                now_data = now_data[...,:-unwanted]
                # print(now_data.shape)
                # 分割数据为1024
                now_data = now_data.reshape(-1,1024)
                now_data_len = now_data.shape[0]
                # 记录标签
                for layer in range(int(now_data_len)):
                    label = np.append(label, int(now_data_label))
                # 第一次记录
                if count == 0:
                    data = now_data
                    count += 1
                    continue
                # 两次以上的记录
                data = np.vstack((data,now_data))
                count += 1
        # 返回数据集的数据和标标签
        data = data.reshape(-1, 1024, 1)
        return data, label

    def load_dataset(self):
        # DONE
        # We first define a download function, supporting both Python 2 and 3.
        # 判断python版本导入包
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        X, y = self.read_mat()
        # train:(913, 1024) test:(392, 1024)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

        ##################################
        # 打乱+合并
        X_train = np.squeeze(X_train)
        X_test = np.squeeze(X_test)

        X_train = np.vstack((X_train, X_test))
        
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        y_train = np.vstack((y_train, y_test))

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        
        X_train = X_train.reshape(-1, 1024, 1)
        X_test = X_test.reshape(-1, 1024, 1)
        ##################################

        # 250538
        X_train, X_val = X_train[:-316], X_train[-160:]
        y_train, y_val = y_train[:-316], y_train[-160:]
        ################# 一维傅里叶变换
        X_train = oneD_Fourier(X_train)
        X_test = oneD_Fourier(X_test)
        X_val = oneD_Fourier(X_val)
        #################

        return X_train, y_train, X_val, y_val, X_test, y_test


    def get_batch_by_labels(self, subset, labels, signal_size=3072, positive_samples_n = 1, sum_term = 8):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
            X_8192 = self.mathandler_8192.X_train
            y_8192 = self.mathandler_8192.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
            X_8192 = self.mathandler_8192.X_val
            y_8192 = self.mathandler_8192.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test
            X_8192 = self.mathandler_8192.X_test
            y_8192 = self.mathandler_8192.y_test

        ##################################8192
        positive_number = positive_samples_n
        # 生成正样本
        if positive_number > 0:
            idx = np.where(y_8192 == labels[0])[0]
            idx_sel = np.random.choice(idx, 1)[0]
            positive_sample = X_8192[np.array(idx_sel), :].reshape((sum_term, 1024, 1))
            positive_number -= 1

        # 保存句子
        sentence_sum = positive_sample
        i = 1
        while i * sum_term <= len(labels):
            position = (i - 1) * sum_term

            temp_idxs = []
            if i != 1:
                # 保存前4个一样
                positive_term_4 = positive_sample[:4,:]
                position += 4
                # 选取后4个
                for j in range(4):
                    if position + j > len(labels):
                        break
                    temp_position = position + j
                    idx = np.where(y == labels[temp_position])[0]
                    # print(y)
                    # idx非空
                    idx_sel = np.random.choice(idx, 1)[0]
                    # print(np.random.choice(idx, 1))
                    temp_idxs.append(idx_sel)
                # 提取后4个
                negative_term_4 = X[np.array(temp_idxs), :].reshape((4, 1024, 1))
                sentence = np.vstack((positive_term_4, negative_term_4))
                
                # 合并数据
                sentence_sum = np.vstack((sentence_sum, sentence))
            
            i += 1

        return sentence_sum.astype('float32'), labels.astype('int32')
        ##################################


    def get_n_samples(self, subset):

        if subset == 'train':
            y_len = self.y_train.shape[0]
        elif subset == 'valid':
            y_len = self.y_val.shape[0]
        elif subset == 'test':
            y_len = self.y_test.shape[0]

        return y_len


################################一维傅里叶变换
def oneD_Fourier(data):
    """
    一维傅里叶变换
    DONE DONE
    """
    # print(data.shape)
    # 数据多了一维
    data = np.squeeze(data)
    # print(data.shape)
    for layer in range(data.shape[0]):
        data[layer] = abs(np.fft.fft(data[layer]))
    data = data.reshape(-1,1024,1)
    
    return data


def oneD_Fourier_8192(data):
    """
    一维傅里叶变换
    DONE DONE
    """
    # print(data.shape)
    # 数据多了一维
    # data = np.squeeze(data)
    data = data.reshape(-1, 1024)
    for layer in range(data.shape[0]):
        data[layer] = abs(np.fft.fft(data[layer]))
    data = data.reshape(-1, 8192, 1)
    return data
################################


if __name__ == '__main__':
    pass

    matHandler_8192 = MatHandler_8192()
    X_train, y_train, X_val, y_val, X_test, y_test = matHandler_8192.load_dataset()
    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_val', X_val.shape)
    print('y_val', y_val.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)
    print('--'*30)
    print('--'*30)
    matHandler_8192 = MatHandler_8192()
    data, label = matHandler_8192.read_mat()
    print('data', data.shape)
    print('label', label.shape)
    print('--'*30)
    print('--'*30)

    matHandler = MatHandler()
    data, label = matHandler.read_mat()
    X_train, y_train, X_val, y_val, X_test, y_test = matHandler.load_dataset()
    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_val', X_val.shape)
    print('y_val', y_val.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)
    print('--'*30)
    print('--'*30)
    print('data', data.shape)
    print('label', label.shape)











