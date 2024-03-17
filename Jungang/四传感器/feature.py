import keras
import tensorflow.keras.backend as K
import numpy as np
from useful_tools import MatHandler
from useful_tools import network_encoder
from tensorflow.keras.layers import Input
from tensorflow.keras import models

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def feature_extraction(code_size=16,signal_size=1024):
    # 模型
    encoder_input = Input([signal_size, 1])  # 输入为([1024*1],[1])
    encoder_output = network_encoder(encoder_input, code_size)  # 输入为（[1024*1],[16*1]）
    encoder_model = models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()
    encoder_model.load_weights(r'.\models\310-10encoder.h5')  # 加载权重文件的路径
    # 生成数据
    mathandler = MatHandler()
    data = mathandler.X_train
    label = mathandler.y_train
    print(data.shape)
    print(label.shape)
    # 预测模型
    output = encoder_model.predict(data)  # output=(,16) 提取出的16维的特征
    print(output.shape)
    # np.save('./feature/Xfeature' + 'Xdataset-314-1' ,output)
    # np.save('./feature/Xfeature' + 'Xlabel-314-1' ,label)
    np.save('Xdataset-314-1' ,output)
    np.save('Xlabel-314-1' ,label)
    # 清理没用的指针
    # K.clear_session()

if __name__ == "__main__":
    feature_extraction()



    # data_X=np.load('Xfearure.npy')
    # print(data_X)
    # data1=np.load('.\\feature\\Xdataset.npy')
    # print(data1.shape)
    # data=data1.reshape(20224,1,order='C')
    # print(data.shape)
    # Dense(
    #     x = data,
    #     classification=6
    # )