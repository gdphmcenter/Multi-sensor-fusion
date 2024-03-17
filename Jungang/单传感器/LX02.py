from tensorflow.keras.layers import Conv1D,Lambda,Reshape, concatenate, BatchNormalization, GRU, Input, TimeDistributed, Activation, MaxPool2D, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape

#　设置随机种子，保证每次生成的随机数一样，可以不设置（去除下面一行代码，将所有的 rd 替换成 np.random 即可）
rd = np.random.RandomState(888)
#
x1 = rd.randint(-100, 300, (3072, 1)) # 随机生成[-2,3)的整数，10x10的矩阵
print(x1.shape)
# # x2 = rd.randint(-100, 300, (1024, 1)) # 随机生成[-2,3)的整数，10x10的矩阵
# # x3 = rd.randint(-100, 300, (1024, 1)) # 随机生成[-2,3)的整数，10x10的矩阵
# A = x1[0:1024]
# print(A.shape)
# print(A)
#
# B = x1[1024:2048]
# print(B.shape)
# print(B)
#
# C = x1[2048:3072]
# print(C.shape)
# print(C)

x = Lambda(tf.split, arguments={'axis': 0, 'num_or_size_splits': 3})(x1)
A = x[0]
B = x[1]
C = x[2]
print(x[0].shape)
print(x[1].shape)
print(x[2].shape)

print(A)
print(B)
print(C)
