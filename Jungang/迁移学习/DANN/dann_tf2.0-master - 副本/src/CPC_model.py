import keras
import keras.backend as K
from tensorflow import keras
from keras import utils
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv1D,Lambda,Reshape, concatenate, BatchNormalization, GRU, Input, TimeDistributed, Activation, MaxPool2D, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras import Model
from keras.utils.vis_utils import plot_model
from os.path import join


#特征提取器

# def network_encoder1(input_shape1,code_size):
#
#     input_img = keras.Input(shape=input_shape1)
#
#     x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(input_img)
#     # x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#     #
#     x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     # x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
#     # x = keras.layers.BatchNormalization()(x)
#     # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.LeakyReLU()(x)
#
#     # x = keras.layers.Conv1D(filters=1024, kernel_size=3, strides=1, activation='linear')(x)
#     # x = keras.layers.BatchNormalization()(x)
#     # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Flatten()(x)
#     # x = keras.layers.Dense(units=1024,activation='relu')(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.Dense(units=256,activation='relu')(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.Dense(units=256,activation='relu')(x)
#     x = keras.layers.Dropout(0.2)(x)
#
#     code_size = keras.layers.Dense(units=16, activation='linear', name='encoder_embedding1')(x)
#
#     encoded1 = tf.keras.models.Model(input=input_img, outputs=code_size, name='encoder1')
#     # encoded1.save(join(output_dir, 'encoded1.h5'))
#
#     return encoded1
#
# def network_encoder2(input_shape2,code_size):
#
#     input_img = keras.Input(shape=input_shape2)
#
#     x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(input_img)
#     # x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#     #
#     x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     # x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
#     # x = keras.layers.BatchNormalization()(x)
#     # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.LeakyReLU()(x)
#
#     # x = keras.layers.Conv1D(filters=1024, kernel_size=3, strides=1, activation='linear')(x)
#     # x = keras.layers.BatchNormalization()(x)
#     # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Flatten()(x)
#     # x = keras.layers.Dense(units=1024,activation='relu')(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.Dense(units=256,activation='relu')(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.Dense(units=256,activation='relu')(x)
#     x = keras.layers.Dropout(0.2)(x)
#
#     code_size = keras.layers.Dense(units=16, activation='linear', name='encoder_embedding2')(x)
#
#     encoded2 = tf.keras.models.Model(input=input_img, outputs=code_size, name='encoder2')
#     # encoded1.save(join(output_dir, 'encoded1.h5'))
#
#     return encoded2
#
# def network_encoder3(input_shape3,code_size):
#
#     input_img = keras.Input(shape=input_shape3)
#
#     x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(input_img)
#     # x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#     #
#     x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     # x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
#     # x = keras.layers.BatchNormalization()(x)
#     # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.LeakyReLU()(x)
#
#     # x = keras.layers.Conv1D(filters=1024, kernel_size=3, strides=1, activation='linear')(x)
#     # x = keras.layers.BatchNormalization()(x)
#     # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Flatten()(x)
#     # x = keras.layers.Dense(units=1024,activation='relu')(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.Dense(units=256,activation='relu')(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.Dense(units=256,activation='relu')(x)
#     x = keras.layers.Dropout(0.2)(x)
#
#     code_size = keras.layers.Dense(units=16, activation='linear', name='encoder_embedding3')(x)
#
#     encoded3 = tf.keras.models.Model(input=input_img, outputs=code_size, name='encoder3')
#     # encoded1.save(join(output_dir, 'encoded1.h5'))
#
#     return encoded3
#
# def network_encoder4(input_shape4,code_size):
#
#     input_img = keras.Input(shape=input_shape4)
#
#     x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(input_img)
#     # x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#     #
#     x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.LeakyReLU()(x)
#
#     # x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
#     # x = keras.layers.BatchNormalization()(x)
#     # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.LeakyReLU()(x)
#
#     # x = keras.layers.Conv1D(filters=1024, kernel_size=3, strides=1, activation='linear')(x)
#     # x = keras.layers.BatchNormalization()(x)
#     # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.LeakyReLU()(x)
#
#     x = keras.layers.Flatten()(x)
#     # x = keras.layers.Dense(units=1024,activation='relu')(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     # x = keras.layers.Dense(units=256,activation='relu')(x)
#     # x = keras.layers.Dropout(0.2)(x)
#     x = keras.layers.Dense(units=256,activation='relu')(x)
#     x = keras.layers.Dropout(0.2)(x)
#
#     code_size = keras.layers.Dense(units=16, activation='linear', name='encoder_embedding4')(x)
#
#     encoded4 = tf.keras.models.Model(input=input_img, outputs=code_size, name='encoder4')
#     # encoded1.save(join(output_dir, 'encoded1.h5'))
#
#     return encoded4



def network_encoder1(x, code_size,output_dir='models'):
    ''' Define the network mapping images to embeddings '''

    # input_img = keras.Input(shape=(x,1))

    x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)
    #
    x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    # x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.LeakyReLU()(x)

    # x = keras.layers.Conv1D(filters=1024, kernel_size=3, strides=1, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(units=1024,activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.Dense(units=256,activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(units=256,activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding1')(x)

    # encoded1 = tf.keras.Model(x, encoded, name='encoder1')
    # encoded1.save(join(output_dir, 'encoded1.h5'))

    return x


def network_encoder2(x, code_size,output_dir='models'):
    ''' Define the network mapping images to embeddings '''

    # input_img = keras.Input(shape=(x,1))

    x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)
    #
    x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    # x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.LeakyReLU()(x)

    # x = keras.layers.Conv1D(filters=1024, kernel_size=3, strides=1, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(units=1024,activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.Dense(units=256,activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(units=256, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding2')(x)

    # encoded2 = tf.keras.Model(x, encoded, name='encoder2')
    # encoded2.save(join(output_dir, 'encoded2.h5'))

    return x

def network_encoder3(x, code_size,output_dir='models'):
    ''' Define the network mapping images to embeddings '''

    # input_img = keras.Input(shape=(x,1))

    x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)
    #
    x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    # x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.LeakyReLU()(x)

    # x = keras.layers.Conv1D(filters=1024, kernel_size=3, strides=1, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(units=1024,activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.Dense(units=256,activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(units=256,activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding3')(x)


    # encoded3 = tf.keras.Model(x, encoded, name='encoder3')
    # encoded3.save(join(output_dir, 'encoded3.h5'))

    return x

def network_encoder4(x, code_size,output_dir='models'):
    ''' Define the network mapping images to embeddings '''


    # input_img = keras.Input(shape=(x,1))

    x = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)
    #
    x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.LeakyReLU()(x)

    # x = keras.layers.Conv1D(filters=512, kernel_size=3, strides=1, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.LeakyReLU()(x)

    # x = keras.layers.Conv1D(filters=1024, kernel_size=3, strides=1, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(units=1024,activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.Dense(units=256,activation='relu')(x)
    # x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(units=256,activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding4')(x)


    # encoded4 = tf.keras.Model(x, encoded, name='encoder4')
    # encoded4.save(join(output_dir, 'encoded4.h5'))

    return x




def network_encoder(encoder_input, code_size,output_dir='model'):
    X = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 4})(encoder_input)
    inputA = X[0]
    inputB = X[1]
    inputC = X[2]
    inputD = X[3]

    encoded1 = network_encoder1(inputA, code_size)
    encoded2 = network_encoder2(inputB, code_size)
    encoded3 = network_encoder3(inputC, code_size)
    encoded4 = network_encoder4(inputD, code_size)

    # combinedInput = concatenate([x_encoded, y_encoded, z_encoded, a_encoded])
    combinedInput = concatenate([encoded1, encoded2, encoded3, encoded4])
    C = Dense(units=32, activation='relu')(combinedInput)
    encoder_output = Dense(units=code_size, activation='linear')(C)

    return encoder_output



# 循环神经网络
def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    x = GRU(units=256, return_sequences=False, name='ar_context')(x)

    return x

# 总体网络
def network_cpc_Mat(image_shape, terms, predict_terms, code_size, learning_rate):

    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)
    # 3072*1
    encoder_input = Input([image_shape,1])
    print('encoder_input',encoder_input.shape)
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = Model(inputs=encoder_input, outputs=encoder_output, name='encoder')
    encoder_model.summary()


    # Define encoder model
    x_input = Input((terms, image_shape,1))
    print('x_input',x_input.shape)
    x_encoded = TimeDistributed(encoder_model)(x_input)
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = Input((predict_terms, image_shape, 1))
    # y_input = Input((predict_terms, input1[0], input1[1], input1[2]))
    y_encoded = TimeDistributed(encoder_model)(y_input)   # 降维度

    # Loss 定义类
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)

    # Compile model
    cpc_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    cpc_model.summary()             # 输出模型参数

    return cpc_model


def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

if __name__ == "__main__":
    print("suc")
    print(keras.__version__)
    model = network_cpc_Mat(image_shape=4096,terms=4,predict_terms=4,code_size=16,learning_rate=1e-3)
    plot_model(model, 'model1.png')  # show_shapes=True, show_dtype=True, show_layer_names=True
    model.summary()