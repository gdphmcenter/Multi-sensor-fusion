import os.path

import h5py
#
from useful_tools import network_encoder
import keras
import csv
import pandas as pd
import tensorflow as tf
import numpy as np

#
#
#
signal_size=1024
code_size=16
#
encoder_input = keras.layers.Input([signal_size, 1])
encoder_output = network_encoder(encoder_input, code_size)
encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
# encoder_model.summary()
encoder_model.load_weights(r'.\models\encoder10-32-512-2.h5')

#
# 获取所有层，返回对象列表
layers = encoder_model.layers
# 获取每一层名字
for layer in layers:
    print(layer.name)