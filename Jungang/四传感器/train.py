from useful_tools import SortedGenerator_Mat_8192
from useful_tools import network_cpc_Mat
from keras.callbacks import TensorBoard
from os.path import join, basename, dirname, exists
from useful_tools import network_encoder1, network_encoder2, network_encoder3,network_encoder4
import keras
from keras import backend as K
import os
import numpy as np
import sys
import pandas as pd
import tensorflow as tf
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# np.set_printoptions(threshold=np.inf)
#
# import os
#
# # # 设置程序使用的GPU编号：我用6号和7号显卡
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#
# import numpy as np
# import keras
# import keras.backend as K
#
# import tensorflow.compat.v1 as tf
#
# tf.disable_v2_behavior()
#
# import keras.backend as K
#
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.99)
# gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def train_model_Mat_8192(epochs, batch_size, output_dir, code_size, lr=1e-4, terms=4, predict_terms=4, signal_size=4096):
    # USEFUL
    # Prepare data
    # TODO
    # new数据集
    train_data = SortedGenerator_Mat_8192(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=1, predict_terms=predict_terms,
                                       signal_size=signal_size)

    validation_data = SortedGenerator_Mat_8192(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=1, predict_terms=predict_terms,
                                            signal_size=signal_size)
    # Prepares the model
    # TODO
    # 整个模型
    # encoder1 = network_cpc_Mat()
    model = network_cpc_Mat(image_shape=signal_size, terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=lr)

    # checkpoint_save_path = "./models/4#-1.ckpt"
    # if os.path.exists(checkpoint_save_path + '.index'):
    #     print('-------------load the model-----------------')
    #     model.load_weights(checkpoint_save_path)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
    #                                                  save_weights_only=True,
    #                                                  save_best_only=True)

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        # callbacks=[cp_callback]
        callbacks=callbacks
    )

    # np.set_printoptions(threshold=1000)
    # np.set_printoptions(threshold=sys.maxsize)
    # print(model.trainable_variables)

    # file = open('./4#weights1-32-512-2.txt', 'w')
    # for v in model.trainable_variables:
    #     file.write(str(v.name) + '\n')
    #     file.write(str(v.shape) + '\n')
    #     file.write(str(v.numpy()) + '\n')
    #     # pd.set_option('display.width', 1000)  # 设置字符显示宽度
    #     # pd.set_option('display.max_rows', None)  # 设置显示最大行
    #     # pd.set_option('display.max_columns', None)  # 设置显示最大行
    # file.close()

    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    model.save(join(output_dir, 'CPC.h5'))

    # Saves the encoder alone
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, 'ABCD-8+64+64+64.h5'))


if __name__ == "__main__":
    # 训练模型
    train_model_Mat_8192(
        epochs=10,
        batch_size=32,
        output_dir='models',
        code_size=16,
        lr=1e-4,
        terms=4,
        predict_terms=4,
        signal_size=4096,
    )
    print("suc")


