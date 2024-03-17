from useful_tools import network_encoder, MatHandler, plot_with_labels,within_the_class_distance, check_label
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from collections import Counter
import keras
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def TSNE_Mat(code_size, signal_size=4096, cluster_num=4):
    """
    TSNE降维训练好的模型并且画聚类图和类内距离和类间距离
    DONE
    """
    #模型
    encoder_input = keras.layers.Input([signal_size, 1])
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()
    encoder_model.load_weights(r'.\models\ABCD-8+64+64+64.h5')
    # 生成数据                                        
    mathandler = MatHandler()
    # data = mathandler.X_val
    # label = mathandler.y_val
    data = mathandler.X_train
    label = mathandler.y_train
    # data = mathandler.X_test
    # label = mathandler.y_test
    # 预测模型
    output = encoder_model.predict(data)
    # 清理没用的指针
    K.clear_session()
    # tsne降维 
    tsne = TSNE(perplexity=10, n_components=2, init='random', n_iter=5000)
    
    low_dim_embs = tsne.fit_transform(output)
    labels = label
    
    # 类内距离
    # DONE
    within_the_class_distance(low_dim_embs, labels, class_range_number=cluster_num)
    # 画图
    # DONE
    plot_with_labels(low_dim_embs, labels, clusters_number=cluster_num)


##########################画混淆矩阵+计算正确率
def check_10_cluster(code_size, signal_size=4096):
    """
    PCA
    多次
    k_means后数据后计算轮廓系数
    """

    #模型
    encoder_input = keras.layers.Input([signal_size, 1])
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    # encoder_model.summary()
    encoder_model.load_weights(r'.\models\encoder.h5')
    # 生成数据                                        
    mathandler = MatHandler()
    data = mathandler.X_train
    label = mathandler.y_train
    # data = mathandler.X_test
    # label = mathandler.y_test
    # data = mathandler.X_val
    # label = mathandler.y_val
    # 预测模型
    data = encoder_model.predict(data)

    pca_data = data

    cluter_num = 4

    ################# k_means
    kmeans = KMeans(n_clusters=cluter_num)
    result_list = kmeans.fit_predict(pca_data)
    #################

    confusion, list_Counter = check_label(label, result_list, result_num=cluter_num)
    confusion_Matrix(confusion*100)
    print(confusion)

    print(list_Counter)
    max_accuracy = counter_max_accuracy(list_Counter, confusion, result_num=cluter_num)
    print("max_accuracy:"+str(max_accuracy))
    return "suc"


def confusion_Matrix(confusion):
    """
    画混淆矩阵
    """
    confusion = np.array(confusion[:,:], dtype=np.int32)
    from matplotlib import rcParams
    config = {
        "font.family":'Times New Roman',  # 设置字体类型
        "font.size": 10,
    }
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',    
            'size'   : 10,
    }
    rcParams.update(config)

    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)

    indices = range(len(confusion))
    # # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表

    plt.xticks(indices, range(4))
    plt.yticks(indices, range(4))

    cb = plt.colorbar()
    cb.set_label('percentage(%)')
    
    plt.xlabel('real class', font)
    plt.ylabel('cluster class', font)

    # 显示数据
    for first_index in range(len(confusion)):    #第几行
        for second_index in range(len(confusion[first_index])):    #第几列
            plt.text(second_index, first_index, confusion[first_index][second_index])

    plt.savefig('./picture/confusion/Fuse-5.png')
    plt.show()


def counter_max_accuracy(list_Counter, confusion, result_num):
    # 计算每个分类总的准确率
    sum_right_number = 0
    sum = 0
    for i in range(result_num):
        temp_class_number = []
        class_num = 0.0
        for j in range(result_num):
            temp_class_number.append(list_Counter[i][class_num])
            sum += list_Counter[i][class_num]
            class_num += 1
        print("第"+str(i)+"类：最大的数"+str(max(temp_class_number)))
        sum_right_number += max(temp_class_number)
    
    return sum_right_number/sum
##########################

def check_label_classifier(label, predict_label, result_num):
    predict_label = predict_label.argmax(axis=1).astype(float)
    list_Counter = []
    cluster_num = 4
    confusion = np.zeros((cluster_num, cluster_num))

    print(label)
    print(predict_label)
    print(label.shape)
    print(predict_label.shape)
    print(type(label[1]))
    print(type(predict_label[1]))

    for i in range(result_num):
        list_Counter.append(Counter(label[predict_label == i]))
    for i in range(result_num):
        print("第" + str(i) + "类：")
        sum = 0
        class_num = 0.0
        for j in range(result_num):
            sum += list_Counter[i][class_num]
            class_num += 1

        class_num = 0.0
        for j in range(result_num):
            if list_Counter[i][class_num] != 0:
                print("类" + str(class_num) + "的百分比：" + str(list_Counter[i][class_num] / sum))
                confusion[i][int(class_num)] = list_Counter[i][class_num] / sum
            class_num += 1

    confusion = np.around(confusion, 2)

    return confusion

def feature_extraction(code_size=16,signal_size=4096):
    # 模型
    encoder_input = keras.layers.Input([signal_size, 1])  # 输入为([1024*1],[1])
    encoder_output = network_encoder(encoder_input, code_size)  # 输入为（[1024*1],[16*1]）
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    # encoder_model.summary()
    encoder_model.load_weights(r'.\models\encoder.h5')  # 加载权重文件的路径
    # 生成数据
    mathandler = MatHandler()
    data = mathandler.X_train
    label = mathandler.y_train
    # 预测模型
    output = encoder_model.predict(data)  # output=(,16) 提取出的16维的特征
    print(output.shape)
    # np.save('./feature/Xfeature/Xdataset-314-2', output)
    # np.save('./feature/Xfeature/Xlabel-314-2' ,label)
    np.save('./feature/Xfeature/Z02data.npy', output)
    np.save('./feature/Xfeature/Z02label.npy' ,label)

    # 清理没用的指针
    K.clear_session()


def classify(epochs, batch_size, output_dir, code_size, signal_size=4096, cluster_num=4):

    # 加载预训练的VGG16模型
    encoder_input = tf.keras.layers.Input([signal_size, 1])
    encoder_output = network_encoder(encoder_input, code_size)
    feature = keras.models.Model(encoder_input, encoder_output, name='encoder')

    # h5文件
    # feature.load_weights(r'.\models\new_models\四传感器\encoder10-32-512.h5',by_name=True)
    feature.load_weights(r'.\models\多源域\ABC-184+8+8-10.h5')

    feature.summary()

    for layer in feature.layers:
        print(layer.name, layer.get_weights())

    # # 冻结VGG16模型中的所有层
    for layer in feature.layers:
        layer.trainable = False
        # layer.trainable = True

    # 添加自定义的分类层
    # x = tf.keras.layers.Dense(512, activation='relu',name='Dense_1')(encoder_output)
    # classifier_output = tf.keras.layers.Dense(4, activation='softmax', name='Dense_2')(x)
    classifier_output = tf.keras.layers.Dense(4, activation='softmax', name='Dense_2')(encoder_output)

    # 构建新的模型
    model = tf.keras.models.Model(inputs=encoder_input, outputs=classifier_output)
    model.summary()

    # 编译模型
    model.compile(
                optimizer='adam',
                # optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 准备数据
    mathandler = MatHandler()

    # data = mathandler.X_train
    # label = mathandler.y_train
    data = mathandler.X_val
    label = mathandler.y_val
    # X_train = np.vstack((x_train,x_val))
    # Y_train = np.vstack((y_train,y_val))
    label = tf.keras.utils.to_categorical(label, cluster_num)

    x_test = mathandler.X_test
    y_test = mathandler.y_test
    tmp_label = y_test
    y_test = tf.keras.utils.to_categorical(y_test, cluster_num)

    # 训练模型
    model.fit(
        data,label,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        )
    model.summary()

    # 可训练层
    print("可训练层")
    for x in model.trainable_weights:
        print(x.name)
    print('\n')

    # 不可训练层
    print("不可训练层")
    for x in model.non_trainable_weights:
        print(x.name)
    print('\n')

    # 保存模型
    model.save('./models/Transfer-1.h5')

    # 打印精度
    # 加载测试数据集并进行测试
    score = model.evaluate(x_test, y_test)
    predict_label = model.predict(x_test)
    predict_label = predict_label.argmax(axis=1).astype(float)
    cm = confusion_matrix(tmp_label, predict_label , normalize='true')
    labels=['0','1','2','3']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    print(cm)
    # disp.plot(cmap = 'Greens')  # 绿色
    disp.plot(cmap = 'Blues')    # 蓝色
    plt.savefig('./picture/confusion/CPC/' + '_' + str(score[1]) + '--' +str(epochs) +'--'+str(score[0]) + '.png')
    plt.show()

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    K.clear_session()






if __name__ == "__main__":
    # 加载模型+画图+计算类内距离+类间距离
    TSNE_Mat(
        code_size=16,
        cluster_num=4,
        signal_size=4096
    )
    # 画混淆矩阵+计算正确率
    # check_10_cluster(
    #     code_size=16,
    #     signal_size=1024
    # )
    # 特征保存
    # feature_extraction(
    #     code_size=16,
    # )
    # 进行分类测试
    # Classfier(
    #     epochs=1,
    #     batch_size=32,
    #     code_size=16,
    #     signal_size=1024,
    #     lr=1e-3
    # )
    print("suc")

