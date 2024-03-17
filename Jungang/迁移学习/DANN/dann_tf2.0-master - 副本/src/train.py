from preprocessing import prepare_dataset, prepare_dataset_single
from models import DANN_Model

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime
from openTSNE import TSNE
(source_train_dataset, source_test_dataset) = prepare_dataset_single('ABCD-8+64+64+64')
# (_, target_test_dataset_1) = prepare_dataset_single('MNIST')
(target_train_dataset, target_test_dataset) = prepare_dataset_single('A')
lp_lr = 0.001
dc_lr = 0.0015
fe_lr = 0.0015

lr = (lp_lr, dc_lr, fe_lr)
model = DANN_Model(input_shape=(4096), model_type='ABCD-8+64+64+64', run_name='svhn2mnist', lr=lr)

EPOCHS = 50

for epoch in range(EPOCHS):

    print(datetime.datetime.now())

    for (source_images, class_labels), (target_images, _) in zip(source_train_dataset, target_train_dataset):
        model.train(source_images, class_labels, target_images)

    latent_source = []
    latent_target = []
    for (test_images, test_labels), (target_test_images, target_test_labels) in zip(source_test_dataset,
                                                                                    target_test_dataset):
        model.test_source(test_images, test_labels, target_test_images)
        model.test_target(target_test_images, target_test_labels)

        if len(latent_source) == 0:
            latent_source = model.return_latent_variables(test_images)
        else:
            latent_source = np.concatenate([latent_source, model.return_latent_variables(test_images)])

        if len(latent_target) == 0:
            latent_target = model.return_latent_variables(target_test_images)
        else:
            latent_target = np.concatenate([latent_target, model.return_latent_variables(target_test_images)])

    print('Epoch: {}'.format(epoch + 1))
    print(model.log())

    index = [0, len(latent_source), len(latent_source) + len(latent_target)]
    latent_variables = np.concatenate([latent_source, latent_target])

    pca_embedding = PCA(n_components=2).fit_transform(latent_variables)

    plt.figure()
    # plt.title('Epoch #{}'.format(epoch + 1))
    for i in range(len(index) - 1):
        plt.plot(pca_embedding[index[i]:index[i + 1], 0], pca_embedding[index[i]:index[i + 1], 1], 'o', alpha=0.5)
    # plt.legend(['A', 'B'])
    plt.savefig('./picture/多域/ABCD/8+64+64+64/DANN/softmax/记数/ABCD=A/' + '--' + str(epoch) + '--' + '.png')
    # plt.savefig('./picture/多域/ABCD/8+64+64+64/DANN/Dense/记数/ABCD=A/' + '--' + str(epoch) + '--' + '.png')
    # plt.savefig('./picture/多域/ABCD/8+64+64+64/CPC/Dense/记数/ABCD=A/' + '--' + str(epoch) + '--' + '.png')
    # plt.savefig('./picture/多域/ABCD/8+64+64+64/CPC/softmax/记数/ABCD=A/' + '--' + str(epoch) + '--' + '.png')

    # 单域
    # plt.savefig('./picture/单域/D-C/CPC/softmax/记数/' + '--' + str(epoch) + '--' + '.png')
    # plt.savefig('./picture/单域/D-C/CPC/Dense/记数/' + '--' + str(epoch) + '--' + '.png')
    # plt.savefig('./picture/单域/D-C/DANN/Dense/记数/' + '--' + str(epoch) + '--' + '.png')
    # plt.savefig('./picture/单域/D-C/DANN/softmax/记数/' + '--' + str(epoch) + '--' + '.png')

    # plt.show()
    # tsne = TSNE(n_components=2, initialization="pca")
    tsne = TSNE(perplexity=10, n_components=2, initialization="pca", n_iter=5000)

    # print(datetime.datetime.now())

    tsne_embedding = tsne.fit(latent_variables)

    print(datetime.datetime.now())

    plt.figure()
    # plt.title('Epoch #{}'.format(epoch + 1))
    for i in range(len(index) - 1):
        # plt.scatter(tsne_embedding[index[i]:index[i + 1], 0], tsne_embedding[index[i]:index[i + 1], 1], marker='o',s=5)
        plt.plot(tsne_embedding[index[i]:index[i + 1], 0], tsne_embedding[index[i]:index[i + 1], 1], 'o', alpha=0.5)  # alpha=0.5
    # plt.legend(['A', 'B'])
    plt.savefig('./picture/多域/ABCD/8+64+64+64/DANN/softmax/ABCD=A/' + 'TSNE' +'.png')
    # plt.savefig('./picture/多域/ABCD/8+64+64+64/DANN/Dense/ABCD=A/' + 'TSNE' + '.png')
    # plt.savefig('./picture/多域/ABCD/8+64+64+64/CPC/Dense/ABCD=A/' + 'TSNE' + '.png')
    # plt.savefig('./picture/多域/ABCD/8+64+64+64/CPC/softmax/ABCD=A/' + 'TSNE' + '.png')

    # 单域
    # plt.savefig('./picture/单域/D-C/CPC/softmax/' + 'TSNE' + '.png')
    # plt.savefig('./picture/单域/D-C/CPC/Dense/' + 'TSNE' + '.png')
    # plt.savefig('./picture/单域/D-C/DANN/Dense/' + 'TSNE' + '.png')
    # plt.savefig('./picture/单域/D-C/DANN/softmax/' + 'TSNE' + '.png')

    # plt.show()

    # plt.figure()
    # plt.title('Epoch #{}'.format(epoch + 1))
    # for i in range(1, -1, -1):
    #     plt.plot(tsne_embedding[index[i]:index[i + 1], 0], tsne_embedding[index[i]:index[i + 1], 1], '.', alpha=0.5)
    # plt.legend(['B', 'A'])
    # plt.savefig('./picture/' + 'TSNE'+'A-B' + '--' + '.png')
    # # plt.show()


