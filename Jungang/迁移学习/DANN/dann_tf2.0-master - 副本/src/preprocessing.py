import numpy as np
import tensorflow as tf

# TRAIN_NUM = 60000
# TEST_NUM = 9000


def load_data(data_category):


    if (data_category == 'ABCD-8+64+64+64'):
        x_train = np.load('../data/多域/ABCD/ABCD-8+64+64+64/x_train.npy')
        y_train = np.load('../data/多域/ABCD/ABCD-8+64+64+64/y_train.npy')

        x_test = np.load('../data/多域/ABCD/ABCD-8+64+64+64/x_test.npy')
        y_test = np.load('../data/多域/ABCD/ABCD-8+64+64+64/y_test.npy')

    # if (data_category == 'ABCD-32+56+56+56'):
    #     x_train = np.load('../data/多域/ABCD/ABCD-32+56+56+56/x_train.npy')
    #     y_train = np.load('../data/多域/ABCD/ABCD-32+56+56+56/y_train.npy')
    #
    #     x_test = np.load('../data/多域/ABCD/ABCD-32+56+56+56/x_test.npy')
    #     y_test = np.load('../data/多域/ABCD/ABCD-32+56+56+56/y_test.npy')

    # if (data_category == 'ABCD-80+40+40+40'):
    #     x_train = np.load('../data/多域/ABCD/ABCD-80+40+40+40/x_train.npy')
    #     y_train = np.load('../data/多域/ABCD/ABCD-80+40+40+40/y_train.npy')
    #
    #     x_test = np.load('../data/多域/ABCD/ABCD-80+40+40+40/x_test.npy')
    #     y_test = np.load('../data/多域/ABCD/ABCD-80+40+40+40/y_test.npy')

    # if (data_category == 'ABCD-128+24+24+24'):
    #     x_train = np.load('../data/多域/ABCD/ABCD-128+24+24+24/x_train.npy')
    #     y_train = np.load('../data/多域/ABCD/ABCD-128+24+24+24/y_train.npy')
    #
    #     x_test = np.load('../data/多域/ABCD/ABCD-128+24+24+24/x_test.npy')
    #     y_test = np.load('../data/多域/ABCD/ABCD-128+24+24+24/y_test.npy')

    # if (data_category == 'ABCD-176+8+8+8'):
    #     x_train = np.load('../data/多域/ABCD/ABCD-176+8+8+8/x_train.npy')
    #     y_train = np.load('../data/多域/ABCD/ABCD-176+8+8+8/y_train.npy')
    #
    #     x_test = np.load('../data/多域/ABCD/ABCD-176+8+8+8/x_test.npy')
    #     y_test = np.load('../data/多域/ABCD/ABCD-176+8+8+8/y_test.npy')



    # if (data_category == 'A'):
    #     x_train = np.load('../data/单域/A/x_train.npy')
    #     y_train = np.load('../data/单域/A/y_train.npy')
    #
    #     x_test = np.load('../data/单域/A/x_test.npy')
    #     y_test = np.load('../data/单域/A/y_test.npy')



    elif (data_category == 'A'):
        x_train = np.load('../data/单域/A/x_train.npy')
        y_train = np.load('../data/单域/A/y_train.npy')

        x_test = np.load('../data/单域/A/x_test.npy')
        y_test = np.load('../data/单域/A/y_test.npy')

    # elif (data_category == 'B'):
    #     x_train = np.load('../data/单域/B/x_train.npy')
    #     y_train = np.load('../data/单域/B/y_train.npy')
    #
    #     x_test = np.load('../data/单域/B/x_test.npy')
    #     y_test = np.load('../data/单域/B/y_test.npy')


    # elif (data_category == 'C'):
    #     x_train = np.load('../data/单域/C/x_train.npy')
    #     y_train = np.load('../data/单域/C/y_train.npy')
    #
    #     x_test = np.load('../data/单域/C/x_test.npy')
    #     y_test = np.load('../data/单域/C/y_test.npy')

    # elif (data_category == 'D'):
    #     x_train = np.load('../data/单域/D/x_train.npy')
    #     y_train = np.load('../data/单域/D/y_train.npy')
    #
    #     x_test = np.load('../data/单域/D/x_test.npy')
    #     y_test = np.load('../data/单域/D/y_test.npy')


    x_train = x_train
    y_train = y_train

    x_test = x_test
    y_test = y_test

    return (x_train, y_train, x_test, y_test)

def data2dataset(x, y, data_category):

    if (data_category == 'ABCD-8+64+64+64'):
    # if (data_category == 'ABCD-32+56+56+56'):
    # if (data_category == 'ABCD-80+40+40+40'):
    # if (data_category == 'ABCD-128+24+24+24'):
    # if (data_category == 'ABCD-176+8+8+8'):
    # if (data_category == 'A'):
    # if (data_category == 'A'):
    # if (data_category == 'B'):
    # if (data_category == 'C'):
    # if (data_category == 'A'):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        # dataset = dataset.map(pad_image)
        # dataset = dataset.map(duplicate_channel)
        # dataset = dataset.map(cast)
        dataset = dataset.shuffle(len(y))


    elif (data_category == 'A'):
    # elif (data_category == 'B'):
    # elif (data_category == 'C'):
    # elif (data_category == 'D'):
    # elif (data_category == 'A' or data_category == 'SYN'):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        # dataset = dataset.map(cast)
        dataset = dataset.shuffle(len(y))

    return dataset


def prepare_dataset(source, target):

    (x_train, y_train, x_test, y_test) = load_data(source)

    (x_target, y_target, x_test_target, y_test_target) = load_data(target)

    source_train_dataset = data2dataset(x_train, y_train, source)
    source_test_dataset = data2dataset(x_test, y_test, source)
    target_dataset = data2dataset(x_target, y_target, target)
    target_test_dataset = data2dataset(x_test_target, y_test_target, target)

    source_train_dataset = source_train_dataset.batch(32)
    source_train_dataset = source_train_dataset.prefetch(3)

    source_test_dataset = source_test_dataset.batch(32)
    source_test_dataset = source_test_dataset.prefetch(3)

    target_dataset = target_dataset.batch(32)
    target_dataset = target_dataset.prefetch(3)

    target_test_dataset = target_test_dataset.batch(32)
    target_test_dataset = target_test_dataset.prefetch(3)

    return (source_train_dataset, source_test_dataset, target_dataset, target_test_dataset)

def prepare_dataset_single(data_category):

    (x_train, y_train, x_test, y_test) = load_data(data_category)

    train_dataset = data2dataset(x_train, y_train, data_category)
    test_dataset = data2dataset(x_test, y_test, data_category)

    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.prefetch(1)

    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.prefetch(1)

    return (train_dataset, test_dataset)

