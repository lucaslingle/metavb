import tensorflow as tf
import tensorflow_datasets as tfds
import os
import scipy.io
import numpy as np

SUPPORTED_DATASETS = ['mnist', 'celeb_a', 'cifar10', 'omniglot']

def get_dataset(name, hps):
    # get the dataset specified, and make a 90/10 train/validation set split from the training data.

    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise NotImplementedError('Dataset not supported.')

    if name == 'omniglot':
        ds_train, ds_val, ds_test = get_omniglot()
    else:
        train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:90])
        val_split = tfds.Split.TRAIN.subsplit(tfds.percent[90:])
        test_split = tfds.Split.TEST
        ds_train, ds_val, ds_test = tfds.load(name=name, split=[train_split, val_split, test_split])
        
    ds_train = preprocess_dataset(name, ds_train, hps)
    ds_val = preprocess_dataset(name, ds_val, hps)
    ds_test = preprocess_dataset(name, ds_test, hps)

    return ds_train, ds_val, ds_test


def preprocess_dataset(name, dataset, hps):
    f1 = lambda row: row["image"]
    f2 = lambda img: tf.cast(img, dtype=tf.int32)
    f3 = lambda img: tf.cast(img, dtype=tf.float32)

    if hps.discrete_outputs:
        # assumes all datasets to be binarized begin in the 0-255 range; while true for the ones above, take care if modifying this code
        f4 = lambda img: img * tf.constant((1.0 / 255.0))
    else:
        # assumes all continuous-valued datasets are in 0-255 range; while true for the ones above, take care if modifying this code
        f4 = lambda img: img * tf.constant((1.0 / 256.0))

    if name == 'celeb_a':
        f5a = lambda img: img[((tf.shape(img)[0]//2)-54):((tf.shape(img)[0]//2)+54), ((tf.shape(img)[1]//2)-54):((tf.shape(img)[1]//2)+54)]
        f5b = lambda img: tf.image.resize_images(img, [hps.img_height, hps.img_width])
        f5 = lambda img: f5b(f5a(img))
    else:
        f5 = lambda img: tf.image.resize_images(img, [hps.img_height, hps.img_width])

    if hps.discrete_outputs:
        f6 = lambda img: tf.round(tf.expand_dims(img[:,:,0], -1)) # static binarization for discrete data
        f7 = lambda img: tf.cast(img, dtype=tf.int32)
    else:
        f6 = lambda img: img + tf.random_uniform(minval=0.0, maxval=(1.0/256.0), shape=[hps.img_height, hps.img_width, hps.img_channels])
        f7 = lambda img: tf.cast(img, dtype=tf.float32)

    process_image = lambda row: f7(f6(f5(f4(f3(f2(f1(row)))))))
    convert_to_episode = lambda batch: tf.reshape(batch, [-1, hps.episode_len, hps.img_height, hps.img_width, hps.img_channels])

    ds = dataset.map(process_image)
    ds = ds.shuffle(buffer_size=100000, reshuffle_each_iteration=True)
    ds = ds.batch(hps.batch_size * hps.episode_len, drop_remainder=True)
    ds = ds.map(convert_to_episode)

    return ds


def get_omniglot():
    # loads the 28x28 version of Omniglot from Burda, with standard train/test. 
    # then creates a validation set from the training set, which is consistent across runs.
    fp = 'data/OMNIGLOT/chardata.mat'
    omni_raw = scipy.io.loadmat(fp)

    def reshape_data(data):
        return data.reshape((-1, 28, 28, 1))

    train_data_orig = reshape_data(omni_raw['data'].T.astype('float32')) # shape: [24000ish, 28, 28, 1]
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))   # shape: [8070, 28, 28, 1]

    num_train_orig = train_data_orig.shape[0]
    num_validation = int(0.10 * num_train_orig)

    # loaded data may not be in random order, so we best shuffle it before creating a validation set.
    # use a fixed random seed, to ensure consistent split across runs. 
    np.random.seed(123)
    perm = np.random.permutation(num_train_orig)

    train_data = train_data_orig[perm][0:(num_train_orig-num_validation)]
    validation_data = train_data_orig[perm][(num_train_orig-num_validation):]

    train_ds = tf.data.Dataset.from_tensor_slices(train_data)
    valid_ds = tf.data.Dataset.from_tensor_slices(validation_data)
    test_ds = tf.data.Dataset.from_tensor_slices(test_data)

    # tfds pipeline works great, but uses rows with 'image' key and 0-255 scale even for MNIST. 
    # to minimize variation accross preprocessing pipelines, we will be wasteful here and map everything to that format:
    standard_rows = lambda img: {'image': 255.0 * img}
    train_ds = train_ds.map(standard_rows)
    valid_ds = valid_ds.map(standard_rows)
    test_ds = test_ds.map(standard_rows)

    return train_ds, valid_ds, test_ds

