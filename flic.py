import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import toimage
from tqdm import tqdm
import tensorflow as tf

from flic_pre_processing import LSPPreProcess
from flags import FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecords(tfrecords_file, is_train):

    print('Writing', tfrecords_file)
    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    count = 0
    for image, joints_gt in tqdm(LSPPreProcess(is_train=is_train)):

        if image is None or joints_gt is None:
            continue

        count += 1
        # Serialise and write data
        example = tf.train.Example(features=tf.train.Features(feature={
            'labels': _bytes_feature(joints_gt.tostring()),
            'image_array': _bytes_feature(image.tostring())}))

        writer.write(example.SerializeToString())
    writer.close()
    print('{}, {} done'.format(count, tfrecords_file))


def parse_fn(example):

    label_shape = tf.stack([FLAGS.no_of_joints + FLAGS.no_of_dense_joints,
                            FLAGS.coord_per_joint])
    image_shape = tf.stack([FLAGS.resize_input_image,
                            FLAGS.resize_input_image, 3])

    feature = {
        "image_array": tf.FixedLenFeature([], tf.string),
        "labels": tf.FixedLenFeature([], tf.string),
    }

    parsed = tf.parse_single_example(example, feature)

    # Convert from serialized data to proper dimensions
    images = tf.decode_raw(parsed['image_array'], tf.float32)
    images = tf.reshape(images, image_shape)

    labels = tf.decode_raw(parsed['labels'], tf.float32)
    labels = tf.reshape(labels, label_shape)

    return images, labels


def data_input_fn(tf_record, is_train):

    dataset = tf.data.TFRecordDataset(tf_record)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.map(
        lambda images, labels: tf.py_func(
            LSPPreProcess.on_the_fly,
            [images, labels, is_train],
            [tf.float32, tf.float32]))

    if is_train:
        dataset = dataset.batch(batch_size=FLAGS.batch_size)
        dataset = dataset.repeat(None)
    else:
        dataset = dataset.batch(1)
        dataset = dataset.repeat(1)

        # dataset = dataset.prefetch(buffer_size=1)

    return dataset


# def val_input_fn(tf_record):
#
#     dataset = tf.data.TFRecordDataset(tf_record)
#     dataset = dataset.map(map_func=parse_fn)
#     dataset = dataset.map(
#         lambda images, labels: tf.py_func(
#             LSPPreProcess.on_the_fly,
#             [images, labels],
#             [tf.float32, tf.float32, tf.bool]))
#     dataset = dataset.batch(1)
#     dataset = dataset.repeat(1)
#
#     return dataset


def main():

    # generate_tfrecords(FLAGS.train_tfrecords, True)
    # generate_tfrecords(FLAGS.test_tfrecords, False)

    # dataset = data_input_fn(FLAGS.train_tfrecords, is_train=True)
    dataset = data_input_fn(FLAGS.test_tfrecords, is_train=False)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:

        try:

            while True:

                image_batch, joints_gt_batch = sess.run(next_element)

                for count, zipped in enumerate(zip(image_batch, joints_gt_batch)):

                    image, joints_gt = zipped

                    # r = joints_gt[[1, 15], 0] * FLAGS.resize_input_image
                    # theta = joints_gt[[1, 15], 1] * 360

                    print(joints_gt.shape)
                    # plt.imshow(joints_gt[:, :, 0])
                    # plt.show()
                    # print(joints_gt)
                    joints_gt = LSPPreProcess.get_joint_from_hm(joints_gt)
                    # image, _ = LSPPreProcess.resize_img(image, joints_gt, 56)
                    # print(joints_gt)
                    joints_gt = LSPPreProcess.resize_joints(joints_gt,
                                                            FLAGS.resize_input_image,
                                                            FLAGS.heatmap_size)
                    # print(joints_gt)

                    # joints_gt = LSPPreProcess.coord_de_normalize(joints_gt)
                    # joints_gt[[1, 15], 0] = r
                    # joints_gt[[1, 15], 1] = theta

                    # print('count', count, r, theta)

                    print(joints_gt)
                    plt.imshow(toimage(image))
                    # plt.scatter(np.delete(joints_gt[:, 0], [1, 15]),
                    #             np.delete(joints_gt[:, 1], [1, 15]))
                    plt.scatter(joints_gt[:, 1], joints_gt[:, 0])
                    plt.show()

                print('next batch:')

        except tf.errors.OutOfRangeError:
            print('finished')
            pass
        except KeyboardInterrupt:
            print('geeeee')


if __name__ == '__main__':
    main()




