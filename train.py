import tensorflow as tf
from keras import backend as K
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from flags import FLAGS

from keras_models import mobile_net, mobile_net_split, mobile_nets_hg, dense_net
from regression_loss import loss_func, hm_loss
from optimizer import rms_prop_optimizer, adam_optimizer
from flic import data_input_fn
from evaluate_pck import evaluate_pck


def train():

    # keras session parameters
    K.set_learning_phase(FLAGS.learning_phase)

    # global step
    global_step = tf.Variable(0, trainable=False)

    # data handle
    handle = tf.placeholder(tf.string, shape=[])

    # val error
    val_error = tf.placeholder(tf.float32, [])

    # Get input pipeline
    training_dataset = data_input_fn(FLAGS.train_tfrecords, is_train=True)
    validation_dataset = data_input_fn(FLAGS.test_tfrecords, is_train=False)

    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    # get model and inference
    # model = mobile_net_split()
    # model = mobile_net()
    model = mobile_nets_hg()
    # model = dense_net()
    inference = model(next_element[0])

    # calculate loss and train step
    mean_loss = loss_func(next_element[1], inference)
    # mean_loss = hm_loss(next_element[1], inference)
    # train_step = rms_prop_optimizer(mean_loss=mean_loss, global_step=global_step)
    train_step = adam_optimizer(mean_loss=mean_loss, global_step=global_step)

    # joints_gt for prediction joints_gt
    joints_gt_tensor = next_element[1]

    # prepare session parameters
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    # Create saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    # Get all summaries
    summary_op = tf.summary.merge_all()
    summary_val = tf.summary.scalar('val_error', val_error)
    # for check pointing
    min_mean_loss = FLAGS.checkpoint_thresh
    pck_thresh = FLAGS.pck_thresh

    with tf.Session() as sess:

        # initialize session
        sess.run(init)

        # Check for saved checkpoint
        chk_pt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if chk_pt_state and chk_pt_state.model_checkpoint_path:
            saver.restore(sess, chk_pt_state.model_checkpoint_path)
            curr_step = int(sess.run(global_step))
            print('resuming from saved model: {} step: {}'
                  .format(chk_pt_state.model_checkpoint_path, curr_step, flush=True))
        else:
            curr_step = 0
            print('no checkpoint', flush=True)
            my_model = FLAGS.my_model
            model.load_weights(my_model)

        # Summary writer object
        summary_obj = tf.summary.FileWriter(FLAGS.tensorboard_dir,
                                            graph=sess.graph,
                                            filename_suffix=FLAGS.checkpoint_dir.split('/')[-1])

        # Start queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            # initialize data handles
            training_handle = sess.run(training_iterator.string_handle())
            validation_handle = sess.run(validation_iterator.string_handle())

            start_time = time.time()
            for curr_step in range(curr_step, FLAGS.total_steps):

                _, mean_loss_val = sess.run([train_step, mean_loss],
                                            feed_dict={handle: training_handle})

                if curr_step % 10 == 0:
                    duration = time.time() - start_time
                    print('step: {} time: {:.2f} mean_error: {:.8f}'.
                          format(curr_step, duration, mean_loss_val))
                    start_time = time.time()

                    if curr_step % 1000 == 0:

                        test_preds = []
                        test_gt = []
                        mean_loss_val =[]
                        mean_time = []

                        try:
                            # initialize validation iterator
                            sess.run(validation_iterator.initializer)
                            count = 0

                            while True:
                                # start_time = time.time()
                                # predictions = sess.run(inference, feed_dict={handle: validation_handle})
                                # mean_time.append(time.time() - start_time)

                                inf_time = time.time()
                                predictions, joints_gt, single_mean_loss = \
                                    sess.run([inference, joints_gt_tensor, mean_loss],
                                             feed_dict={handle: validation_handle})
                                mean_time.append(time.time() - inf_time)

                                test_preds.extend(predictions)
                                test_gt.extend(joints_gt)
                                mean_loss_val.append(single_mean_loss)
                                count += 1
                                print('\r example: {}'.format(count), end='')

                        except tf.errors.OutOfRangeError:
                            print('\nmean_time', np.mean(np.array(mean_time)))
                            pass

                        pck = evaluate_pck(joints_gt_batch=test_gt,
                                           inference_batch=test_preds,
                                           batch_size=FLAGS.test_batch_size)

                        print('validation mse {:.8f}'.format(np.mean(mean_loss_val)))
                        summary = sess.run(summary_val, feed_dict={val_error: np.mean(mean_loss_val)})
                        summary_obj.add_summary(summary, curr_step)

                        if pck > pck_thresh:
                            pck_thresh = pck
                            saver.save(sess,
                                       FLAGS.checkpoint_dir + '/model.ckpt',
                                       global_step=curr_step)
                            print('saved model for step: {} for new min pck: {:.5f}'
                                  .format(curr_step, pck), flush=True)

                        if curr_step % 1000 == 0:

                            summary = sess.run(summary_op, feed_dict={handle: training_handle})
                            summary_obj.add_summary(summary, curr_step)
                            print('saved summary for step %d' % curr_step, flush=True)

        except KeyboardInterrupt as e:
                model.save(FLAGS.my_model)
                raise e

        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()


if __name__ == '__main__':
    train()

