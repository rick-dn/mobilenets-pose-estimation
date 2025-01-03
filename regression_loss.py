import tensorflow as tf

from flags import FLAGS


def loss_func(joints_gt, inference):
    with tf.variable_scope('squared_diff') as scope:

        # joints_gt = tf.reshape(joints_gt,
        #                        [-1, (FLAGS.no_of_joints + FLAGS.no_of_dense_joints) *
        #                         FLAGS.coord_per_joint])

        # Squared Diff
        squared_diff = tf.squared_difference(x=joints_gt, y=inference, name='squared_diff')

        # Calculate MSE from individual joint losses
        mean_loss = tf.reduce_mean(squared_diff, name='squared_diff')

    tf.summary.scalar('squared_diff', mean_loss)

    return mean_loss


def final_loss_func(mean_loss, pw_loss):

    alpha = tf.constant(FLAGS.alpha, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)

    with tf.variable_scope('final_loss') as scope:
        final_loss = tf.add(tf.scalar_mul(alpha, mean_loss),
                            tf.scalar_mul(tf.subtract(one, alpha), pw_loss))
        tf.add_to_collection('final_loss_op', final_loss)

    tf.summary.scalar("final_loss", final_loss)

    return final_loss


def hm_loss(label, inference):

    with tf.variable_scope('squared_diff') as scope:
        mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=inference, labels=label),
                                   name='cross_entropy_loss')

        tf.summary.scalar('squared_diff', mean_loss)

    return mean_loss
