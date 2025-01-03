import tensorflow as tf

from flags import FLAGS


def rms_prop_optimizer(mean_loss, global_step):

    print('using rms prop optimizer')
    trainable_var_list = tf.trainable_variables()

    # Gradient descent through RMS prop optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate,
                                          decay=FLAGS.decay,
                                          momentum=FLAGS.momentum,
                                          epsilon=FLAGS.epsilon)

    grads = optimizer.compute_gradients(mean_loss, var_list=trainable_var_list)
    train_step = optimizer.apply_gradients(grads_and_vars=grads,
                                         global_step=global_step)

    # Maintain moving average
    ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ema_decay)
    maintain_averages_op = ema.apply(trainable_var_list)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/grads', grad)

    with tf.control_dependencies([train_step]):
        train_op = tf.group(maintain_averages_op)

    return train_op


def adam_optimizer(mean_loss, global_step):

    print('using adam optimizer')
    trainable_var_list = tf.trainable_variables()

    # Gradient descent through RMS prop optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    grads = optimizer.compute_gradients(mean_loss, var_list=trainable_var_list)
    train_step = optimizer.apply_gradients(grads_and_vars=grads,
                                           global_step=global_step)

    # Maintain moving average
    ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ema_decay)
    maintain_averages_op = ema.apply(trainable_var_list)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/grads', grad)

    with tf.control_dependencies([train_step]):
        train_op = tf.group(maintain_averages_op)

    return train_op

