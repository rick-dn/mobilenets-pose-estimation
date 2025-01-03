import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# ################ only edit this ###################
data_dir = 'E:/Rick/datasets/FLIC/FLIC/'
project_dir = 'E:/Rick/data/mobile_net_pose_estimation_flic/'
# check_point_dir = 'mobilenets_224_17_05_dual_split_upsample_filter_red'
# check_point_dir = 'mobilenets_224_17_05_dual_split_upsample_filter_red_further_red'
# check_point_dir = 'mobilenets_224_20_05_single_split_upsample_256_128'
# check_point_dir = 'mobilenets_224_25_05_single_split_3_layers_256'
check_point_dir = 'mobilenets_flic_15_jun_hourglass'
# ###################################################

images_dir = data_dir + 'images/'
annotations = data_dir + 'examples.mat'
train_tfrecords = project_dir + 'tf_records/train_FLIC_dense.tfrecords'
test_tfrecords = project_dir + 'tf_records/test_FLIC_dense.tfrecords'
keras_model = project_dir + 'my_model.h5'
checkpoint_dir = project_dir + 'checkpoint/' + check_point_dir
log_dir = project_dir + 'log_dir'

flags.DEFINE_string('data_dir', data_dir, 'data directory')
flags.DEFINE_string('images_dir', images_dir, 'data directory')
flags.DEFINE_string('annotations', annotations, 'annotations')
flags.DEFINE_string('train_tfrecords', train_tfrecords, 'train tfrecords file')
flags.DEFINE_string('test_tfrecords', test_tfrecords, 'test tf records file')
flags.DEFINE_string('my_model', keras_model, 'model in keras format')
flags.DEFINE_string('checkpoint_dir', checkpoint_dir, 'checkpoint directory')
flags.DEFINE_string('tensorboard_dir', log_dir, 'tensorboard')


flags.DEFINE_integer('batch_size', 128, 'size of individual batches')
flags.DEFINE_integer('test_batch_size', 1016, '1008 during train batches')
flags.DEFINE_integer('resize_input_image', 224, 'network input size')
flags.DEFINE_integer('heatmap_size', 112, 'size of heatmap')
flags.DEFINE_integer('no_of_joints', 11, 'no of joints')
flags.DEFINE_integer('no_of_dense_joints', 0, 'no of extra annotations')
flags.DEFINE_integer('coord_per_joint', 2, 'x and y')
flags.DEFINE_float('low_thresh', 0.2, 'pck thresh')

flags.DEFINE_bool('learning_phase', True, 'train or test?')
flags.DEFINE_integer('total_steps', 500001, 'number of steps')

flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('decay', 0.9, 'optimizer decay')
flags.DEFINE_float('momentum', 0.09, 'optimizer momentum')
flags.DEFINE_float('epsilon', 1.0, 'optimizer epsilon')
flags.DEFINE_float('ema_decay', 0.9, 'ema decay')

flags.DEFINE_float('checkpoint_thresh', 0.002, 'save if mse below thresh')
# flags.DEFINE_float('pck_thresh', 0.97387, 'save if above thresh')
flags.DEFINE_float('pck_thresh', 0.90, 'save if above thresh')