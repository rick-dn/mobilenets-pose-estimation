"""MobileNet v1 models for Keras.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNets support any input size greater than 32 x 32, with larger image sizes
offering better performance.
The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 16 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.75, 0.5 and 0.25.
For each of these `alpha` values, weights for 4 different input image sizes
are provided (224, 192, 160, 128).

The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------

The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------

The weights for all 16 models are obtained and translated
from TensorFlow checkpoints found at
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import add
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D, UpSampling2D
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Lambda, Concatenate
from keras import backend as K


BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'


def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf')


class DepthwiseConv2D(Conv2D):
    """Depthwise separable 2D convolution.

    Depthwise Separable convolutions consists in performing
    just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.

    # Arguments
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `'valid'` or `'same'` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be 'channels_last'.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. 'linear' activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its 'activation').
            (see [regularizer](../regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `[batch, channels, rows, cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, rows, cols, channels]` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config


def MobileNet_HourGlass(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000):
    """Instantiates the MobileNet architecture.

    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.

    To load a MobileNet model via `load_model`, import the custom
    objects `relu6` and `DepthwiseConv2D` and pass them to the
    `custom_objects` parameter.
    E.g.
    model = load_model('mobilenet.h5', custom_objects={
                       'relu6': mobilenet.relu6,
                       'DepthwiseConv2D': mobilenet.DepthwiseConv2D})

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution
            (also called the resolution multiplier)
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    if K.backend() != 'tensorflow':
        raise RuntimeError('Only TensorFlow backend is currently supported, '
                           'as other backends do not support '
                           'depthwise convolution.')

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if K.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            raise ValueError('If imagenet weights are being loaded, '
                             'input must have a static square shape (one of '
                             '(128,128), (160,160), (192,192), or (224, 224)).'
                             ' Input shape provided = %s' % (input_shape,))

    if K.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    # x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    #
    # x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
    #                           strides=(2, 2), block_id=2)
    # x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    #
    # x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
    #                           strides=(2, 2), block_id=4)
    # x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    #
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
    #                           strides=(2, 2), block_id=6)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    #
    # x = _depthwise_conv_block_upscale(x, 256, alpha, depth_multiplier, block_id=120)
    # x = _depthwise_conv_block_upscale(x, 128, alpha, depth_multiplier, block_id=130)
    # x = _depthwise_conv_block_upscale(x, 64, alpha, depth_multiplier, block_id=140)

    #####################################last change 08#################################
    # x0 = Lambda(lambda x: x[:, :, :, 0:52])(x)
    # x1 = Lambda(lambda x: x[:, :, :, 52:105])(x)
    # x2 = Lambda(lambda x: x[:, :, :, 105:175])(x)
    # x3 = Lambda(lambda x: x[:, :, :, 175:227])(x)
    # x4 = Lambda(lambda x: x[:, :, :, 227:280])(x)
    # x5 = Lambda(lambda x: x[:, :, :, 280:350])(x)
    # x6 = Lambda(lambda x: x[:, :, :, 350:402])(x)
    # x7 = Lambda(lambda x: x[:, :, :, 402:455])(x)
    # # x67 = Lambda(lambda x: x[:, :, :, 350:455])(x)
    # x8 = Lambda(lambda x: x[:, :, :, 455:475])(x)
    # x9 = Lambda(lambda x: x[:, :, :, 475:495])(x)
    # x10 = Lambda(lambda x: x[:, :, :, 495:512])(x)
    #
    # # filter size 256
    # x0 = _depthwise_conv_block_upscale(x0, 25, alpha, depth_multiplier, block_id=120)
    # x1 = _depthwise_conv_block_upscale(x1, 27, alpha, depth_multiplier, block_id=121)
    # x2 = _depthwise_conv_block_upscale(x2, 35, alpha, depth_multiplier, block_id=122)
    # x3 = _depthwise_conv_block_upscale(x3, 25, alpha, depth_multiplier, block_id=123)
    # x4 = _depthwise_conv_block_upscale(x4, 27, alpha, depth_multiplier, block_id=124)
    # x5 = _depthwise_conv_block_upscale(x5, 35, alpha, depth_multiplier, block_id=125)
    # x6 = _depthwise_conv_block_upscale(x6, 26, alpha, depth_multiplier, block_id=126)
    # x7 = _depthwise_conv_block_upscale(x7, 26, alpha, depth_multiplier, block_id=127)
    # # x67 = _depthwise_conv_block_upscale(x67, 105, alpha, depth_multiplier, block_id=1267)
    # x8 = _depthwise_conv_block_upscale(x8, 10, alpha, depth_multiplier, block_id=128)
    # x9 = _depthwise_conv_block_upscale(x9, 10, alpha, depth_multiplier, block_id=129)
    # x10 = _depthwise_conv_block_upscale(x10, 8, alpha, depth_multiplier, block_id=1210)
    #
    # # filter size 256
    # x0 = _depthwise_conv_block_upscale(x0, 25, alpha, depth_multiplier, block_id=130)
    # x1 = _depthwise_conv_block_upscale(x1, 27, alpha, depth_multiplier, block_id=131)
    # x2 = _depthwise_conv_block_upscale(x2, 35, alpha, depth_multiplier, block_id=132)
    # x3 = _depthwise_conv_block_upscale(x3, 25, alpha, depth_multiplier, block_id=133)
    # x4 = _depthwise_conv_block_upscale(x4, 27, alpha, depth_multiplier, block_id=134)
    # x5 = _depthwise_conv_block_upscale(x5, 35, alpha, depth_multiplier, block_id=135)
    # x6 = _depthwise_conv_block_upscale(x6, 26, alpha, depth_multiplier, block_id=136)
    # x7 = _depthwise_conv_block_upscale(x7, 26, alpha, depth_multiplier, block_id=137)
    # # x67 = _depthwise_conv_block_upscale(x67, 105, alpha, depth_multiplier, block_id=1367)
    # x8 = _depthwise_conv_block_upscale(x8, 10, alpha, depth_multiplier, block_id=138)
    # x9 = _depthwise_conv_block_upscale(x9, 10, alpha, depth_multiplier, block_id=139)
    # x10 = _depthwise_conv_block_upscale(x10, 8, alpha, depth_multiplier, block_id=1310)
    #####################################last change 08#################################

    # x = _depthwise_conv_block_upscale(x, 256, alpha, depth_multiplier, block_id=120)
    # x = _depthwise_conv_block_upscale(x, 128, alpha, depth_multiplier, block_id=121)

    # x0 = Lambda(lambda x: x[:, :, :, 0:175])(x)
    # x1 = Lambda(lambda x: x[:, :, :, 175:350])(x)
    # x2 = Lambda(lambda x: x[:, :, :, 350:455])(x)
    # x3 = Lambda(lambda x: x[:, :, :, 455:512])(x)
    #
    # # x0 = _depthwise_conv_block_upscale(x0, 175, alpha, depth_multiplier, block_id=120)
    # # x1 = _depthwise_conv_block_upscale(x1, 175, alpha, depth_multiplier, block_id=121)
    # # x2 = _depthwise_conv_block_upscale(x2, 105, alpha, depth_multiplier, block_id=122)
    # # x3 = _depthwise_conv_block_upscale(x3, 57, alpha, depth_multiplier, block_id=123)
    #
    # # x0 = _depthwise_conv_block_upscale(x0, 175, alpha, depth_multiplier, block_id=120)
    # # x1 = _depthwise_conv_block_upscale(x1, 175, alpha, depth_multiplier, block_id=121)
    # # x2 = _depthwise_conv_block_upscale(x2, 105, alpha, depth_multiplier, block_id=122)
    # # x3 = _depthwise_conv_block_upscale(x3, 57, alpha, depth_multiplier, block_id=123)
    # #
    # # x0 = _depthwise_conv_block_upscale(x0, 87, alpha, depth_multiplier, block_id=130)
    # # x1 = _depthwise_conv_block_upscale(x1, 88, alpha, depth_multiplier, block_id=131)
    # # x2 = _depthwise_conv_block_upscale(x2, 53, alpha, depth_multiplier, block_id=132)
    # # x3 = _depthwise_conv_block_upscale(x3, 28, alpha, depth_multiplier, block_id=133)
    #
    # #  256 filters size 28 to be published
    # # x0 = _depthwise_conv_block_upscale(x0, 87, alpha, depth_multiplier, block_id=120)
    # # x1 = _depthwise_conv_block_upscale(x1, 88, alpha, depth_multiplier, block_id=121)
    # # x2 = _depthwise_conv_block_upscale(x2, 53, alpha, depth_multiplier, block_id=122)
    # # x3 = _depthwise_conv_block_upscale(x3, 28, alpha, depth_multiplier, block_id=123)
    #
    # # 128 filters size 56 to be published
    # # x0 = _depthwise_conv_block_upscale(x0, 43, alpha, depth_multiplier, block_id=130)
    # # x1 = _depthwise_conv_block_upscale(x1, 44, alpha, depth_multiplier, block_id=131)
    # # x2 = _depthwise_conv_block_upscale(x2, 27, alpha, depth_multiplier, block_id=132)
    # # x3 = _depthwise_conv_block_upscale(x3, 14, alpha, depth_multiplier, block_id=133)
    #
    # # split 2.2
    # # ensure split matches with last split 128 with split
    # x00 = Lambda(lambda x: x[:, :, :, 0:13])(x0)
    # x01 = Lambda(lambda x: x[:, :, :, 13:26])(x0)
    # x02 = Lambda(lambda x: x[:, :, :, 26:43])(x0)
    #
    # x13 = Lambda(lambda x: x[:, :, :, 0:13])(x1)
    # x14 = Lambda(lambda x: x[:, :, :, 13:26])(x1)
    # x15 = Lambda(lambda x: x[:, :, :, 26:44])(x1)
    #
    # x26 = Lambda(lambda x: x[:, :, :, 0:13])(x2)
    # x27 = Lambda(lambda x: x[:, :, :, 13:27])(x2)
    # # x267 = Lambda(lambda x: x[:, :, :, 0:27])(x2)
    #
    # x38 = Lambda(lambda x: x[:, :, :, 0:5])(x3)
    # x39 = Lambda(lambda x: x[:, :, :, 5:10])(x3)
    # x310 = Lambda(lambda x: x[:, :, :, 10:14])(x3)
    #
    # x00 = Conv2D(1, (1, 1), padding='same', name='conv_pred0')(x00)
    # x01 = Conv2D(1, (1, 1), padding='same', name='conv_preds1')(x01)
    # x02 = Conv2D(1, (1, 1), padding='same', name='conv_preds2')(x02)
    # x13 = Conv2D(1, (1, 1), padding='same', name='conv_preds3')(x13)
    # x14 = Conv2D(1, (1, 1), padding='same', name='conv_preds4')(x14)
    # x15 = Conv2D(1, (1, 1), padding='same', name='conv_preds5')(x15)
    # x26 = Conv2D(1, (1, 1), padding='same', name='conv_preds6')(x26)
    # x27 = Conv2D(1, (1, 1), padding='same', name='conv_preds7')(x27)
    # # x267 = Conv2D(2, (1, 1), padding='same', name='conv_preds7')(x267)
    # x38 = Conv2D(1, (1, 1), padding='same', name='conv_preds8')(x38)
    # x39 = Conv2D(1, (1, 1), padding='same', name='conv_preds9')(x39)
    # x310 = Conv2D(1, (1, 1), padding='same', name='conv_preds10')(x310)

    # x00 = Conv2D(1, (1, 1), padding='same', name='conv_pred0')(x0)
    # x01 = Conv2D(1, (1, 1), padding='same', name='conv_preds1')(x0)
    # x02 = Conv2D(1, (1, 1), padding='same', name='conv_preds2')(x0)
    # x13 = Conv2D(1, (1, 1), padding='same', name='conv_preds3')(x1)
    # x14 = Conv2D(1, (1, 1), padding='same', name='conv_preds4')(x1)
    # x15 = Conv2D(1, (1, 1), padding='same', name='conv_preds5')(x1)
    # x26 = Conv2D(1, (1, 1), padding='same', name='conv_preds6')(x2)
    # x27 = Conv2D(1, (1, 1), padding='same', name='conv_preds7')(x2)
    # x38 = Conv2D(1, (1, 1), padding='same', name='conv_preds8')(x3)
    # x39 = Conv2D(1, (1, 1), padding='same', name='conv_preds9')(x3)
    # x310 = Conv2D(1, (1, 1), padding='same', name='conv_preds10')(x3)

    # x = [x00, x01, x02, x13, x14, x15, x26, x27, x38, x39, x310]
    # # x = [x00, x01, x02, x13, x14, x15, x267, x38, x39, x310]
    # x = Concatenate()(x)

    #############last change 08##############
    # x0 = Conv2D(1, (1, 1), padding='same', name='conv_pred0')(x0)
    # x1 = Conv2D(1, (1, 1), padding='same', name='conv_preds1')(x1)
    # x2 = Conv2D(1, (1, 1), padding='same', name='conv_preds2')(x2)
    # x3 = Conv2D(1, (1, 1), padding='same', name='conv_preds3')(x3)
    # x4 = Conv2D(1, (1, 1), padding='same', name='conv_preds4')(x4)
    # x5 = Conv2D(1, (1, 1), padding='same', name='conv_preds5')(x5)
    # x6 = Conv2D(1, (1, 1), padding='same', name='conv_preds6')(x6)
    # x7 = Conv2D(1, (1, 1), padding='same', name='conv_preds7')(x7)
    # x8 = Conv2D(1, (1, 1), padding='same', name='conv_preds8')(x8)
    # x9 = Conv2D(1, (1, 1), padding='same', name='conv_preds9')(x9)
    # x10 = Conv2D(1, (1, 1), padding='same', name='conv_preds10')(x10)
    #
    # x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    # # x = [x0, x1, x2, x3, x4, x5, x67, x8, x9, x10]
    # x = Concatenate()(x)
    #############last change 08##############

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))

    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    xr3 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=103)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    xr2 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=102)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    xr1 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=101)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    xr0 = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=100)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = add([x, xr0])

    x = _depthwise_conv_block_upscale(x, 256, alpha, depth_multiplier, block_id=200)
    x = add([x, xr1])

    x = _depthwise_conv_block_upscale(x, 256, alpha, depth_multiplier, block_id=201)
    x = add([x, xr2])

    x = _depthwise_conv_block_upscale(x, 256, alpha, depth_multiplier, block_id=202)
    x = add([x, xr3])

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=203)

    x = Conv2D(11, (1, 1), padding='same', name='conv_pred_hm_1')(x)
    # x = Conv2D(11, (1, 1), padding='same', name='conv_pred_hm_2')(x)

    # if include_top:
    #     if K.image_data_format() == 'channels_first':
    #         shape = (int(1024 * alpha), 1, 1)
    #     else:
    #         shape = (1, 1, int(1024 * alpha))
    #
    #     x = GlobalAveragePooling2D()(x)
    #     x = Reshape(shape, name='reshape_1')(x)
    #     x = Dropout(dropout, name='dropout')(x)
    #     x = Conv2D(classes, (1, 1),
    #                padding='same', name='conv_preds')(x)
    #     x = Activation('softmax', name='act_softmax')(x)
    #     x = Reshape((classes,), name='reshape_2')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True, skip_mismatch=True, reshape=True)
    elif weights is not None:
        model.load_weights(weights)

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).

    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each 6layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def _depthwise_conv_block_upscale(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = UpSampling2D((2, 2))(inputs)
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)