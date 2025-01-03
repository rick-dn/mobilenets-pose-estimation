from keras.models import Model
from keras.layers import Flatten
from keras.layers import Input, Dense, Conv2D
from keras.layers import Lambda, Concatenate
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121

from mobile_nets_hg import MobileNet_HourGlass

from flags import FLAGS

output_size = (FLAGS.no_of_joints + FLAGS.no_of_dense_joints) * \
              FLAGS.coord_per_joint


def dense_net():

    model = DenseNet121(include_top=False,
                     weights=None,
                     input_shape=(FLAGS.resize_input_image,
                                  FLAGS.resize_input_image, 3))
                     # blocks=[3, 6, 12, 8])

    x = model.output
    x = Flatten()(x)
    x = Dense(output_size, activation=None, name='predictions')(x)

    model = Model(inputs=model.input, outputs=x)
    model.save(FLAGS.my_model)

    print(model.summary())

    return model


def mobile_net():

    model = MobileNet(include_top=False,
                      weights='imagenet',
                      input_shape=(FLAGS.resize_input_image,
                                   FLAGS.resize_input_image, 3))

    x = model.output
    x = Flatten()(x)
    x = Dense(output_size, activation=None, name='predictions')(x)

    model = Model(inputs=model.input, outputs=x)
    model.save(FLAGS.my_model)

    print(model.summary())

    return model


def mobile_nets_hg():

    model = MobileNet_HourGlass(include_top=False,
                                weights='imagenet',
                                input_shape=(FLAGS.resize_input_image,
                                             FLAGS.resize_input_image, 3))

    x = model.output

    # split 2.2
    # ensure split matches with last split 512
    # x0 = Lambda(lambda x: x[:, :, :, 0:52])(x)
    # x1 = Lambda(lambda x: x[:, :, :, 52:105])(x)
    # x2 = Lambda(lambda x: x[:, :, :, 105:175])(x)
    #
    # x3 = Lambda(lambda x: x[:, :, :, 175:227])(x)
    # x4 = Lambda(lambda x: x[:, :, :, 227:280])(x)
    # x5 = Lambda(lambda x: x[:, :, :, 280:350])(x)
    #
    # x6 = Lambda(lambda x: x[:, :, :, 350:455])(x)
    # # x7 = Lambda(lambda x: x[:, :, :, 402:455])(x)
    #
    # x8 = Lambda(lambda x: x[:, :, :, 455:473])(x)
    # x9 = Lambda(lambda x: x[:, :, :, 473:491])(x)
    # x10 = Lambda(lambda x: x[:, :, :, 491:512])(x)

    # split 2.2
    # ensure split matches with last split 512
    # x0 = Lambda(lambda x: x[:, :, :, 0:52])(x)
    # x1 = Lambda(lambda x: x[:, :, :, 52:105])(x)
    # x2 = Lambda(lambda x: x[:, :, :, 105:175])(x)
    #
    # x3 = Lambda(lambda x: x[:, :, :, 175:227])(x)
    # x4 = Lambda(lambda x: x[:, :, :, 227:280])(x)
    # x5 = Lambda(lambda x: x[:, :, :, 280:350])(x)
    #
    # x6 = Lambda(lambda x: x[:, :, :, 350:455])(x)
    # # x7 = Lambda(lambda x: x[:, :, :, 402:455])(x)
    #
    # x8 = Lambda(lambda x: x[:, :, :, 455:473])(x)
    # x9 = Lambda(lambda x: x[:, :, :, 473:491])(x)
    # x10 = Lambda(lambda x: x[:, :, :, 491:512])(x)

    # split 2.2
    # ensure split matches with last split 256
    # x0 = Lambda(lambda x: x[:, :, :, 0:26])(x)
    # x1 = Lambda(lambda x: x[:, :, :, 26:52])(x)
    # x2 = Lambda(lambda x: x[:, :, :, 52:87])(x)
    #
    # x3 = Lambda(lambda x: x[:, :, :, 87:113])(x)
    # x4 = Lambda(lambda x: x[:, :, :, 113:139])(x)
    # x5 = Lambda(lambda x: x[:, :, :, 139:175])(x)
    #
    # x6 = Lambda(lambda x: x[:, :, :, 175:228])(x)
    # # x7 = Lambda(lambda x: x[:, :, :, 402:455])(x)
    #
    # x8 = Lambda(lambda x: x[:, :, :, 228:237])(x)
    # x9 = Lambda(lambda x: x[:, :, :, 237:246])(x)
    # x10 = Lambda(lambda x: x[:, :, :, 246:256])(x)

    # split 2.2
    # ensure split matches with last split 256
    # x0 = Lambda(lambda x: x[:, :, :, 0:13])(x)
    # x1 = Lambda(lambda x: x[:, :, :, 13:26])(x)
    # x2 = Lambda(lambda x: x[:, :, :, 26:43])(x)
    #
    # x3 = Lambda(lambda x: x[:, :, :, 43:56])(x)
    # x4 = Lambda(lambda x: x[:, :, :, 56:69])(x)
    # x5 = Lambda(lambda x: x[:, :, :, 69:87])(x)
    #
    # x6 = Lambda(lambda x: x[:, :, :, 87:100])(x)
    # x7 = Lambda(lambda x: x[:, :, :, 100:114])(x)
    #
    # x8 = Lambda(lambda x: x[:, :, :, 114:119])(x)
    # x9 = Lambda(lambda x: x[:, :, :, 119:124])(x)
    # x10 = Lambda(lambda x: x[:, :, :, 124:128])(x)
    #
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

    # x0 = Flatten()(x0)
    # x1 = Flatten()(x1)
    # x2 = Flatten()(x2)
    # x3 = Flatten()(x3)
    # x4 = Flatten()(x4)
    # x5 = Flatten()(x5)
    # x6 = Flatten()(x6)
    # x7 = Flatten()(x7)
    # x8 = Flatten()(x8)
    # x9 = Flatten()(x9)
    # x10 = Flatten()(x10)
    #
    # x0 = Dense(2, activation=None, name='predictions0')(x0)
    # x1 = Dense(2, activation=None, name='predictions1')(x1)
    # x2 = Dense(2, activation=None, name='predictions2')(x2)
    # x3 = Dense(2, activation=None, name='predictions3')(x3)
    # x4 = Dense(2, activation=None, name='predictions4')(x4)
    # x5 = Dense(2, activation=None, name='predictions5')(x5)
    # x6 = Dense(2, activation=None, name='predictions6')(x6)
    # x7 = Dense(2, activation=None, name='predictions7')(x7)
    # x8 = Dense(2, activation=None, name='predictions8')(x8)
    # x9 = Dense(2, activation=None, name='predictions9')(x9)
    # x10 = Dense(2, activation=None, name='predictions10')(x10)

    # x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    # # x = [x0, x1, x2, x3, x4, x5, x6, x8, x9, x10]
    # x = Concatenate()(x)

    model = Model(inputs=model.inputs, outputs=x)

    print(model.summary())
    model.save(FLAGS.my_model)

    return model


def mobile_net_split():
    model = MobileNet(include_top=False,
                      weights='imagenet',
                      input_shape=(FLAGS.resize_input_image,
                                   FLAGS.resize_input_image, 3))

    x = model.output

    # split 2.2
    x0 = Lambda(lambda x: x[:, :, :, 0:105])(x)
    x1 = Lambda(lambda x: x[:, :, :, 105:210])(x)
    x2 = Lambda(lambda x: x[:, :, :, 210:350])(x)
    x3 = Lambda(lambda x: x[:, :, :, 350:455])(x)
    x4 = Lambda(lambda x: x[:, :, :, 459:560])(x)
    x5 = Lambda(lambda x: x[:, :, :, 560:700])(x)
    x6 = Lambda(lambda x: x[:, :, :, 700:805])(x)
    x7 = Lambda(lambda x: x[:, :, :, 805:910])(x)
    x8 = Lambda(lambda x: x[:, :, :, 910:945])(x)
    x9 = Lambda(lambda x: x[:, :, :, 945:980])(x)
    x10 = Lambda(lambda x: x[:, :, :, 980:1024])(x)

    x0 = Flatten()(x0)
    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    x3 = Flatten()(x3)
    x4 = Flatten()(x4)
    x5 = Flatten()(x5)
    x6 = Flatten()(x6)
    x7 = Flatten()(x7)
    x8 = Flatten()(x8)
    x9 = Flatten()(x9)
    x10 = Flatten()(x10)

    x0 = Dense(2, activation=None, name='predictions0')(x0)
    x1 = Dense(24, activation=None, name='predictions1')(x1)
    x2 = Dense(2, activation=None, name='predictions2')(x2)
    x3 = Dense(2, activation=None, name='predictions3')(x3)
    x4 = Dense(24, activation=None, name='predictions4')(x4)
    x5 = Dense(2, activation=None, name='predictions5')(x5)
    x6 = Dense(2, activation=None, name='predictions6')(x6)
    x7 = Dense(2, activation=None, name='predictions7')(x7)
    x8 = Dense(2, activation=None, name='predictions8')(x8)
    x9 = Dense(2, activation=None, name='predictions9')(x9)
    x10 = Dense(2, activation=None, name='predictions10')(x10)

    x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    x = Concatenate()(x)

    model = Model(inputs=model.inputs, outputs=x)

    print(model.summary())
    model.save(FLAGS.my_model)

    return model


def main():
    mobile_nets_hg()


if __name__ == '__main__':
    main()

