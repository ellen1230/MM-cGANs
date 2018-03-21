from keras.layers import Input, Dense, BatchNormalization, initializers, Concatenate
from keras.layers.core import Activation, Reshape, Flatten, Lambda
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Model
from ops import duplicate_conv, lrelu
import tensorflow as tf

# map the label of age + gender to size (size_image)
# def label_model(size_label, size_image):
#     inputs = Input(shape=(size_label, ))
#     outputs = Dense(units=size_image*size_image, activation='tanh')(inputs)
#     outputs = Reshape(target_shape=(size_image, size_image, 1))(outputs)
#     return Model(inputs=inputs, outputs=outputs)

# add the label of age + gender to input layer to align the structure of discriminator_img
def encoder_model(size_image, size_age_label, num_input_channels, size_kernel, size_z, num_encoder_channels):
    # map the label of age + gender to size (size_image)

    kernel_initializer = initializers.truncated_normal(stddev=0.02)
    bias_initializer = initializers.constant(value=0.0)

    # input of input images
    input_images = Input(shape=(size_image, size_image, num_input_channels))

    # input of age labels (use {function dupilicate_conv} to time the age_labels to match with images, then concatenate )
    input_ages_conv = Input(shape=(1, 1, size_age_label)) # (1, 1, 10*tile_ratio)
    input_ages_conv_repeat = Lambda(duplicate_conv, output_shape=(size_image, size_image, size_age_label),
                                   arguments={'times': size_image})(input_ages_conv) #(128, 128, 10*tile_ratio)

    current = Concatenate(axis=-1)([input_images, input_ages_conv_repeat])

    # E_conv layer + Batch Normalization
    num_layers = len(num_encoder_channels)


    for i in range(num_layers):
        name = 'E_conv' + str(i)
        current = Conv2D(
            filters=num_encoder_channels[i],
            kernel_size=(size_kernel, size_kernel),
            strides=(2, 2),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=name)(current)
        size_image = int(size_image/2)
        current = Lambda(lrelu, output_shape=(size_image, size_image, int(current.shape[3])))(current)
        # current = Lambda(tf.contrib.layers.batch_norm, output_shape=(size_image, size_image, int(current.shape[3])),
        #                  arguments={'decay':0.9, 'epsilon': 1e-5, 'scale':True})(current)


    # reshape
    current = Flatten()(current)

    # fully connection layer
    kernel_initializer = initializers.random_normal(stddev=0.02)
    name = 'E_fc'
    current = Dense(units=size_z, activation='tanh',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name= name)(current)

    # output
    return Model(inputs=[input_images, input_ages_conv], outputs=current)
