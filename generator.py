from keras.layers import Input, Dense, BatchNormalization, initializers, Concatenate, Lambda
from keras.layers.core import Activation, Reshape, Flatten

from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose

from keras.models import Model

def generator_model(size_z, size_age_label, size_mini_map, size_kernel, size_gen, num_input_channels, num_gen_channels):

    kernel_initializer = initializers.random_normal(stddev=0.02)
    bias_initializer = initializers.constant(value=0.0)

    #Input layer
    input_z = Input(shape=(size_z, ))
    input_age_label = Input(shape=(size_age_label, ))
    current = Concatenate(axis=-1)([input_z, input_age_label])

    # fc layer
    name = 'G_fc'
    current = Dense(
        units=size_mini_map * size_mini_map * size_gen,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=name)(current)
    # Reshape
    current = Reshape(target_shape=(size_mini_map, size_mini_map, size_gen))(current)
    # BatchNormalization
    # current = Lambda(tf.contrib.layers.batch_norm, output_shape=(size_mini_map, size_mini_map, size_gen),
    #                  arguments={'decay': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    # Activation
    current = Activation(activation='relu')(current)

    # deconv layers with stride 2
    num_layers = len(num_gen_channels)
    size_image = size_mini_map
    for i in range(num_layers - 1):
        name = 'G_deconv' + str(i)
        current = Conv2DTranspose(
            filters=num_gen_channels[i],
            kernel_size=(size_kernel, size_kernel),
            padding='same',
            strides=(2, 2),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=name)(current)
        # size_image = size_image * 2
        # current = Lambda(tf.contrib.layers.batch_norm, output_shape=(size_image, size_image, int(current.shape[3])),
        #                  arguments={'decay':0.9, 'epsilon': 1e-5, 'scale':True})(current)
        current = Activation(activation='relu')(current)

    # final layer of generator---> activation: tanh
    name = 'G_deconv' + str(i + 1)
    current = Conv2DTranspose(
        filters=num_gen_channels[-1],
        kernel_size=(size_kernel,size_kernel),
        padding='same',
        strides=(2, 2),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=name)(current)
    current = Activation('tanh')(current)

    # output
    return Model(inputs=[input_z, input_age_label], outputs=current)