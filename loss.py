from keras.layers import Input, Lambda, Concatenate
from keras.models import Model
from keras.layers.merge import Subtract, subtract
import tensorflow as K
import numpy as np
import keras.losses

# size: all output
def get_loss_all(size_z, size_image, num_input_channels, weights):
    output_E = Input(shape=(size_z, ))
    output_E_center = Input(shape=(size_z, ))
    output_E_value = Subtract()([output_E, output_E_center])

    output_D_real = Input(shape=(1, ))
    output_D_fake = Input(shape=(1, ))
    output_EGD = Input(shape=(1, ))

    output_image = Input(shape=(size_image, size_image, num_input_channels))
    output_re_image = Input(shape=(size_image, size_image, num_input_channels))
    output_image_value = Subtract()([output_image, output_re_image])


    loss_E = Lambda(get_loss_latant, output_shape=(None,))(output_E_value)
    loss_D_real = Lambda(get_loss_binary, arguments={'binary': K.ones_like(output_D_real)}, output_shape=(None,))(output_D_real)
    loss_D_fake = Lambda(get_loss_binary, arguments={'binary': K.zeros_like(output_D_real)}, output_shape=(None,))(output_D_fake)
    loss_EGD = Lambda(get_loss_binary, arguments={'binary': K.ones_like(output_EGD)}, output_shape=(None,))(output_EGD)
    loss_image = Lambda(get_loss_all, output_shape=(None,))(output_image_value)



    loss = Lambda(lambda x: (weights[0]*x[0]+weights[1]*x[1]+weights[2]*x[2]+weights[3]*x[3]+weights[4]*x[4])
                            /(weights[0]+weights[1]+weights[2]+weights[3]+weights[4]))([loss_E, loss_D_real, loss_D_fake, loss_EGD, loss_image])
    # loss = Lambda(lambda x: (weights[4] * x[4]) / (weights[4]))([loss_E, loss_D_real, loss_D_fake, loss_EGD, loss_image])
    # loss = loss_image

    return Model(inputs=[output_E, output_E_center, output_D_real, output_D_fake, output_EGD, output_image, output_re_image], outputs=loss)

# re-write the loss_all_model
def loss_all_model(size_image, size_z, size_age_label, size_name_label, size_gender_label, num_input_channels, weights,
                   encoder, discriminator, EG, EGD):
    input_real_images = Input(shape=(size_image, size_image, num_input_channels))
    input_fake_images = Input(shape=(size_image, size_image, num_input_channels))

    input_ages_conv = Input(shape=(1, 1, size_age_label))
    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_genders_conv = Input(shape=(1, 1, size_gender_label))

    output_E = encoder(inputs=[input_real_images, input_ages_conv, input_names_conv, input_genders_conv])
    output_E_center = Input(shape=(size_z, ))
    output_E_value = Subtract()([output_E, output_E_center])

    discriminator.trainable = False
    output_D_real = discriminator(inputs=[input_real_images, input_ages_conv, input_names_conv, input_genders_conv])
    output_D_fake = discriminator(inputs=[input_fake_images, input_ages_conv, input_names_conv, input_genders_conv])
    output_EGD = EGD([input_real_images, input_ages_conv, input_names_conv, input_genders_conv])

    output_image = input_real_images
    output_re_image = EG([input_real_images, input_ages_conv, input_names_conv, input_genders_conv])

    output_image_value = Subtract()([output_image, output_re_image])

    loss_E = Lambda(get_loss_latant, output_shape=(None,))(output_E_value)
    loss_D_real = Lambda(get_loss_binary, arguments={'binary': K.ones_like(output_D_real)}, output_shape=(None,))(output_D_real)
    loss_D_fake = Lambda(get_loss_binary, arguments={'binary': K.zeros_like(output_D_real)}, output_shape=(None,))(output_D_fake)
    loss_EGD = Lambda(get_loss_binary, arguments={'binary': K.ones_like(output_EGD)}, output_shape=(None,))(output_EGD)
    loss_image = Lambda(get_loss_image, output_shape=(None,),
                        arguments={'size_image': size_image, 'num_input_channels': num_input_channels})(output_image_value)

    loss = Lambda(
        lambda x: (weights[0] * x[0] + weights[1] * x[1] + weights[2] * x[2] + weights[3] * x[3] + weights[4] * x[4])
                  / (weights[0] + weights[1] + weights[2] + weights[3] + weights[4]))([loss_E, loss_D_real, loss_D_fake, loss_EGD, loss_image])
    # loss = Lambda(lambda x: (weights[4] * x[4]) / (weights[4]))([loss_E, loss_D_real, loss_D_fake, loss_EGD, loss_image])
    # loss = loss_image

    return Model(
        inputs=[input_real_images, input_fake_images, input_ages_conv, input_names_conv, input_genders_conv, output_E_center],
        outputs=loss)




def get_loss_latant(t):
    return K.sqrt(K.reduce_sum(K.square(t)))

def get_loss_image(t, size_image, num_input_channels):
    return K.sqrt(K.reduce_sum(K.square(t)))/size_image/size_image/num_input_channels

def get_loss_D_real(x):
    K.reduce_mean(K.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=K.ones_like(x)))
    import tensorflow


def get_loss_D_fake(x):
    K.reduce_mean(K.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=K.zeros_like(x)))

def get_loss_binary(x, binary):
    from keras.losses import binary_crossentropy
    return binary_crossentropy(x, binary)

def loss_by_weights(list, weights):
    loss = 0
    for i in range(len(weights)):
        loss = loss + weights[i] * list[i]
    return loss