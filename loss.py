from keras.layers import Input, Lambda, Concatenate
from keras.models import Model
from keras.layers.merge import Subtract, subtract
import tensorflow as K
import numpy as np
import keras.losses

# size: all output
def get_loss_all(size_z, weights):
    output_E = Input(shape=(size_z, ))
    output_E_center = Input(shape=(size_z, ))
    output_E_value = Subtract()([output_E, output_E_center])

    output_D_real = Input(shape=(1, ))
    output_D_fake = Input(shape=(1, ))

    output_EGD = Input(shape=(1, ))


    loss_E = Lambda(get_loss_E, output_shape=(None,))(output_E_value)
    loss_D_real = Lambda(get_loss_binary, arguments={'binary': K.ones_like(output_D_real)}, output_shape=(None,))(output_D_real)
    loss_D_fake = Lambda(get_loss_binary, arguments={'binary': K.zeros_like(output_D_real)}, output_shape=(None,))(output_D_fake)
    loss_EGD = Lambda(get_loss_binary, arguments={'binary': K.ones_like(output_EGD)}, output_shape=(None,))(output_EGD)



    # list = [loss_E, loss_D_fake, loss_D_real, loss_EGD]
    # losses = Concatenate(axis=-1)(list)
    # loss = Lambda(loss_by_weights, arguments={'weights': weights})(losses)
    loss = Lambda(lambda x: (weights[0] * x[0] +weights[1] * x[1] + weights[2] * x[2] + weights[3] * x[3])/(x[0]+x[1]+x[2]+x[3]))\
        ([loss_E, loss_D_real, loss_D_fake, loss_EGD])

    return Model(inputs=[output_E, output_E_center, output_D_real, output_D_fake, output_EGD], outputs=loss)



def get_loss_E(t):
    return K.reduce_mean(K.nn.l2_loss(t))

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