from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model

def generator_model():
    inputs = Input(shape=(100, ))
    # kernel_initializer = initializers.random_normal(stddev=0.02)

    dense1 = Dense(1024, activation="tanh")(inputs)
    output = Dense(1, activation="tanh")(dense1)

    return Model(inputs=inputs, outputs=output)


def my_loss(y_true,y_pred):
    return K.mean((y_pred-y_true),axis = -1)

#model = generator_model()
#model.compile(loss=my_loss, optimizer='SGD', metrics=['accuracy'])

import scipy.io as scio
from PIL import Image
from scipy.misc import imread, imsave

f = scio.loadmat('./data/mat/celebrityImageData.mat')
names = f['celebrityImageData']['name'][0,0]

for i in range(100):
    name = names[i, 0][0]
    image = imread('E:/data/CACD2000/'+ str(name))
    imsave('./data/CACD/'+str(name), image)

print('Done')
