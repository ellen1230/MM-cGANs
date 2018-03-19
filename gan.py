from keras.layers import Input, Concatenate, Dense, Reshape
from keras.models import Model

def egdModel(EG, discriminator, size_image, size_age_label, num_input_channels):
    # label of age&gender + EG model ---> generated image
    input_images_EG = Input(shape=(size_image, size_image, num_input_channels))
    input_ages_conv = Input(shape=(1, 1, size_age_label))

    x = EG([input_images_EG, input_ages_conv])
    output_dcgan = discriminator([x, input_ages_conv])
    return Model(inputs=[input_images_EG, input_ages_conv], outputs=output_dcgan)

    # input_labels = Input(shape=(size_age_label,))
    # input_images_EG = Input(shape=(size_image, size_image, num_input_channels))
    # x = EG([input_images_EG, input_labels])
    #
    # # label of age&gender + discriminator_img ---> dcgan
    # #input_labels_Dimg = Input(shape=(size_label,))
    # # labels_Dimg = Dense(units=size_image * size_image, activation='tanh')(input_labels)
    # # labels_Dimg = Reshape(target_shape=(size_image, size_image, 1))(labels_Dimg)
    # #
    # # input_dimg = Concatenate(axis=-1)([x, input_labels])
    # discriminator.trainable = False
    # output_dcgan = discriminator([x, input_labels])
    # return Model(inputs=[input_images_EG, input_labels], outputs=output_dcgan)
    #
    #
    # #dcganInput = Input(shape=(size_image, size_image, num_input_channels+1))
    # # x = EG(dcganInput)
    # # Concatenate(axis=-1)([x, ])
    # # discriminator.trainable = False
    # # dcganOutput = discriminator(x)
    # # return Model(inputs=dcganInput, outputs=dcganOutput)

def egModel(encoder, generator, size_image, size_age_label, num_input_channels):

    input_images_EG = Input(shape=(size_image, size_image, num_input_channels))
    input_ages_conv = Input(shape=(1, 1, size_age_label))
    input_ages = Reshape(target_shape=(size_age_label, ))(input_ages_conv)

    x = encoder(inputs=[input_images_EG, input_ages_conv])
    egoutput = generator([x, input_ages])

    return Model(inputs=[input_images_EG, input_ages_conv], outputs=egoutput)

def gdModel(generator, D_img, size_z, size_age_label):

    input_z_GD = Input(shape=(size_z,))
    input_age_label = Input(shape=(size_age_label,))
    input_age_label_conv = Reshape(target_shape=(1, 1, size_age_label))(input_age_label)

    x = generator([input_z_GD, input_age_label])
    D_img.trainable = False
    gdoutput = D_img([x, input_age_label_conv])

    return Model(inputs=[input_z_GD, input_age_label], outputs=gdoutput)