from keras.layers import Input, Concatenate, Dense, Reshape
from keras.models import Model

def egdModel(E, G, discriminator, size_image, size_age_label, size_name_label, size_gender_label, num_input_channels):
    # label of age&gender + E model + G model ---> generated image
    input_images = Input(shape=(size_image, size_image, num_input_channels))
    input_ages_conv = Input(shape=(1, 1, size_age_label))
    input_ages = Reshape(target_shape=(size_age_label, ))(input_ages_conv)

    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_names = Reshape(target_shape=(size_name_label,))(input_names_conv)
    input_genders_conv = Input(shape=(1, 1, size_gender_label))
    input_genders = Reshape(target_shape=(size_gender_label,))(input_genders_conv)

    z = E([input_images, input_ages_conv, input_names_conv, input_genders_conv])
    generated_image = G([z, input_ages, input_names, input_genders])
    discriminator.trainable = False
    output = discriminator([generated_image, input_ages_conv, input_names_conv, input_genders_conv])
    return Model(inputs=[input_images, input_ages_conv, input_names_conv, input_genders_conv], outputs=output)

def egModel(encoder, generator, size_image, size_age_label, size_name_label, size_gender_label, num_input_channels):

    input_images_EG = Input(shape=(size_image, size_image, num_input_channels))
    input_ages_conv = Input(shape=(1, 1, size_age_label))
    input_ages = Reshape(target_shape=(size_age_label, ))(input_ages_conv)

    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_names = Reshape(target_shape=(size_name_label,))(input_names_conv)
    input_genders_conv = Input(shape=(1, 1, size_gender_label))
    input_genders = Reshape(target_shape=(size_gender_label,))(input_genders_conv)

    x = encoder(inputs=[input_images_EG, input_ages_conv, input_names_conv, input_genders_conv])
    egoutput = generator([x, input_ages, input_names, input_genders])

    return Model(inputs=[input_images_EG, input_ages_conv, input_names_conv, input_genders_conv], outputs=egoutput)

def gdModel(generator, D_img, size_z, size_age_label, size_name_label, size_gender_label):

    input_z_GD = Input(shape=(size_z,))
    input_age_label = Input(shape=(size_age_label,))
    input_age_label_conv = Reshape(target_shape=(1, 1, size_age_label))(input_age_label)

    input_name_label = Input(shape=(size_name_label,))
    input_name_label_conv = Reshape(target_shape=(1, 1, size_name_label))(input_name_label)
    input_gender_label = Input(shape=(size_gender_label,))
    input_gender_label_conv = Reshape(target_shape=(1, 1, size_gender_label))(input_gender_label)

    x = generator([input_z_GD, input_age_label, input_name_label, input_gender_label])
    D_img.trainable = False
    gdoutput = D_img([x, input_age_label_conv, input_name_label_conv, input_gender_label_conv])

    return Model(inputs=[input_z_GD, input_age_label, input_name_label, input_gender_label], outputs=gdoutput)