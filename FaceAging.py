import encoder, generator, gan, discriminator, loss
from train import train_E_model, generate_latant_z, generate_latent_center, train_loss_all
from ops import load_image, age_group_label, duplicate, \
    concat_label, save_image, save_weights, load_celebrity_image, save_loss, copy_array, name_gender_label, get_inner_inter_center_by_age_name, draw_loss_metric
from keras.optimizers import SGD, Adam
from keras.models import load_model
from glob import glob
import os
import numpy as np
import time
import keras.backend as K
from keras.utils import multi_gpu_model

class FaceAging(object):
    def __init__(self,
                 size_image,  # size the input images
                 size_batch,  # size of one batch
                 dataset_name,  # name of the dataset in the folder ./data
                 sample_dir,

                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 num_input_channels=3,  # number of channels of input images

                 size_z=100,  # number of channels of the layer z (noise or code)
                 # num_encoder_channels=[64, 128, 256, 512],  # number of channels of every conv layers of encoder
                 num_encoder_channels=[512, 256, 128, 64, 3],  # number of channels of every conv layers of encoder

                 size_gen=512, # number of channels of the generator's start layer
                 num_gen_channels=[512, 256, 128, 64, 3],  # number of channels of every deconv layers of generator

                 num_Dz_channels=[128, 64, 32, 16, 1],  # number of channels of every conv layers of discriminator_z

                 #num_Dimg_channels=3,  # number of channels of discriminator input image
                 num_Dimg_channels=[32, 64, 64*2, 64*4, 64*8],  #number of channels of  every conv layers of discriminator_img
                 num_Dimg_fc_channels = 1024, # number of channels of last fc layer of discriminator_img


                 size_age=10,  # number of categories (age segments) in the training dataset
                 size_name=133, # number of name array
                 size_name_total=133, # number of total name
                 size_gender=2, # male & female

                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and z
                 is_training=True,  # flag for training or testing mode
                 save_dir='./save',  # path to save checkpoints, samples, and summary

                 image_mode='RGB',
                 loss_weights=[0, 0, 0, 0],
                 num_GPU=2,
                 num_D_img_loss=1,
                 num_all_loss=1
                 ):
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_batch = size_batch
        self.size_kernel = size_kernel
        self.num_input_channels = num_input_channels
        self.size_z = size_z
        self.num_encoder_channels = num_encoder_channels
        self.num_Dz_channels = num_Dz_channels
        self.num_Dimg_channels = num_Dimg_channels
        self.num_Dimg_fc_channels = num_Dimg_fc_channels
        self.size_age = size_age
        self.size_name = size_name
        self.size_name_total = size_name_total
        self.size_gender = size_gender
        self.size_gen = size_gen
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.sample_dir = sample_dir
        self.image_mode = image_mode
        self.loss_weights = loss_weights
        self.num_GPU = num_GPU
        self.num_D_img_loss = num_D_img_loss
        self.num_all_loss = num_all_loss

        print("\n\tBuilding the graph...")

        # label of age + gender duplicate size
        self.size_age_label = duplicate(
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio,
            size_age=self.size_age)

        # encoder model: input_image --> z
        self.E_model = encoder.encoder_model(
            size_image=self.size_image,
            size_age_label=self.size_age_label,
            size_name_label=self.size_name,
            size_gender_label=self.size_gender,
            num_input_channels=self.num_input_channels,
            size_kernel=self.size_kernel,
            size_z=self.size_z,
            num_encoder_channels=self.num_encoder_channels
        )

        # generator model: z + label --> generated image
        num_layers = len(num_gen_channels)
        self.G_model = generator.generator_model(
            size_z=self.size_z,
            size_age_label=self.size_age_label,
            size_name_label=self.size_name,
            size_gender_label=self.size_gender,
            size_mini_map=int(self.size_image / 2 ** num_layers),
            size_kernel=self.size_kernel,
            size_gen=self.size_gen,
            num_input_channels=self.num_input_channels,
            num_gen_channels=self.num_gen_channels
        )

        # # discriminator model on z
        # self.D_z_model = discriminator.discriminator_z_model(
        #     size_z=self.size_z,
        #     num_Dz_channels=self.num_Dz_channels
        # )

        # discriminator model on G
        self.D_img_model = discriminator.discriminator_img_model(
            size_image=self.size_image,
            size_kernel=self.size_kernel,
            size_age_label=self.size_age_label,
            size_name_label=self.size_name,
            size_gender_label=self.size_gender,
            num_input_channels=self.num_input_channels,
            num_Dimg_channels=self.num_Dimg_channels,
            num_Dimg_fc_channels=self.num_Dimg_fc_channels
        )

        # E + G Model
        self.EG_model = gan.egModel(
            self.E_model, self.G_model,
            self.size_image, self.size_age_label,
            self.size_name, self.size_gender, self.num_input_channels)

        # G + D_img Model
        self.GD_model = gan.gdModel(
            self.G_model, self.D_img_model,
            self.size_z, self.size_age_label, self.size_name, self.size_gender,)

        # E + G + Dimg Model
        self.EGD_model = gan.egdModel(
            self.E_model, self.G_model, self.D_img_model,
            self.size_image, self.size_age_label, self.size_name, self.size_gender, self.num_input_channels)

        # ****************************** multi-GPU ***********************************
        # self.E_model = multi_gpu_model(self.E_model, gpus=self.num_GPU)
        # self.G_model = multi_gpu_model(self.G_model, gpus=self.num_GPU)
        # self.D_img_model = multi_gpu_model(self.D_img_model, gpus=self.num_GPU)
        # self.EG_model = multi_gpu_model(self.EG_model, gpus=self.num_GPU)
        # self.GD_model = multi_gpu_model(self.GD_model, gpus=self.num_GPU)
        # self.EGD_model = multi_gpu_model(self.EGD_model, gpus=self.num_GPU)

        # ************************** EGD weighted loss *************************************
        self.loss_Model = loss.loss_all_model(
            self.size_image, self.size_z, self.size_age, self.size_name, self.size_gender, self.num_input_channels,
            self.loss_weights, self.E_model, self.D_img_model, self.EG_model, self.EGD_model)
        adam_loss = Adam(lr=0.0002, beta_1=0.5, beta_2=0.99)
        self.loss_Model.compile(optimizer=adam_loss, loss=lambda y_true, y_pred: y_pred)
        # self.loss_Model.compile(optimizer=adam_loss, loss='mae')



        # ************************************* optimizer *******************************************
        adam_E = Adam(lr=0.0002, beta_1=0.5)
        # adam_G = Adam(lr=0.0002, beta_1=0.5)
        # adam_EG = Adam(lr=0.0002, beta_1=0.5)
        adam_GD = Adam(lr=0.0001, beta_1=0.5)
        adam_EGD = Adam(lr=0.0001, beta_1=0.5)

        adam_D_img = Adam(lr=0.0002, beta_1=0.5)
        # adam_D_z = Adam(lr=0.0002, beta_1=0.5)

        # ************************************* Compile loss  *******************************************************
        # loss model of encoder + generator
        self.E_model.compile(optimizer=adam_E, loss='mean_squared_error') # mean squared error
        # self.G_model.compile(optimizer=adam_G, loss='mean_squared_error')
        # self.EG_model.compile(optimizer=adam_EG, loss='mean_squared_error')

        # loss model of discriminator on generated image
        self.GD_model.compile(optimizer=adam_GD, loss='binary_crossentropy')
        self.EGD_model.compile(optimizer=adam_EGD, loss='binary_crossentropy')
        # loss model of discriminator on generated + real image
        self.D_img_model.trainable = True
        self.D_img_model.compile(optimizer=adam_D_img, loss='binary_crossentropy')





    def train(self,
              num_epochs,  # number of epochs
              size_batch,  # mini-batch size for training and testing, must be square of an integer
              path_data,  # upper dir of data
              # learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              use_trained_model=True,  # used the saved checkpoint to initialize the model
              ):

        #  *************************** load file names of images ***************************************
        file_names = glob(os.path.join(path_data, self.dataset_name, '*.jpg'))
        # ******************************************* training *******************************************************
        print('\n\tPreparing for training ...')

        # load model
        if use_trained_model:
            # path_weights = self.save_dir + '/load'
            if os.path.exists('EGD.h5') :
                self.EGD_model.load_weights('EGD.h5')
                print("\t**************** LOADING EGD MODEL SUCCESS! ****************")
            else:
                print("\t**************** LOADING EGD MODEL FAILED! ****************")


        # *************************** preparing data for epoch iteration *************************************

        # images = [load_image(
        #     image_path=file_name,
        #     image_size=self.size_image,
        #     image_value_range=self.image_value_range,
        #     is_gray=(self.num_input_channels == 1),
        # ) for file_name in file_names]
        # if self.num_input_channels == 1:
        #     real_images = np.array(images).astype(np.float32)[:, :, :, None]
        # else:
        #     real_images = np.array(images).astype(np.float32)

        file_names_reorder, inner_centers, inter_centers, age_labels, name_labels, gender_labels = \
            get_inner_inter_center_by_age_name(file_names, self.size_image, self.image_value_range, self.num_input_channels,
                                               self.size_age, self.size_name, self.size_gender, self.size_name_total, self.E_model)


        loss_E, loss_Dimg, loss_EGD1, loss_EGD2, loss_all = [], [], [], [], []
        for epoch in range(num_epochs):
            num_batches = len(file_names) // size_batch
            for index_batch in range(num_batches):
                # ********************** real batch images and labels **********************
                # real images
                batch_file_names = file_names_reorder[index_batch * size_batch:(index_batch + 1) * size_batch]
                images = [load_image(
                    image_path=file_name,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for file_name in batch_file_names]
                if self.num_input_channels == 1:
                    batch_real_images = np.array(images).astype(np.float32)[:, :, :, None]
                else:
                    batch_real_images = np.array(images).astype(np.float32)

                batch_inner_center = inner_centers[index_batch * size_batch:(index_batch + 1) * size_batch]
                batch_inter_center = inter_centers[index_batch * size_batch:(index_batch + 1) * size_batch]
                batch_real_label_age = age_labels[index_batch * size_batch:(index_batch + 1) * size_batch]
                batch_real_label_name = name_labels[index_batch * size_batch:(index_batch + 1) * size_batch]
                batch_real_label_gender = gender_labels[index_batch * size_batch:(index_batch + 1) * size_batch]

                batch_real_label_age_conv = np.reshape(batch_real_label_age, [len(batch_real_label_age), 1, 1, batch_real_label_age.shape[-1]])
                batch_real_label_name_conv = np.reshape(batch_real_label_name, [len(batch_real_label_name), 1, 1, batch_real_label_name.shape[-1]])
                batch_real_label_gender_conv = np.reshape(batch_real_label_gender, [len(batch_real_label_gender), 1, 1, batch_real_label_gender.shape[-1]])

                batch_fake_image = self.EG_model.predict([batch_real_images, batch_real_label_age_conv, batch_real_label_name_conv, batch_real_label_gender_conv])

                for D in range(int(self.num_D_img_loss)):
                    # ************************* start train D_img_model with D_img_loss ****************************
                    train_batch_x = np.concatenate((batch_real_images, batch_fake_image), axis=0)
                    train_batch_age_conv = np.concatenate((batch_real_label_age_conv, batch_real_label_age_conv), axis=0)
                    train_batch_name_conv = np.concatenate((batch_real_label_name_conv, batch_real_label_name_conv), axis=0)
                    train_batch_gender_conv = np.concatenate((batch_real_label_gender_conv, batch_real_label_gender_conv), axis=0)
                    # label
                    train_batch_y = np.concatenate((np.ones(size_batch), np.zeros(size_batch)), axis=0)

                    # train D_img
                    loss_Dimg.append(self.D_img_model.train_on_batch([train_batch_x, train_batch_age_conv, train_batch_name_conv, train_batch_gender_conv], train_batch_y))
                    print('loss_Dimg on b_', index_batch, ' e_', epoch, ' is ', loss_Dimg[-1])
                    # ************************* end train D_img_model with D_img_loss ****************************



                for all in range(int(self.num_all_loss)):
                    # ************************* start train EGD_model with all loss ****************************
                    # shorten inner distance and largen inter distance
                    input_real_images = np.concatenate((batch_real_images, batch_real_images), axis=0)
                    input_fake_images = np.concatenate((batch_fake_image, batch_fake_image), axis=0)
                    input_ages_conv = np.concatenate((batch_real_label_age_conv, batch_real_label_age_conv), axis=0)
                    input_names_conv = np.concatenate((batch_real_label_name_conv, batch_real_label_name_conv), axis=0)
                    input_genders_conv = np.concatenate((batch_real_label_gender_conv, batch_real_label_gender_conv), axis=0)
                    output_E_center = np.concatenate((batch_inner_center, batch_inter_center), axis=0)

                    self.D_img_model.trainable = False

                    loss_all.append(self.loss_Model.train_on_batch(
                        [input_real_images[0:size_batch], input_fake_images[0:size_batch],
                         input_ages_conv[0:size_batch], input_names_conv[0:size_batch], input_genders_conv[0:size_batch], output_E_center[0:size_batch]],
                        np.zeros(size_batch)))
                    loss_all.append(self.loss_Model.train_on_batch(
                        [input_real_images[size_batch:2*size_batch], input_fake_images[size_batch:2*size_batch],
                         input_ages_conv[size_batch:2*size_batch], input_names_conv[size_batch:2*size_batch], input_genders_conv[size_batch:2*size_batch],
                         output_E_center[size_batch:2*size_batch]],
                        np.zeros(size_batch)))
                    print('loss_all on b_', index_batch, 'e_', epoch, 'for shorten inner distance is', loss_all[-1])

                    self.D_img_model.trainable = True
                    # ************************* end train EGD_model with all loss ****************************


                # *********************************** save images *******************************************
                # *************************  start E_model to generate latant_z ***********************************
                if (epoch % 5 == 0) or (epoch == 1):
                    sample_names = glob(os.path.join(path_data, self.sample_dir, '*.jpg'))
                    sample_images = [load_image(
                        image_path=file_name,
                        image_size=self.size_image,
                        image_value_range=self.image_value_range,
                        is_gray=(self.num_input_channels == 1),
                    ) for file_name in sample_names]
                    if self.num_input_channels == 1:
                        sample_images = np.array(sample_images).astype(np.float32)[:, :, :, None]
                    else:
                        sample_images = np.array(sample_images).astype(np.float32)


                    num_sample = len(sample_images)
                    fake_age = np.zeros((num_sample, self.size_age))
                    fake_age[:, 4] = 1
                    fake_age_conv = np.reshape(fake_age, [len(fake_age), 1, 1, fake_age.shape[-1]])
                    fake_name = np.zeros((num_sample, self.size_name))
                    if int(self.size_name) > 1:
                        fake_name[:, 0] = 1
                    else:
                        fake_name[:] = 1 / self.size_name_total
                    fake_name_conv = np.reshape(fake_name, [len(fake_name), 1, 1, fake_name.shape[-1]])
                    fake_gender = np.zeros((num_sample, self.size_gender))
                    fake_gender[:, 1] = 1
                    fake_gender_conv = np.reshape(fake_gender, [len(fake_gender), 1, 1, fake_gender.shape[-1]])

                    sample_fake_image = self.EG_model.predict([sample_images, fake_age_conv, fake_name_conv, fake_gender_conv])
                    save_image(sample_fake_image, self.size_image, self.image_value_range, self.num_input_channels, epoch, index_batch, self.image_mode,
                               self.save_dir + '/image')
                # ************************  end of E to generate latant_z ************************

            if (epoch % 20 == 0) or (epoch == 1):
                save_weights(self.save_dir + '/weight', self.EGD_model, None,  epoch, 0)
                save_loss(self.save_dir+'/metric', loss_E, loss_Dimg, loss_all)
                # draw_loss_metric(self.save_dir+'/metric/', 'loss_E')
                draw_loss_metric(self.save_dir + '/metric/', 'loss_Dimg')
                draw_loss_metric(self.save_dir + '/metric/', 'loss_all')


    # def generate_fake_image(self, size_batch):
        # size_pix = self.size_image * self.size_image * self.num_input_channels
        # noise_image = np.zeros((size_batch, size_pix))
        # for b in range(size_batch):
        #     noise_image[b, :] = np.random.uniform(
        #         self.image_value_range[0], self.image_value_range[-1], size_pix)
        # noise_image = noise_image.reshape(
        #     [size_batch, self.size_image, self.size_image, self.num_input_channels])
        #
        # noise_label_age = np.zeros((size_batch, self.size_age))
        # random_age = np.random.randint(0, self.size_age, size_batch)
        # for i, age in enumerate(random_age):
        #     noise_label_age[i, age] = self.image_value_range[-1]
        # noise_label_age = concat_label(noise_label_age, self.enable_tile_label, self.tile_ratio)
        #
        # fake_img = self.EG_model.predict([noise_image, noise_label_age], verbose=0)
        # fake_label_age = noise_label_age

        # return [fake_img, fake_label_age]


    #def generate_compare_target(self):

    # **************************************** useless code ******************************************
    # def get_array_center(self, array):
    #     num = len(array)
    #     sum = np.zeros(array.shape[1])
    #     average = np.zeros(array.shape[1])
    #     for i in range(num):
    #         sum = sum + array[i]
    #     average = sum/num
    #
    #     return average
    #
    # def generate_img_for_gather(self, images_age_label_identity_list):
    #     file_path = os.path.join('./data/', self.dataset_name)
    #     center = self.generate_latant_center(images_age_label_identity_list)
    #     list_img = []
    #     list_center = []
    #     for i in range(len(images_age_label_identity_list)):
    #         img_center = center[i]
    #         for j in range(len(images_age_label_identity_list[i])):
    #             image_name = images_age_label_identity_list[i][j].name
    #             image_name = str(file_path) + '/' + image_name
    #             list_img.append(load_image(image_name).tolist())
    #             list_center.append(img_center.tolist())
    #     return [np.array(list_img), np.array(list_center)]

    # def generate_EGgenerated_center(self, images_age_label_identity_list):
    #     center = []
    #     img_source = []
    #     img_target = []
    #     age = []
    #     file_path = os.path.join('./data/', self.dataset_name)
    #     for i in range(len(images_age_label_identity_list)):
    #         list_part = images_age_label_identity_list[i]
    #         list_image = []
    #         age_label = np.zeros(shape=(len(list_part), self.size_age), dtype=np.float)
    #         for j in range(len(list_part)):
    #             ciInfo = list_part[j]
    #             image_name = str(file_path) + '/' + ciInfo.name
    #             image_age_label = int(ciInfo.age_label)
    #             list_image.append(load_image(image_name).tolist())
    #             age_label[j, image_age_label] = self.image_value_range[-1]
    #
    #         age_label = concat_label(age_label, self.enable_tile_label, self.tile_ratio)
    #         array_target = self.EG_model.predict([np.array(list_image), age_label], verbose=0)
    #         array_center = self.copy_array(np.average(array_target, axis=0), len(array_target))
    #
    #         for i in range(len(array_target)):
    #             img_target.append(array_target[i].tolist())
    #             center.append(array_center[i].tolist())
    #             img_source.append(list_image[i])
    #             age.append(age_label[i])
    #         # center.append(self.get_array_center(array_latant).tolist())
    #     return [np.array(img_source), np.array(age), np.array(img_target), np.array(center)]
