import encoder, generator, gan, discriminator
from train import train_E_model, generate_latant_z, generate_latent_center
from ops import load_image, age_group_label, duplicate, load_weights, \
    concat_label, save_image, save_weights, load_celebrity_image, save_loss, copy_array
from keras.optimizers import SGD, Adam
from keras.models import load_model
from glob import glob
import os
import numpy as np
import time
import tensorflow as tf
import DataClass

class FaceAging(object):
    def __init__(self,
                 size_image=128,  # size the input images
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 num_input_channels=3,  # number of channels of input images

                 size_z=100,  # number of channels of the layer z (noise or code)
                 num_encoder_channels=[64, 128, 256, 512],  # number of channels of every conv layers of encoder

                 size_gen=1024, # number of channels of the generator's start layer
                 num_gen_channels=[512, 256, 128, 64],  # number of channels of every deconv layers of generator

                 num_Dz_channels=[64, 32, 16, 1],  # number of channels of every conv layers of discriminator_z

                 #num_Dimg_channels=3,  # number of channels of discriminator input image
                 num_Dimg_channels=[64, 64*2, 64*4, 64*8],  #number of channels of  every conv layers of discriminator_img
                 num_Dimg_fc_channels = 1024, # number of channels of last fc layer of discriminator_img


                 size_age=10,  # number of categories (age segments) in the training dataset

                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=2.0,  # ratio of the length between tiled label and z
                 is_training=True,  # flag for training or testing mode
                 save_dir='./save',  # path to save checkpoints, samples, and summary
                 dataset_name='CACD',  # name of the dataset in the folder ./data
                 image_mode='RGB'
                 ):
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.num_input_channels = num_input_channels
        self.size_z = size_z
        self.num_encoder_channels = num_encoder_channels
        self.num_Dz_channels = num_Dz_channels
        self.num_Dimg_channels = num_Dimg_channels
        self.num_Dimg_fc_channels = num_Dimg_fc_channels
        self.size_age = size_age
        self.size_gen = size_gen
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.image_mode = image_mode

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
            num_input_channels=self.num_input_channels,
            num_Dimg_channels=self.num_Dimg_channels,
            num_Dimg_fc_channels=self.num_Dimg_fc_channels
        )

        # # E + G Model
        # self.EG_model = gan.egModel(
        #     self.E_model, self.G_model,
        #     self.size_image, self.size_age_label, self.num_input_channels)
        #
        # G + D_img Model
        self.GD_model = gan.gdModel(
            self.G_model, self.D_img_model,
            self.size_z, self.size_age_label)

        # E + G + Dimg Model
        # self.EGD_model = gan.egdModel(
        #     self.EG_model, self.D_img_model,
        #     self.size_image, self.size_age_label, self.num_input_channels)

        # ************************************* optimizer *******************************************
        adam_e = Adam(lr=0.0002, beta_1=0.5)
        adam_G = Adam(lr=0.0002, beta_1=0.5)
        adam_EG = Adam(lr=0.0002, beta_1=0.5)
        adam_GD = Adam(lr=0.0002, beta_1=0.5)
        adam_EGD = Adam(lr=0.0002, beta_1=0.5)

        adam_D_img = Adam(lr=0.0002, beta_1=0.5)
        # adam_D_z = Adam(lr=0.0002, beta_1=0.5)

        # ************************************* Compile loss  *******************************************************
        # loss model of encoder + generator
        self.E_model.compile(optimizer=adam_e, loss='mean_squared_error') # mean squared error
        self.G_model.compile(optimizer=adam_G, loss='mean_squared_error')
        # self.EG_model.compile(optimizer=adam_EG, loss='mean_squared_error')

        # loss model of discriminator on generated image
        self.GD_model.compile(optimizer=adam_GD, loss='binary_crossentropy')
        # self.EGD_model.compile(optimizer=adam_EGD, loss='binary_crossentropy')
        # loss model of discriminator on generated + real image
        self.D_img_model.trainable = True
        self.D_img_model.compile(optimizer=adam_D_img, loss='binary_crossentropy')

        # loss model of discriminator on z
        # self.D_z_model.compile(optimizer=adam_D_z, loss='mean_squared_error')

    def train(self,
              num_epochs=200,  # number of epochs
              size_batch=10,  # mini-batch size for training and testing, must be square of an integer
              learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # used the saved checkpoint to initialize the model
              path_data='./data', # upper dir of data
              isRandom=False, # random to pick the CACD dataset's image
              ):

        #  *************************** load file names of images ***************************************
        file_names = glob(os.path.join(path_data, self.dataset_name, '*.jpg'))
        size_data = len(file_names)

        np.random.seed(seed=2017)
        if enable_shuffle:
            np.random.shuffle(file_names)

            # ************* get some random samples as testing data to visualize the learning process *********************
            # sample_files = file_names[0:size_batch]
            # file_names[0:size_batch] = []
            # # sample image
            # sample = [load_image(
            #     image_path=sample_file,
            #     image_size=self.size_image,
            #     image_value_range=self.image_value_range,
            #     is_gray=(self.num_input_channels == 1),
            # ) for sample_file in sample_files]
            # if self.num_input_channels == 1:
            #     sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            # else:
            #     sample_images = np.array(sample).astype(np.float32)
            #
            # # age label
            # sample_label_age = np.zeros(
            #     shape=(len(sample_files), self.size_age), dtype=np.float32
            # ) #* self.image_value_range[0]
            # # gender label
            # # sample_label_gender = np.zeros(
            # #     shape=(len(sample_files), 2), dtype=np.float32
            # # ) #* self.image_value_range[0]
            #
            # for i, label in enumerate(sample_files):
            #     if self.dataset_name == 'UTKFace':
            #         age = int(str(sample_files[i]).split('/')[-1].split('_')[0].split('\\')[-1])
            #     elif self.dataset_name == 'CACD':
            #         age = int(str(sample_files[i]).split('/')[-1].split('_')[0])
            #     age = age_group_label(age)
            #     sample_label_age[i, age] = self.image_value_range[-1]
            # sample_label_age = concat_label(sample_label_age, self.enable_tile_label, self.tile_ratio)
            #
            #     # gender = int(str(sample_files[i]).split('/')[-1].split('_')[1])
            #     # sample_label_gender[i, gender] = self.image_value_range[-1]

            # ******************************************* training *******************************************************
            print('\n\tPreparing for training ...')

            # load model
            # if use_trained_model:
            #     path_weights = './weight'
                # if os.path.exists(path_weights + '/Dz.h5') and \
                #     os.path.exists(path_weights + '/Dimg.h5') and \
                #     os.path.exists(path_weights + '/EG.h5') and \
                #     os.path.exists(path_weights + '/dcgan.h5'):
                #     [self.D_z_model, self.D_img_model, self.EG_model, self.dcgan_model] = load_weights(path_weights)
                #     print("\tSUCCESS!")


            # # images list for shorten the distance of same age && name
            # images_age_label_identity_list = \
            #     load_celebrity_image(os.path.join(path_data, 'mat/'), 'celebrityImageData.mat', num_pick_image, False)

            # *************************** preparing data for epoch iteration *************************************
            num_batches = len(file_names) // size_batch
            loss_E, loss_Dimg, loss_GD1, loss_GD2 = [], [], [], []
            for epoch in range(num_epochs):
                if enable_shuffle:
                    np.random.shuffle(file_names)
                for index_batch in range(num_batches):
                    start_time = time.time()
                    # ********************** real batch images and labels **********************
                    # real images
                    batch_files = file_names[index_batch * size_batch:(index_batch + 1) * size_batch]
                    batch = [load_image(
                        image_path=batch_file,
                        image_size=self.size_image,
                        image_value_range=self.image_value_range,
                        is_gray=(self.num_input_channels == 1),
                    ) for batch_file in batch_files]
                    if self.num_input_channels == 1:
                        batch_real_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_real_images = np.array(batch).astype(np.float32)

                    # age label
                    batch_real_label_age = np.zeros(
                        shape=(len(batch_files), self.size_age), dtype=np.float
                    ) #* self.image_value_range[0]
                    # # gender label
                    # batch_label_gender = np.zeros(
                    #     shape=(len(batch_files), 2), dtype=np.float
                    # ) #* self.image_value_range[0]
                    batch_files_name = []
                    for i, label in enumerate(batch_files):
                        if self.dataset_name == 'UTKFace':
                            age = int(str(batch_files[i]).split('/')[-1].split('_')[0].split('/')[-1])
                        elif self.dataset_name == 'CACD':
                            age = int(str(batch_files[i]).split('\\')[-1].split('_')[0])
                        age = age_group_label(age)
                        batch_real_label_age[i, age] = self.image_value_range[-1]
                        batch_files_name.append(str(batch_files[i]).split('\\')[-1])
                    batch_real_label_age = concat_label(batch_real_label_age, self.enable_tile_label, self.tile_ratio)
                    batch_real_label_age_conv = np.reshape(batch_real_label_age,
                                                           [len(batch_real_label_age), 1, 1, batch_real_label_age.shape[-1]])
                        # gender = int(str(batch_files[i]).split('/')[-1].split('_')[1])
                        # batch_label_gender[i, gender] = self.image_value_range[-1]

                    # ********************** training E_model to generate latant z *********************
                    start_time = time.time()
                    # ********************* 1 no shorten inner distance & largen inter distance *************************
                    # self.E_model, batch_z = generate_latant_z(self.E_model, batch_real_images, batch_real_label_age)

                    # ********************* 2 shorten inner distance & no largen inter distance *************************
                    self.E_model, batch_z, batch_latent_center = generate_latent_center(self.E_model, batch_real_images, batch_files_name,
                                           self.size_age, self.dataset_name, self.enable_tile_label, self.tile_ratio)
                    loss_E_batch = self.E_model.train_on_batch([batch_real_images, batch_real_label_age_conv], batch_latent_center)

                    loss_E.append(loss_E_batch)
                    end_time = time.time()
                    print('E_model time: ', (end_time-start_time)/60)


                    # ********************** fake batch images and labels **********************

                    # noise = np.random.uniform(self.image_value_range[0], self.image_value_range[1], size=(size_batch, self.size_z))
                    batch_fake_image = self.G_model.predict([batch_z, batch_real_label_age], verbose=0)

                    start_time = time.time()
                    # ********************** training GD model ***************************
                    # ********************** training the model discriminator_img ***************************
                    start_time = time.time()
                    # using real image + label and generated fake image + label to train the discriminator_img
                    # image
                    train_batch_x = np.concatenate((batch_real_images, batch_fake_image), axis=0)
                    # feature age
                    train_batch_age = np.concatenate((batch_real_label_age, batch_real_label_age), axis=0)
                    train_batch_age_conv = np.reshape(train_batch_age, [len(train_batch_age), 1, 1, train_batch_age.shape[-1]])
                    # label
                    train_batch_y = np.concatenate((np.ones(size_batch), np.zeros(size_batch)), axis=0)

                    # train D_img
                    loss_batch_Dimg = self.D_img_model.train_on_batch([train_batch_x, train_batch_age_conv], train_batch_y)
                    loss_Dimg.append(loss_batch_Dimg)
                    #print('loss_Dimg on b_' + str(index_batch) + ' e_' + str(epoch) + ' is ' + loss_batch_Dimg )
                    print('loss_Dimg on b_', index_batch, ' e_', epoch, ' is ' , loss_batch_Dimg )
                    # ************************** training the model generator + discriminator_img ***********************************
                    # train Dimg once and GD twice
                    self.D_img_model.trainable = False
                    loss_GD1.append(self.GD_model.train_on_batch([batch_z, batch_real_label_age], np.ones(size_batch)))
                    loss_GD2.append(self.GD_model.train_on_batch([batch_z, batch_real_label_age], np.ones(size_batch)))
                    print('loss_GD1 on b_', index_batch, ' e_', epoch, ' is ', loss_GD1[-1])
                    print('loss_GD2 on b_', index_batch, ' e_', epoch, ' is ', loss_GD2[-1])
                    self.D_img_model.trainable = True

                    end_time = time.time()
                    print('GD_model time: ', str((end_time - start_time) / 60))

                    # ************************ save images && model *******************************************
                    if (epoch % 20 == 0) or (epoch == 1):
                        batch_real_label_age_conv = np.reshape(batch_real_label_age,
                                                               [len(batch_real_label_age), 1, 1, batch_real_label_age.shape[-1]])
                        latant_z = self.E_model.predict([batch_real_images, batch_real_label_age_conv])
                        batch_fake_image2 = self.G_model.predict([latant_z, batch_real_label_age])
                        save_image(batch_fake_image2, self.size_image,
                                   self.image_value_range, self.num_input_channels, epoch, index_batch, self.image_mode,
                                   self.save_dir+'/image/')
                        # save_weights(self.save_dir+'/weight/', self.E_model,
                        #              self.G_model,self.D_z_model, self.D_img_model, epoch, index_batch)
                        save_weights(self.save_dir + '/weight/', self.E_model,
                                     self.G_model, None, self.D_img_model, epoch, index_batch)
                        save_loss(self.save_dir+'/metric/', loss_E, loss_Dimg, loss_GD1, loss_GD2)


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
