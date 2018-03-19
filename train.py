from ops import load_image, age_group_label, duplicate, load_weights, \
    concat_label, save_image, save_weights, load_celebrity_image, save_loss, copy_array
import os
import numpy as np
import time
from DataClass import Sort_by_AgeLabel_Name

def train_E_model(e_model, size_age, file_names,
                  dataset_name, enable_tile_label, tile_ratio, image_value_range):
    # ************************ training the model to preserve identity *********************************
    # The original design is to shorten the distance of n images of same person in same age(n = 3),
    # gathering the shared feature of a person's at same age to preserve the identity information.
    # Now for convenient operation, we calculate the center of target and minimization the
    # distance(mean_squared_error) to achieve the goal above.

    # pick_images = []:(picked face images of CACD or UTKFace)
    # isRandom: shuffle the images name or not

    # first edition: e_model center
    # 1 (random) images map to latant space
    # 2 latant array to calculate a center
    # 3 copy the center array len(images) time as target_latant
    # 4 loss_e is self.e_model.train on mean_squared_error

    images_age_label_identity_list = \
            load_celebrity_image('./data/mat/', 'celebrityImageData.mat', file_names)
    [batch_img, batch_age_conv, batch_latant, batch_latant_center] = \
        generate_latent_center(images_age_label_identity_list, e_model,
                             size_age, dataset_name, enable_tile_label, tile_ratio, image_value_range)

    loss_batch_e = e_model.train_on_batch([batch_img, batch_age_conv], batch_latant_center)



    # ****************************** reconsider the complexity of algorithm ********************************
    # give up on the load mat + create the instances of Class CelebrityImageInfo
    # just using the file_names to operate
    # images_age_label_identity_list = \
    #     load_celebrity_image('./data/mat/', 'celebrityImageData.mat', file_names)
    # [batch_img, batch_age_conv, batch_latant, batch_latant_center] = \
    #     generate_latent_center(images_age_label_identity_list, e_model,
    #                          size_age, dataset_name, enable_tile_label, tile_ratio, image_value_range)
    #
    # loss_batch_e = e_model.train_on_batch([batch_img, batch_age_conv], batch_latant_center)
    # ****************************** reconsider the complexity of algorithm ********************************



    # second edition: (G_model center)
    # 1 (random) images to latant space
    # 2 latant array to calculate a center
    # 3 use G_model to generate target_img (source = latant)
    # 4 loss_G is self.G_model.train on mean_squared_error

    # images_age_label_identity_list = \
    #     load_celebrity_image('./data/mat/', 'celebrityImageData.mat', num_pick_image, isRandom)
    # [img, age, latant, latant_center] = \
    #     self.generate_center_target(images_age_label_identity_list, self.e_model)
    # target_img = self.G_model.predict(latant)
    # target_img = self.copy_array(np.average(target_img, axis=0), len(target_img))
    #
    # num_batches = len(img) // size_batch
    # loss_G = []
    # for index_batch in range(num_batches):
    #     start_time = time.time()
    #     batch_latant = latant[index_batch * size_batch:(index_batch + 1) * size_batch]
    #     batch_img_target = target_img[index_batch * size_batch:(index_batch + 1) * size_batch]
    #     loss_batch_G = self.G_model.train_on_batch(batch_latant, batch_img_target)
    #     loss_G.append(loss_batch_G)
    #     end_time = time.time()

    # third edition: (EG_model center)
    # 1 (random) images as source images
    # 2 source images to calculate a center
    # 3 use EG_model to generate target_img (source = latant)
    # 4 loss_G is self.G_model.train on mean_squared_error

    # images_age_label_identity_list = \
    #     load_celebrity_image('./data/mat/', 'celebrityImageData.mat', num_pick_image, isRandom)
    # [img_source, age, img_target, center] = \
    #     self.generate_center_target(images_age_label_identity_list, self.EG_model)
    # num_batches = len(img_source) // size_batch
    # loss_EG = []
    # for index_batch in range(num_batches):
    #     start_time = time.time()
    #     batch_img_source = img_source[index_batch * size_batch:(index_batch + 1) * size_batch]
    #     batch_age = age[index_batch * size_batch:(index_batch + 1) * size_batch]
    #     batch_center = center[index_batch * size_batch:(index_batch + 1) * size_batch]
    #     loss_batch_EG = self.EG_model.train_on_batch([batch_img_source, batch_age], batch_center)
    #     loss_EG.append(loss_batch_EG)
    #     end_time = time.time()

    # ********************** training the model to largen the difference among age group ************************

    return e_model, batch_latant, loss_batch_e


# def generate_latent_center(images_age_label_identity_list, E_model,
#                            size_age, dataset_name, enable_tile_label, tile_ratio, image_value_range):
#     center=[]
#     target=[]
#     img = []
#     age = []
#     file_path = os.path.join('./data/', dataset_name)
#     for i in range(len(images_age_label_identity_list)):
#         list_part = images_age_label_identity_list[i]
#         list_image = []
#         age_label = np.zeros(shape=(len(list_part), size_age), dtype=np.float)
#         for j in range(len(list_part)):
#             ciInfo = list_part[j]
#             image_name = str(file_path) + '/' + ciInfo.name
#             image_age_label = int(ciInfo.age_label)
#             list_image.append(load_image(image_name).tolist())
#             age_label[j, image_age_label] = image_value_range[-1]
#
#         age_label = concat_label(age_label, enable_tile_label, tile_ratio)
#         age_label_conv = np.reshape(age_label, [len(list_part), 1, 1, age_label.shape[-1]])
#         array_target = E_model.predict([np.array(list_image), age_label_conv], verbose=0)
#         array_center = copy_array(np.average(array_target, axis=0), len(array_target))
#
#         for i in range(len(array_target)):
#             target.append(array_target[i].tolist())
#             center.append(array_center[i].tolist())
#             img.append(list_image[i])
#             age.append(age_label_conv[i])
#         #center.append(self.get_array_center(array_latant).tolist())
#     return [np.array(img), np.array(age), np.array(target), np.array(center)]

def generate_latant_z(E_model, real_images, real_label_age):
    real_label_age_conv = np.reshape(real_label_age, [len(real_label_age), 1, 1, real_label_age.shape[-1]])
    return E_model, E_model.predict([np.array(real_images), real_label_age_conv], verbose=0)


def generate_latent_center(E_model, real_images, file_names, size_age,
                           dataset_name, enable_tile_label, tile_ratio):
    center = []
    target = []
    images = []
    age_names = []
    for i in range(len(file_names)):
        file_name = file_names[i]
        if dataset_name == 'UTKFace':
            age = int(str(file_names[i]).split('/')[-1].split('_')[0].split('/')[-1])
        elif dataset_name == 'CACD':
            age = int(str(file_names[i]).split('\\')[-1].split('_')[0])
        age = age_group_label(age)
        name = file_name[file_name.index('_')+1: file_name.index('_00')]
        age_name = str(age) + '_' + name

        try:
            index = age_names.index(age_name)
            images[index].append(real_images[i])
        except:
            age_names.append(age_name)
            images.append([real_images[i]])

    # shorten the inner distance
    for i in range(len(age_names)):
        num = len(images[i])
        # age_label_conv
        age = int(age_names[i].split('_')[0])
        age_label = np.zeros((num, size_age))
        age_label[:, age] = 1
        age_label = concat_label(age_label, enable_tile_label, tile_ratio)
        age_label_conv = np.reshape(age_label, [num, 1, 1, age_label.shape[-1]])

        # inner z center
        t = E_model.predict([np.array(images[i]), age_label_conv], verbose=0)
        c = copy_array(np.average(t, axis=0), num)

        # inner center + target
        for index in range(len(c)):
            center.append(c[index].tolist())
            target.append(t[index].tolist())

    # largen the inter distance
    for i in range(len(age_names)):
        age = age_names[i].split('_')[0]
        name = age_names[i].split('_')[-1]
        current_age_names = age_names[0:i] + age_names[i+1:len(age_names)]
        for j in range(current_age_names):
            current_age = current_age_names[j].split('_')[0]
            current_name = current_age_names[j].split('_')[-1]
            if (current_age != age) and (current_name == name):
                index = age_names.index(str(current_age+'_'+current_name))

                num = len(images[index])
                # age_label_conv
                age = int(age_names[index].split('_')[0])
                age_label = np.zeros((num, size_age))
                age_label[:, age] = 1
                age_label = concat_label(age_label, enable_tile_label, tile_ratio)
                age_label_conv = np.reshape(age_label, [num, 1, 1, age_label.shape[-1]])

                # inter z center
                t = E_model.predict([np.array(images[index]), age_label_conv], verbose=0)
                c = copy_array(-np.average(t, axis=0), num)

                # inner center + target
                for index in range(len(c)):
                    center.append(c[index].tolist())
                    target.append(t[index].tolist())

    return E_model, np.array(target), np.array(center)


