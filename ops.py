from scipy.misc import imread, imresize, imsave
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model
from keras.layers import Concatenate
from PIL import Image
import scipy.io as scio
import DataClass
import random
import keras
from pylab import *
import matplotlib as mpl
mpl.use('Agg')

def load_image(
        image_path,  # path of a image
        image_size,  # expected size of the image
        image_value_range=(-1, 1),  # expected pixel value range of the image
        is_gray=False,  # gray scale or color image
        ):
    if is_gray:
        image = imread(image_path, flatten=True).astype(np.float32)
    else:
        image = imread(image_path).astype(np.float32)
        # image = np.array(Image.open(image_path).resize((image_size, image_size)))
    image = imresize(image, [image_size, image_size])
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    return image

def age_group_label(label):
    if 0 <= label <= 5:
        label = 0
    elif 6 <= label <= 10:
        label = 1
    elif 11 <= label <= 15:
        label = 2
    elif 16 <= label <= 20:
        label = 3
    elif 21 <= label <= 30:
        label = 4
    elif 31 <= label <= 40:
        label = 5
    elif 41 <= label <= 50:
        label = 6
    elif 51 <= label <= 60:
        label = 7
    elif 61 <= label <= 70:
        label = 8
    else:
        label = 9
    return label

# duplicate the label of age + gender to tile_ratio times
def duplicate(enable_tile_label, tile_ratio, size_age):
    # age duplicate (tile_ratio) times
    if enable_tile_label:
        size_label = int(size_age * tile_ratio)
    else:
        size_label = size_age
    # # gender duplicate (tile_ratio) times
    # if enable_tile_label:
    #     size_label = size_label + int(2 * tile_ratio)
    # else:
    #     size_label = size_label + 2

    return size_label


def concat_label(label, enable_tile_label, tile_ratio):
    if enable_tile_label:
        for i in range(int(tile_ratio)-1):
            label = np.concatenate((label, label), axis=-1)

    return label
    #
    # x_shape = x.get_shape().as_list()
    # if duplicate < 1:
    #     return x
    # # duplicate the label to enhance its effect, does it really affect the result?
    # label = tf.tile(label, [1, duplicate])
    # label_shape = label.get_shape().as_list()
    # if len(x_shape) == 2:
    #     return tf.concat([x, label], 1)
    # elif len(x_shape) == 4:
    #     label = tf.reshape(label, [x_shape[0], 1, 1, label_shape[-1]])
    #     return tf.concat([x, label*tf.ones([x_shape[0], x_shape[1], x_shape[2], label_shape[-1]])], 3)


def load_weights(save_dir, EGD):
    print("\n\tLoading pre-trained model ...")
    #e_model = load_model(str(save_dir) + '/e.h5')
    #G_model = load_model(str(save_dir) + '/G.h5')
    # D_z_model = load_model(str(save_dir) + '/Dz.h5')
    # D_img_model = load_model((save_dir) + '/Dimg.h5')
    # EG_model = load_model(str(save_dir) + '/EG.h5')
    EGD = EGD.load_weights(str(save_dir) + '/EGD.h5', by_name=True)
    return EGD
    #return [e_model, G_model, D_z_model, D_img_model, EG_model, dcgan_model]

def save_weights(save_dir, EGD_model, Dz_model, epoch, batch):
    print("\n\tsaving trained model_e", epoch, " ...")
    EGD_model.save(filepath=save_dir +"/EGD_" + str(epoch) + 'b' + str(batch) + ".h5", overwrite=True)
    # D_z_model.save(filepath=save_dir +"/D_z_" + str(epoch) + 'b' + str(batch) + ".h5", overwrite=True)
    return "SUCCESS!"

def save_image(images, size_image, image_value_range, num_input_channels, epoch, batch, mode, image_path):
    print("\n\tsaving generated images_e", epoch, " ...")
    num_images = len(images)
    if num_images < 16:
        num_picture = int(np.sqrt(num_images))
        picture = np.zeros([size_image * num_picture, size_image * num_picture, num_input_channels])

        for i in range(num_picture):
            for j in range(num_picture):
                index = i * num_picture + j
                picture[i * size_image:(i + 1) * size_image, j * size_image:(j + 1) * size_image] = images[index]
    else:
        num_picture = 4
        picture = np.zeros([size_image * num_picture, size_image * num_picture, num_input_channels])

        for i in range(num_picture):
            for j in range(num_picture):
                index = i * num_picture + j
                picture[i * size_image:(i + 1) * size_image, j * size_image:(j + 1) * size_image] = images[index]




    # image = images[0].reshape(28, 28)
    # for i in range(1, images.shape[0]):
    #     image = np.append(image, images[i].reshape(28, 28), axis=1)

    # picture = \
    #     (picture.astype(np.float32) - image_value_range[0])* 255.0 / (image_value_range[-1] - image_value_range[0])
        # picture= picture * 127.5 + 127.5
    # picture = Image.fromarray(picture, mode=mode)
    # picture.save(str(image_path) + "/e" + str(epoch) + 'b' + str(batch) + ".jpg")

    picture = ((picture + 1)/2 * 255.0).astype(np.uint8)
    path = str(image_path)  + "/e" + str(epoch) + 'b' + str(batch) + ".jpg"
    imsave(path, picture)



def save_loss(save_dir, loss_E, loss_Dimg, loss_all):

    np.save(save_dir+'/loss_E.npy', np.array(loss_E))
    np.save(save_dir+'/loss_Dimg.npy', np.array(loss_Dimg))
    np.save(save_dir+'/loss_all.npy', np.array(loss_all))

    # f_e = open(save_dir+'/loss_e', 'wb')
    # f_dimg = open(save_dir+'/loss_dimg', 'wb')
    # f_gd1 = open(save_dir+'loss_gd1', 'wb')
    # f_gd2 = open(save_dir + 'loss_gd2', 'wb')
    #
    # f_e.write(loss_E)
    # f_dimg.write(loss_Dimg)
    # f_gd1.write(loss_GD1)
    # f_gd2.write(loss_GD2)
    #
    # f_e.close(), f_dimg.close(), f_gd1.close(), f_gd2

def load_celebrity_image(path, file_name, picked_names):# num is size of random picked images
    if str(file_name).find('.mat'):
        if str(file_name).find('celebrityImageData') != -1:
            f = scio.loadmat(path + file_name)


            # f = h5py.File(path + file_name, 'r')
            # arrays = {}
            # for k, v in f.items():
            #     arrays[k] = np.array(v)
            celebrityImageData = f['celebrityImageData']
            #celebrityImageInfo_list = []

            # celebrityImageData to celebrityImageInfo listload_celebrity_image('E:/data/','celebrityImageData.mat',100)
            # age = celebrityImageData['age'].value[0]
            ages, identities = celebrityImageData['age'][0,0], celebrityImageData['identity'][0,0]
            years, features = celebrityImageData['year'][0,0], celebrityImageData['feature'][0,0]
            ranks, lfws = celebrityImageData['rank'][0,0], celebrityImageData['lfw'][0,0]
            births, names = celebrityImageData['birth'][0,0], celebrityImageData['name'][0,0]

            # if isRandom:
            #     # random pick n=num images to run
            #     r = range(1, len(ages))
            #     random_pick = random.sample(r, num)
            # else:
            #     random_pick = range(0, num)

            images_age_label_identity_list = []
            age_label_identity_list = []

            for i in range(len(picked_names)):
                name = picked_names[i]
                index = names[:, 0].tolist().index([name])

                celebrityImageInfo = DataClass.CelebrityImageInfo(
                    age=int(ages[index,0]), identity=int(identities[index,0]), year=int(years[index,0]),
                    rank=int(ranks[index,0]), lfw=int(lfws[index,0]), birth=int(births[index,0]), name=name)
                age_label_identity = str(celebrityImageInfo.age_label)+'_'+str(celebrityImageInfo.identity)
                try:
                    index = age_label_identity_list.index(age_label_identity)
                    images_age_label_identity_list[index].append(celebrityImageInfo)
                except ValueError:
                    age_label_identity_list.append(age_label_identity)
                    images_age_label_identity = []
                    images_age_label_identity.append(celebrityImageInfo)
                    images_age_label_identity_list.append(images_age_label_identity)
                #celebrityImageInfo_list.append(celebrityImageInfo)

            return images_age_label_identity_list

        elif str(file_name).find('celebrityData') != -1:
            # f = h5py.File(path + file_name)
            # celebrityData = f['celebrityData']

            f = scio.loadmat(path + file_name)
            celebrityData = f['celebrityData']
            celebrityInfo_list = []

            # celebrityData to celebrityInfo list
            names, identities = celebrityData['name'][0,0], celebrityData['identity'][0,0]
            births, ranks = celebrityData['birth'][0,0], celebrityData['rank'][0,0]
            lfws = celebrityData['lfw'][0,0]

            for i in range(len(names)):
                celebrityInfo = DataClass.CelebrityInfo(
                    name=names[i,0][0], identity=int(identities[i,0]), birth=int(births[i,0]),
                    rank=int(ranks[i,0]), lfw=int(lfws[i,0]))
                celebrityInfo_list.append(celebrityInfo)

            return celebrityInfo_list
    else:
        return

def copy_array(array, times):
    a = []
    for i in range(times):
        a.append(array.tolist())
    return np.array(a)


def duplicate_conv(x, times):
    list = []
    for i in range(times):
        list.append(x)
    x = keras.layers.Concatenate(axis=1)(list)
    list = []
    for i in range(times):
        list.append(x)
    output = keras.layers.Concatenate(axis=2)(list)
    return output

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def name_gender_label(file_name):
    names = ['Michael_Cera', 'Julianna_Margulies', 'Treat_Williams', 'Peter_Jackson', 'Roseanne_Barr', 'William_Katt', 'Sam_Raimi', 'Rosanna_Arquette', 'Tilda_Swinton', 'Tim_Daly', 'Dana_Delany', 'Jennifer_Tilly', 'Kevin_Pollak', 'Madonna', 'Tom_Sizemore', 'John_Goodman', 'Vicki_Lewis', 'Virginia_Madsen', 'Woody_Harrelson', 'Zeljko_Ivanek', 'Adam_Arkin', 'Alfre_Woodard', 'Jon_Bon_Jovi', 'Julia_Louis-Dreyfus', 'ulianna_Margulies', 'Julianne_Moore', 'Ray_Romano', 'Peter_Jackson', 'Matt_LeBlanc', 'Fran_Drescher', 'Hugh_Grant', 'Geena_Davis', 'Eric_Roberts', 'Kevin_Spacey', 'Jeff_Perry', 'Jennifer_Coolidge', 'Dylan_McDermott', 'Patricia_Clarkson', 'Jennifer_Tilly', 'Mykelti_Williamson', 'Michael_Cerveris', 'Melanie_Griffith', 'Kelsey_Grammer', 'Lauren_Graham', 'Mark_Wahlberg', 'Megyn_Price', 'Nathan_Fillion', 'Paget_Brewster', 'Paul_Rudd', 'Virginia_Madsen', 'Sean_Patrick_Flanery', 'Shannen_Doherty', 'Taraji_P._Henson', 'Tim_Guinee', 'Rachel_Weisz', 'Stephen_Baldwin', 'Mark_Ruffalo', 'Tina_Fey', 'Aaron_Eckhart', 'Adewale_Akinnuoye-Agbaje', 'Sharlto_Copley', 'Amanda_Peet', 'Bradley_Cooper', 'Carla_Gallo', 'Casper_Van_Dien', 'Christina_Hendricks', 'Maya_Rudolph', 'Martin_Freeman', 'Naomi_Watts', 'Denise_Richards', 'Edward_Norton', 'Elizabeth_Mitchell', 'Michael_Pitt', 'Emily_Procter', 'Lucy_Liu', 'Aaron_Johnson', 'Britt_Robertson', 'Caitlin_Stasey', 'Chris_Colfer', 'Diego_Boneta', 'Emma_Watson', 'Jennifer_Lawrence', 'Liam_Hemsworth', 'Rupert_Grint', 'Rose_McIver', 'Naya_Rivera', 'Daniel_Radcliffe', 'Amanda_Bynes', 'Brant_Daugherty', 'Dakota_Johnson', 'Hunter_Parrish', 'Emma_Stone', 'Kristen_Connolly', 'Keira_Knightley', 'Taylor_Kinney', 'Hayden_Christensen', 'Katy_Perry', 'January_Jones', 'Chace_Crawford', 'Lily_Rabe', 'Matt_Bomer', 'Masiela_Lusha', 'Matthew_Goode', 'Nick_Cannon', 'Britney_Spears', 'Aimee_Teegarden', 'Jesse_Tyler_Ferguson', 'Drea_de_Matteo', 'Corey_Haim', 'Shawn_Ashmore', 'Sarah_Drew', 'Emily_Deschanel', 'Hugh_Dancy', 'Lauren_German', 'Lee_Pace', 'Kate_Hudson', 'Luke_Wilson', 'Sarah_Michelle_Gellar', 'Patrick_Dempsey', 'Ryan_Kwanten', 'Ali_Larter', 'Desmond_Harrington', 'Milla_Jovovich', 'Samantha_Barks', 'Chelsea_Kane', 'Tom_Mison', 'Matthew_Goode', 'Clark_Duke', 'Bryce_Dallas_Howard', 'Rosario_Dawson', 'Eric_Dane', 'Deborah_Ann_Woll', 'Luke_Evans', 'Jason_Segel', 'Elizabeth_Mitchell']

    # woman: 0 man:1
    genders = ['1', '0', '1', '1', '0', '1', '1', '0', '0', '1', '0', '0', '1', '0', '1', '1', '0', '0', '1', '1', '1', '0', '1', '0', '0', '0', '1', '1', '1', '0', '1', '0', '1', '1', '1', '0', '1', '0', '0', '1', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '0', '1', '0', '1', '1', '0', '1', '1', '1', '0', '1', '0', '1', '0', '0', '1', '0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '1', '1', '0', '0', '1', '1', '0', '0', '1', '0', '1', '0', '1', '0', '0', '0', '1', '1', '0', '0', '1', '0', '1', '0', '1', '1', '0', '0', '1', '0', '1', '1', '0', '0', '1', '0', '1', '0', '1', '0', '1', '1', '0', '1', '0', '0', '0', '1', '1', '1', '0', '0', '1', '0', '1', '1', '0']

    index = names.index(file_name)
    return index, int(genders[index])

def get_inner_inter_center_by_age_name(file_names, size_image, image_value_range, num_input_channels, size_age, size_name, size_gender, size_name_total, E_model):

    file_names_by_name_age = []
    age_name_genders = []
    file_names_reorder = []
    inner_centers = []
    inter_centers = []
    age_labels = []
    name_labels = []
    gender_labels = []

    for i, label in enumerate(file_names):

        # temp = str(file_names[i]).split('/')[-1]
        temp = str(file_names[i]).split('\\')[-1]
        age = int(temp.split('_')[0])
        name = temp[temp.index('_') + 1: temp.index('00') - 1]
        age = age_group_label(age)
        [name, gender] = name_gender_label(name)

        age_name_gender = str(age) + '_' + str(name) + '_' + str(gender)
        try:
            index = age_name_genders.index(age_name_gender)
            file_names_by_name_age[index].append(file_names[i])
        except:
            age_name_genders.append(age_name_gender)
            file_names_by_name_age.append([file_names[i]])


    for i in range(len(age_name_genders)):
        # inner centers
        inputs_file_names = file_names_by_name_age[i]
        input_real_images = \
            [load_image(
                image_path=file_name,
                image_size=size_image,
                image_value_range=image_value_range,
                is_gray=(num_input_channels == 1),
        ) for file_name in inputs_file_names]
        if num_input_channels == 1:
            input_real_images = np.array(input_real_images).astype(np.float32)[:, :, :, None]
        else:
            input_real_images = np.array(input_real_images).astype(np.float32)

        num = len(input_real_images)

        age = int(age_name_genders[i].split('_')[0])
        name = int(age_name_genders[i].split('_')[1])
        gender = int(age_name_genders[i].split('_')[-1])

        age_label = np.zeros((num, size_age))
        age_label[:, age] = 1
        name_label = np.zeros((num, size_name))
        if int(size_name) > 1:
            name_label[:, name] = 1
        else:
            name_label[:] = name / size_name_total
        gender_label = np.zeros((num, size_gender))
        gender_label[:, gender] = 1

        # age_label = concat_label(age_label, enable_tile_label, tile_ratio)
        age_label_conv = np.reshape(age_label, [num, 1, 1, age_label.shape[-1]])
        name_label_conv = np.reshape(name_label, [num, 1, 1, name_label.shape[-1]])
        gender_label_conv = np.reshape(gender_label, [num, 1, 1, gender_label.shape[-1]])

        inner_target = E_model.predict([input_real_images, age_label_conv, name_label_conv, gender_label_conv], verbose=0)
        inner_center = copy_array(np.average(inner_target, axis=0), num)

        # inter centers
        current_age_name_genders = age_name_genders[0: i] + age_name_genders[i + 1: len(age_name_genders)]
        for j in range(len(current_age_name_genders)):
            current_age = int(current_age_name_genders[j].split('_')[0])
            current_name = int(current_age_name_genders[j].split('_')[1])
            current_gender = int(current_age_name_genders[j].split('_')[-1])

            if (current_age != age) and (current_name == name):
                index = age_name_genders.index(str(current_age) + '_' + str(current_name) + '_' + str(current_gender))

                current_num = len(file_names_by_name_age[index])
                current_age_label = np.zeros((current_num, size_age))
                current_age_label[:, current_age] = 1
                current_name_label = np.zeros((current_num, size_name))
                if int(size_name) > 1:
                    current_name_label[:, current_name] = 1
                else:
                    current_name_label[:] = current_name / size_name_total
                current_gender_label = np.zeros((current_num, size_gender))
                current_gender_label[:, current_gender] = 1

                # current_age_label = concat_label(current_age_label, enable_tile_label, tile_ratio)
                current_age_label_conv = np.reshape(current_age_label, [current_num, 1, 1, current_age_label.shape[-1]])
                current_name_label_conv = np.reshape(current_name_label, [current_num, 1, 1, current_name_label.shape[-1]])
                current_gender_label_conv = np.reshape(current_gender_label, [current_num, 1, 1, current_gender_label.shape[-1]])

                current_inputs_file_names = file_names_by_name_age[index]
                current_image = [load_image(
                        image_path=file_name,
                        image_size=size_image,
                        image_value_range=image_value_range,
                        is_gray=(num_input_channels == 1),
                ) for file_name in current_inputs_file_names]
                if num_input_channels == 1:
                    current_image = np.array(current_image).astype(np.float32)[:, :, :, None]
                else:
                    current_image = np.array(current_image).astype(np.float32)


                inter_target = E_model.predict([current_image, current_age_label_conv, current_name_label_conv, current_gender_label_conv], verbose=0)
                inter_center = copy_array(-np.average(inter_target, axis=0), num)

        # print('name:',name)
        for ii in range(num):
            file_names_reorder.append(file_names_by_name_age[i][ii])
            inner_centers.append(inner_center[ii].tolist())
            inter_centers.append(inter_center[ii].tolist())
            age_labels.append(age_label[ii].tolist())
            name_labels.append(name_label[ii].tolist())
            gender_labels.append(gender_label[ii].tolist())

    print('Resorted Done!')
    return np.array(file_names_reorder), np.array(inner_centers), np.array(inter_centers), \
           np.array(age_labels), np.array(name_labels), np.array(gender_labels)

def draw_loss_metric(save_dir, npy_name):



    if os.path.exists(save_dir+npy_name + ".png"):
        print('remove loss png')
        os.remove(save_dir+npy_name + ".png")

    # Create a new figure of size 8x6 points, using 80 dots per inch
    # figure(figsize=(8, 6), dpi=80)

    # Create a new subplot from a grid of 1x1
    # subplot(1, 1, 1)

    # X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
    # C,S = np.cos(X), np.sin(X)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    F = np.load(save_dir + npy_name + '.npy')
    X = np.linspace(1, len(F), len(F), endpoint=True)

    # Plot cosine using blue color with a continuous line of width 1 (pixels)
    ax.plot(X, F, color="blue", linewidth=1.0, linestyle="-")
    savefig(save_dir+npy_name + ".png")
    ax.clear()
    # show()




if __name__ == '__main__':
    draw_loss_metric('./save/metric/', 'loss_all')
