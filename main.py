from FaceAging import FaceAging
import tensorflow as tf
import shutil
import os

flags = tf.app.flags
flags.DEFINE_integer(flag_name='epoch', default_value=200, docstring='number of epochs')
flags.DEFINE_integer(flag_name='size_image', default_value=64, docstring='size of an image')
flags.DEFINE_string(flag_name='size_batch', default_value=128, docstring='size of one batch')
flags.DEFINE_string(flag_name='size_name', default_value=140, docstring='size of name array, 133 or 1')
flags.DEFINE_string(flag_name='size_name_total', default_value=140, docstring='size of total name')
flags.DEFINE_boolean(flag_name='is_train', default_value=True, docstring='training mode')
flags.DEFINE_string(flag_name='path_data', default_value='./data', docstring='upper dir of dataset')
flags.DEFINE_string(flag_name='dataset_name', default_value='CACD', docstring='dataset name')
flags.DEFINE_string(flag_name='savedir', default_value='save', docstring='dir for saving training results')
flags.DEFINE_string(flag_name='testdir', default_value='None', docstring='dir for testing images')
flags.DEFINE_string(flag_name='image_mode', default_value='RGB', docstring='input image mode')
FLAGS = flags.FLAGS




if __name__ == '__main__':

    # delete save dir
    import os
    if os.path.exists(FLAGS.savedir):
        shutil.rmtree(FLAGS.savedir)
    # create save dir and it's baby dirs
    os.makedirs(FLAGS.savedir + '/weight')
    os.makedirs(FLAGS.savedir + '/image')
    os.makedirs(FLAGS.savedir + '/metric')


    fa_model = FaceAging(
        is_training=True,  # flag for training or testing mode
        save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
        dataset_name=FLAGS.dataset_name,  # name of the dataset in the folder ./data
        size_image=FLAGS.size_image,
        size_name=FLAGS.size_name,
        size_name_total=FLAGS.size_name_total
    )
    if FLAGS.is_train:
        print('\n\tTraining Mode')
        fa_model.train(
            num_epochs=FLAGS.epoch,  # number of epochs
            path_data = FLAGS.path_data, # upper dir of dataset
            size_batch= FLAGS.size_batch # size of one batch

        )
    else:
        print('\n\tTesting Mode')
        fa_model.custom_test(
            testing_samples_dir=FLAGS.testdir + '/*jpg'
        )
