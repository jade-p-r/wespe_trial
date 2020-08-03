from __future__ import print_function
import scipy.misc
import os
import numpy as np
import imageio
import sys
from PIL import Image
def load_test_data(phone, dped_dir, IMAGE_SIZE):

    test_directory_phone = dped_dir + '/test/cropped_reduced/'
    test_directory_dslr = dped_dir + '/test/valid_reduced/'

    NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_phone)
                           if os.path.isfile(os.path.join(test_directory_phone, name))])

    test_data = np.zeros((NUM_TEST_IMAGES, IMAGE_SIZE))
    test_answ = np.zeros((NUM_TEST_IMAGES, IMAGE_SIZE))

    test_input_list = os.listdir(test_directory_phone)
    test_output_list = os.listdir(test_directory_dslr)

    for i in range(0, NUM_TEST_IMAGES):
        
        I = np.asarray(Image.open(os.path.join(test_directory_phone, test_input_list[i])))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE]))/255
        test_data[i, :] = I
        
        I = np.asarray(Image.open(os.path.join(test_directory_dslr, test_output_list[i])))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE]))/255
        test_answ[i, :] = I

        if i % 100 == 0:
            print(str(round(i * 100 / NUM_TEST_IMAGES)) + "% done", end="\r")

    return test_data, test_answ


def load_batch(phone, dped_dir, TRAIN_SIZE, IMAGE_SIZE):

    train_directory_phone = dped_dir + '/train/cropped_reduced/'
    train_directory_dslr = dped_dir + '/train/valid_reduced/'

    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone)
                               if os.path.isfile(os.path.join(train_directory_phone, name))])

    # if TRAIN_SIZE == -1 then load all images

    train_input_list = os.listdir(train_directory_phone)
    train_output_list = os.listdir(train_directory_dslr)

    if TRAIN_SIZE == -1:
        TRAIN_SIZE = NUM_TRAINING_IMAGES
        TRAIN_IMAGES = np.arange(0, TRAIN_SIZE)
    else:
        TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)

    train_data = np.zeros((TRAIN_SIZE, IMAGE_SIZE))
    train_answ = np.zeros((TRAIN_SIZE, IMAGE_SIZE))

    i = 0
    for img in TRAIN_IMAGES:

        I = np.asarray(imageio.imread(os.path.join(train_directory_phone, train_input_list[i])))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_data[i, :] = I

        I = np.asarray(imageio.imread(os.path.join(train_directory_dslr, train_output_list[i])))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_answ[i, :] = I

        i += 1
        if i % 100 == 0:
            print(str(round(i * 100 / TRAIN_SIZE)) + "% done", end="\r")

    return train_data, train_answ
