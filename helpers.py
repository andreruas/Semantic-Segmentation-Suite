import cv2
import numpy as np
import itertools
import operator
import os, csv
import tensorflow as tf
import random
import time, datetime

import utils

##----------------- DATA AUGMENTATION HELPER FUNCTIONS -------------------------------------------------------------------##

def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)     # convert from BGR-->HSV
    h, s, v = cv2.split(hsv)

    lim = 255 - value #limiting overflow
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def decrease_brightness(img, value):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert from BGR-->HSV
    value = abs(value) # only allow positive values
    factor = abs(1-value/100)
    hsvImg[...,2] = hsvImg[...,2] * factor

    img = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    return img

def blur_single_circle(image,x,y,w,blur):
    # create a temp image and a mask to work on
    tempImg = image.copy()
    maskShape = (image.shape[0], image.shape[1], 1)
    mask = np.full(maskShape, 0, dtype=np.uint8)
    w = int(w)
    h = w

    tempImg [y-h:y+h, x-w:x+w] = cv2.blur(tempImg [y-h:y+h, x-w:x+w] ,(blur,blur))
    center = (x,y)
    cv2.circle(tempImg , center, h, (255), 1)
    cv2.circle(mask , center, h, (255), -1)

    # oustide of the loop, apply the mask and save
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(image,image,mask = mask_inv)
    img2_fg = cv2.bitwise_and(tempImg,tempImg,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    return dst

def blur_circle(image,x,y,w,blur):
    #blurring several circles so it doesn't have a distinct edge
    dst = blur_single_circle(image,x,y,w*.4,blur)
    dst = blur_single_circle(dst,x,y,w*.6,int(blur*.7))
    dst = blur_single_circle(dst,x,y,w*.8,int(blur*.5))
    dst = blur_single_circle(dst,x,y,w,int(blur*.2))
    return dst

def blur_circle_rand(image, drops):
    blur = 10
    drops = random.randint(0,drops)
    if (drops == 0):
        return image

    for i in range(0,drops):
        w = random.randint(15,40)
        x = random.randint(w+1,image.shape[1]-w-1)
        y = random.randint(w+1,image.shape[0]-w-1)
        dst = blur_circle(image,x,y,w,blur)
        image = dst

    return dst

##------------------------------------------------------------------------------------------------------------------------##

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy

    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

##------------------------------------------------------------------------------------------------------------------------##
## The below functions were moved out of main to de-clutter that file

# Get a list of the training, validation, and testing file paths
def prepare_data(dataset_dir):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names

def data_augmentation(input_image, output_image, crop_height, crop_width, h_flip, v_flip, droplets, brightness, rotation, ):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, crop_height, crop_width)

    if h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if (int(droplets) > 0):
        input_image = blur_circle_rand(input_image,int(droplets)) #adding simulated droplets
    if brightness:
        value = int(random.uniform(-1.0*brightness, brightness)*100)
        if value > 0:
            input_image = increase_brightness(input_image, abs(value))
        if value < 0:
            input_image = decrease_brightness(input_image, abs(value))
    if rotation:
        angle = random.uniform(-1*rotation, rotation)
    if rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image
