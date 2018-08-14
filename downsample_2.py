import cv2
import os, os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

imageDir = "short_64" #specify your path here
image_path_list = []
valid_image_extensions = ".png" #specify your image ext here

#this will loop through all files in imageDir
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))

for imagePath in image_path_list:
    print(imagePath)
    img = cv2.imread(imagePath)

    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imwrite(imagePath, resized)
