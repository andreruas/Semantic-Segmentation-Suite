import cv2
import os, os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


print (cv2.__version__)

imageDir = "datasets/" #specify your path here
image_path_list = []
valid_image_extensions = ".png" #specify your image extension here

#this will loop through all files in imageDir
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))

for imagePath in image_path_list:
    img = cv2.imread(imagePath)
    #imagePath2 = imagePath[:-8] + ".jpg"
    #img2 = cv2.imread(imagePath2)
    if img is None:
        continue

    #put whatever actions you want to do to each image here
    
    #cropping images
    #img[y:y_end, x:x_end]
    img1 = img[15:530, 15:1000]

    #imagePath2 = imagePath[:-4] + "_crop.png"

    #saving image with the same name
    cv2.imwrite(imagePath, img1)
    #cv2.imwrite(imagePath2, img2)

    #displaying images as they are cropped
    #cv2.imshow(imagePath, img)
    #cv2.imshow(imagePath2, img2)
    #key = cv2.waitKey(0)


    #if key == 27: # escape
    #    break

#cv2.destroyAllWindows()
