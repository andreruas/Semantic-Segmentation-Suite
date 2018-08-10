import cv2
import numpy as np
import os, os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statistics

#--------------- HELPER FUNCTIONS ---------------------------------------------------------------------------------------#

def is_surrounded(c, img, img_copy, sensitivity):
    lc = find_left_color(c, img, img_copy)
    rc = find_right_color(c, img, img_copy)
    tc = find_top_color(c, img, img_copy)
    bc = find_bottom_color(c, img, img_copy)

    if (cv2.contourArea(c) < sensitivity): # Preventing rider removal
        #if you find an edge, replace edge color with color of opposite edge
        if ((lc==img_copy[0,0]).all()):
            lc = rc
        if ((rc==img_copy[0,0]).all()):
            rc = lc
        if ((tc==img_copy[0,0]).all()):
            tc = bc
        if ((bc==img_copy[0,0]).all()):
            bc = tc

    t_f = ((lc==rc).all() and (rc==tc).all() and (tc==bc).all())

    if (t_f): #all colors are the same
        output_color = (int(lc[0]), int(lc[1]), int(lc[2]))
        return output_color

    return (0,0,255)

def find_left_color(c, img, img_copy):
    leftmost = tuple(c[c[:,:,0].argmin()][0]) #leftmost[0] is X coordinate
    if (leftmost[0] == 0): #checking x coordinate
        return img_copy[0,0]
    #print("Left Color is: " + str(img[leftmost[1],leftmost[0]-1]))
    return img[leftmost[1],leftmost[0]-1] #img indexing goes img[y,x]

def find_right_color(c, img, img_copy):
    rightmost = tuple(c[c[:,:,0].argmax()][0])
    #print("Max X value: " + str(img.shape[1]))
    if (rightmost[0] >= img.shape[1]-1):
        return img_copy[0,0]
    #print("Right Color is: " + str(img[rightmost[1],rightmost[0]+1]))
    return img[rightmost[1],rightmost[0]+1] #img indexing goes img[y,x]

def find_bottom_color(c, img, img_copy):
    bottommost = tuple(c[c[:,:,1].argmax()][0])
    #print("Max Y value: " + str(img.shape[0]))
    if (bottommost[1] >= img.shape[0]-1):
        return img_copy[0,0]
    #print("Bottom Color is: " + str(img[bottommost[1]+1,bottommost[0]]))
    return img[bottommost[1]+1,bottommost[0]] #img indexing goes img[y,x]

def find_top_color(c, img, img_copy):
    topmost = tuple(c[c[:,:,1].argmin()][0])
    if (topmost[1] == 0):
        return img_copy[0,0]
    #print("Top Color is: " + str(img[topmost[1]-1,topmost[0]]))
    return img[topmost[1]-1,topmost[0]] #img indexing goes img[y,x]

def remove_contours_surrounded(contours, img, img_copy, mask, sensitivity):
    for c in contours:
        # if the contour is bad, draw it on the mask in blue
        output_color = is_surrounded(c,img,img_copy, sensitivity)
        #if (cv2.contourArea(c) < sensitivity): # TODO: this is a band-aid solution to stop rider removal
        if (output_color != (0,0,255)):
            cv2.drawContours(mask, [c], -1, output_color, -1)
    return mask

def remove_contours_sensitivity(contours, img, img_copy, mask, sensitivity):
    for c in contours:
        if (cv2.contourArea(c) < sensitivity):
                cv2.drawContours(mask, [c], -1, (0,0,0), -1)
    return mask

def remove_contours_small(contours, img, img_copy, mask):
    for c in contours:
        area = cv2.contourArea(c)
        if (area < sensitivity):
            cv2.drawContours(mask, [c], -1, (0,0,255), -1)
    return mask

def remove_contours_edge(contours, img, img_copy, mask, sensitivity):
    if (sensitivity == sens_purple): #getting rid of purple below water
        for c in contours:
            top_color = find_top_color(c, img, img_copy)
            if (top_color == img_copy[0,1]).all():
                cv2.drawContours(mask, [c], -1, (0,255,0), -1)
    return mask

def on_horiz(c, img, img_copy):
    tc = find_top_color(c, img, img_copy)
    bc = find_bottom_color(c, img, img_copy)
    blue_on_top = False
    green_on_bot = False

    if ((tc == img_copy[0,2]).all()):
        blue_on_top = True
    if ((bc == img_copy[0,1]).all()):
        green_on_bot = True

    if (blue_on_top and green_on_bot):
        return True

    return False

#removing purple land that is smaller than a certain sensitivity and replacing it with
def remove_false_positives(contours, img, img_copy, mask, sensitivity):
    for c in contours:
        area = cv2.contourArea(c)
        if (area < sensitivity):
            if (on_horiz(c, img, img_copy)):
                cv2.drawContours(mask, [c], -1, (0,255,0), -1)
                #print("Removed false positive.")
    return mask

#finding the horizon line using green mask (bottom to top)
def find_horiz_bt(green_mask, x,img):
    y = img.shape[0]-5
    while (y > 0):
        color = green_mask[y,x]
        if (green_mask[y,x] == 0):
            #print("Transition at x = " + str(x) + ", y = " + str(y))
            return y
        y = y - 1

#finding the horizon line using green mask (top to bottom)
def find_horiz(green_mask, x,img):
    y = 10
    while (y < img.shape[0]-5):
        color = green_mask[y,x]
        if (green_mask[y,x] == 255):
            #print("Transition at x = " + str(x) + ", y = " + str(y))
            return y
        y = y + 1
    return -1

def find_y_points(green_mask2, x_points,img):
    y_points = []
    for x in x_points:
        y = find_horiz(green_mask2,x,img)
        y_points.append(y)
    #print(x_points)
    #print(y_points)
    return y_points

def find_velocities(y_points):
    velocities = []
    for i in range(0,len(y_points)-1):
        v = abs(y_points[i] - y_points[i+1])
        velocities.append(v)
    #print(velocities)
    return velocities

def plot_lines(x_points,y_points,velocities,thresh,mask,color):
    for i in range(0,len(x_points)-1):
        y1 = y_points[i]
        y2 = y_points[i+1]
        x1 = x_points[i]
        x2 = x_points[i+1]
        v1 = velocities[i]
        if(abs(x2-x1) > 400):
            thresh = thresh * 8
        if((y1 != -1) and (y2 != -1) and (v1 < thresh)):
             cv2.line(mask,(x1, y1),(x2, y2),color,2)

#--------------- VARIABLES ---------------------------------------------------------------------------------------#
sens_purple = 501 #these values also serve as identifiers for the color, so they must be different
sens_purple_horiz = 1001
sens_blue = 10002
sens_green = 5003
sens_black = 5004

black  = [0,   0,   0  ]
blue   = [255, 128, 0  ]
purple = [255, 0,   128]
green  = [0,   255, 0  ]
red    = [0,   0,   255]

def ProcessImageMARS(img,removal,args_post_processing):
    if(removal > 0):
        img[1:removal,:] = blue #replacing the top of the image with static sky mask
    mask = img

    img_copy = img
    img_copy[0,0] = [0,0,255]
    img_copy[0,1] = green
    img_copy[0,2] = blue

    # TODO: These multiple loop methods could overlap, as contours aren't being removed from the list
    #------------------REMOVE BLACK--------------------------------------------------------------#
    lower_black = np.array(black, dtype = "uint16")
    black_mask = cv2.inRange(img, lower_black, lower_black)
    im2, contours, hierarchy = cv2.findContours(black_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print("Removing Black Contours...")
    mask = remove_contours_surrounded(contours, img, img_copy, mask, sens_black)
    mask = remove_false_positives(contours, img, img_copy, mask, sens_black)

    #------------------REMOVE BLUE--------------------------------------------------------------#
    lower_blue = np.array(blue, dtype = "uint16")
    blue_mask = cv2.inRange(img, lower_blue, lower_blue)
    im2, contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print("Removing Blue Contours...")
    mask = remove_contours_surrounded(contours, img, img_copy, mask, sens_blue)

    #------------------REMOVE PURPLE--------------------------------------------------------------#
    lower_purple = np.array(purple, dtype = "uint16")
    purple_mask = cv2.inRange(img, lower_purple, lower_purple)
    im2, contours, hierarchy = cv2.findContours(purple_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print("Removing Purple Contours...")
    mask = remove_contours_surrounded(contours, img, img_copy, mask, sens_purple)
    mask = remove_contours_edge(contours, img, img_copy, mask, sens_purple)
    mask = remove_false_positives(contours, img, img_copy, mask, sens_purple_horiz)

    #------------------REMOVE GREEN--------------------------------------------------------------#
    lower_green = np.array(green, dtype = "uint16")
    green_mask = cv2.inRange(img, lower_green, lower_green)
    im2, contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print("Removing Green Contours...")
    mask = remove_contours_surrounded(contours, img, img_copy, mask, sens_green)


    #------------------DRAW HORIZON--------------------------------------------------------------#
    if (args_post_processing == 4):
        green_mask2 = cv2.inRange(mask, lower_green, lower_green)
        x_points = [0,50,100,150,200,250,300,350,400,img.shape[1]-50,img.shape[1]-1]
        y_points = find_y_points(green_mask2, x_points,img)
        velocities = find_velocities(y_points)

        #print(velocities)
        median = statistics.median(velocities)

        thresh = median * 3
        if(thresh < 20):
           thresh = 20
        color = (0,255,255)
        #print(thresh)

        plot_lines(x_points,y_points,velocities,thresh,mask,color)

    return mask






def ProcessImageRail(img,args_post_processing, args_rail_sens):
    #------------------REMOVE RED --------------------------------------------------------------#

    img_copy = img
    img_copy[0,0] = [0,0,255]
    img_copy[0,1] = green
    img_copy[0,2] = blue
    mask = img

    lower_red = np.array(red, dtype = "uint16")
    red_mask = cv2.inRange(img, lower_red, lower_red)
    im2, contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    mask = remove_contours_sensitivity(contours, img, img_copy, mask, args_rail_sens)

    return mask
















    ##
