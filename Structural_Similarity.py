#Author: Ekta Bhojwani
#ECE
#cv2 realNet::dnn
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from PIL import Image
from skimage import data
from skimage.color import rgb2gray
from matplotlib.image import imread
from PIL import Image
import cv2
from skimage.measure import compare_ssim as ssim
width = 640
height = 512
dim = (640, 512)
d=0
import imutils
from PIL import ImageChops

paths= ('/home/ekta/Documents/Ekta_Research/')
thermalFiles = os.listdir(paths+'Thermal Images')
areialFiles = []

for filesJPG in os.listdir(paths+'Aerial Images'):
    if filesJPG.endswith(".JPG"):
        areialFiles.append(filesJPG)

for eachThermal in thermalFiles:
    for eachAerial in areialFiles:
    #thermal Image Processing
        thermalImagePath = os.path.join(paths+'Thermal Images',eachThermal)
        img_thermal = cv2.imread(thermalImagePath)
        #img_thermal_resize = img_thermal.resize(dim,resample=0)
        img_thermal_resize = cv2.resize(img_thermal, dim, interpolation= cv2.INTER_AREA)
        #img_thermal_gray = img_thermal_resize.convert('LA')
        img_thermal_gray = cv2.cvtColor(img_thermal_resize, cv2.COLOR_BGR2GRAY)
        #Aerial Image
        aerialImagePath = os.path.join(paths+'Aerial Images',eachAerial)
        img_aerial = cv2.imread(aerialImagePath)
        #img_aerial_resize = img_aerial.resize(dim,resample=0)
        img_aerial_resize = cv2.resize(img_aerial, dim, interpolation= cv2.INTER_AREA)
        #img_aerial_gray = img_aerial_resize.convert('LA')
        img_aerial_gray = cv2.cvtColor(img_aerial_resize, cv2.COLOR_BGR2GRAY)
        #compared_images = ssim(img_thermal_gray, img_aerial_gray)
        #if compared_images >= 0.70:
            #print("ssim:", compared_images) 
            #absdiff = cv2.absdiff(img_thermal_gray, img_aerial_gray)
        #cv2.imwrite("paths/Different_images",compared_images))
        (score, diff) = ssim(img_thermal_gray, img_aerial_gray, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img_thermal_gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img_aerial_gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
# show the output images
            cv2.imshow("Original", img_thermal_gray)
            cv2.imshow("Modified", img_aerial_gray)
            cv2.imshow("Diff", diff)
            cv2.imshow("Thresh", thresh)
            cv2.waitKey(0)

        #cv2.imwrite(os.path.join(paths+'Different_images_2nd_run' , 'abs_%d.jpg'%d), absdiff)
        #d+=1
        #cv2.waitKey(0)
        #cv2.imwrite(os.path.join(paths+'Different_images',compared_images))
        #print(max(compared_images))


#def resized(res_img,  )
#
