#Author : Ekta Bhojwani

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
aerialFiles = []
images = []
for filesJPG in os.listdir(paths+'Aerial Images'):
    if filesJPG.endswith(".JPG"):
        aerialFiles.append(filesJPG)

for eachThermal in thermalFiles:
    for eachAerial in aerialFiles:
    #thermal Image 
        thermalImagePath = os.path.join(paths+'Thermal Images',eachThermal)
        img_thermal = cv2.imread(thermalImagePath)
        img_thermal_resize = cv2.resize(img_thermal, dim, interpolation= cv2.INTER_AREA)
        img_thermal_gray = cv2.cvtColor(img_thermal_resize, cv2.COLOR_BGR2GRAY)
        #Aerial Image
        aerialImagePath = os.path.join(paths+'Aerial Images',eachAerial)
        img_aerial = cv2.imread(aerialImagePath)
        img_aerial_resize = cv2.resize(img_aerial, dim, interpolation= cv2.INTER_AREA)
        img_aerial_gray = cv2.cvtColor(img_aerial_resize, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Image_%d"%d, img_aerial_gray)
        d+=1
        #def compare_image():
        compared_images = ssim(img_thermal_gray, img_aerial_gray)
                #for (img_thermal_gray).astype("uint8"), (img_aerial_gray).astype("uint8") in compared_images:
        
        if compared_images >= 0.55:
            #cv2.imshow('img_therm', img_thermal_gray)
            print("ssim:", compared_images)
                #cv2.imshow
        
                   # cv2.imwrite(os.path.join(paths+'Different_images_2nd_run' , 'therm_%d.jpg'%d, 'aerial_%d'%d), img_thermal_gray, img_aerial_gray)
        #return()
        fig = plt.figure("Therma_gray, gray_Aerial, difference")
     
        #ax1 = fig.add_subplot(1, 3, 1)
        

        absdiff = cv2.subtract(img_thermal_gray, img_aerial_gray)
        #images = ("First_image", img_thermal_gray), ("Second_image", img_aerial_gray), ("Third_image", absdiff)
        #for (i, (name, image)) in enumerate(images):
            #ax= fig.add_subplot(1, 3, i+1)
            #plt.imshow(image, cmap = plt.cm.gray)
            #plt.axis("off")
        #plt.show(); 
        #fig.savefig(os.path.join(paths+'Plots', 'images_%d.png'%d))    
        #fig, (ax1, ax2)
        cv2.imwrite(os.path.join(paths+'Different_images_2nd_run' , 'abs_%d.jpg'%d),absdiff)
        d+=1
        cv2.waitKey(0)
        #cv2.imwrite(os.path.join(paths+'Different_images',compared_images))
        #print(max(compared_images))



