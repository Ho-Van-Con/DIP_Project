# add library
import cv2              
import numpy as np
import argparse
import pytesseract
import imutils

from imutils.object_detection import non_max_suppression


#------------------------------------------------------------- GET IMAGE FROM COMMAND LINE -------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image")
#ap.add_argument("-m", "--method", required=True, help="Sorting method")
args = vars(ap.parse_args())

#------------------------------------------------------------- READ, RESIZE AND SHOW ORIGINAL IMAGE --------------------------------------------
image = cv2.imread(args["image"])
h,w = image.shape[:2]   # find height and width of image
width = int(1000)
height = int(width * (h/w))
image = cv2.resize(src=image, dsize=(width,height))
cv2.imshow("Original Image", image)
cv2.waitKey(0)  # pause execution until any key pressed

#------------------------------------------------------------- PREPROCESSING IMAGE -------------------------------------------------------------
# convert to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)   

# blurring: Gaussian Filter
blurred = cv2.GaussianBlur(gray, (3,3), 0) 

# binarization
(ret,bin_img) = cv2.threshold(gray,130,255,cv2.THRESH_BINARY_INV)   

# erode to remove noise
kernel = np.ones((3,3),np.uint8)
bin_img = cv2.erode(bin_img,kernel)

#kernel = np.ones((3,3),np.uint8)
#bin_img = cv2.dilate(bin_img,kernel)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
#bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
cv2.imshow("Binary Image",bin_img)
cv2.waitKey(0)


#-----------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------- EXTRACT PHOTO ID ------------------------------------------------------
# cropp the area that contains the image
x=int(0)
y=int(0)
h=int(height)
w=int(width/3.4)
cropped_img = image[y:y+h,x:x+w,:]
#cv2.imshow("Cropped Image",cropped_img)
#cv2.waitKey(0)

# convert to grayscale
gray1 = cv2.cvtColor(cropped_img,cv2.COLOR_RGB2GRAY)

# blurring: Gaussian Filter
blurred = cv2.GaussianBlur(gray, (3,3), 0) 

# binarization
(ret,bin_img1) = cv2.threshold(gray1,130,255,cv2.THRESH_BINARY_INV)  

# fill hole by closing
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
bin_img1 = cv2.morphologyEx(bin_img1,cv2.MORPH_CLOSE,kernel1)

#cv2.imshow("Image2",bin_img)
#cv2.waitKey(0)

# find photo id 
(__,contours, __) = cv2.findContours(bin_img1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for (i,c) in enumerate(contours):
    area = cv2.contourArea(c)
    if area>20000:

        (x,y,w,h) = cv2.boundingRect(c)
        #cv2.rectangle(cropped_img, (x,y), (x+w,y+h), (0,255,0), 2)    #>>> for drawing rect contour

# display photo_id
photo_id = cropped_img[y:y+h,x:x+w,:]
cv2.imshow("photo_id",photo_id)
cv2.waitKey(0)

# save photo_id
name = "photo_id.jpg"
cv2.imwrite(name,photo_id)
#-----------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- EXTRACT ID NUMBER ---------------------------------------------------------------





#------------------------------------------------------------- finding connected areas in the image --------------------------------------------
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 11))
thresh = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel2)

cv2.imshow("Image2",thresh)
cv2.waitKey(0)

# finding contours of text area
_,contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours = imutils.grab_contours(contours)
#contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
#(contours, boundingBoxes) = sort_contours(contours, method=args["method"])
#for (i, c) in enumerate(contours):
#	draw_contour(image, c, i)
#cv2.imshow("Sorted", image)

for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    if ((h-25)*(h-100)<0 and (w>100) and (y>100)):
    	image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    	if ((y>150) and (y<200)):
    		so = bin_img[y:y+h,x:x+w]
    		cv2.imshow("So",so)
    	if ((y>200) and (y<280)):
    		ten = bin_img[y:y+h,x:x+w]
    		cv2.imshow("Ten",ten)
    	if ((y>300) and (y<390) and (w>100)):
    		ngaysinh = bin_img[y:y+h,x:x+w]
    		cv2.imshow("NgaySinh",ngaysinh)
    	if ((y>380) and (y<450) and (h>20)):
    		nguyenquan_1 = bin_img[y:y+h,x:x+w]
    		cv2.imshow("nguyenquan_1",nguyenquan_1)
    	if ((y>440) and (y<500)):
    		nguyenquan_2 = bin_img[y:y+h,x:x+w]
    		cv2.imshow("nguyenquan_2",nguyenquan_2)
    	if ((y>500) and (y<550) and (h>20)):
    		thuongtru_1 = bin_img[y:y+h,x:x+w]
    		cv2.imshow("thuongtru_1",thuongtru_1)
    	if ((y>550) and (y<600)):
    		thuongtru_2 = bin_img[y:y+h,x:x+w]
    		cv2.imshow("thuongtru_2",thuongtru_2)
#cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow("Image3",image)
cv2.waitKey(0)








